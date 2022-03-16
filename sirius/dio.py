#   Copyright 2019 AUI, Inc. Washington DC, USA
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import xarray as xr
import dask.array as da
from astropy.timeseries import TimeSeries
from astropy.time import Time
from astropy import units as u
from collections import Counter
import time
import dask
import os
import shutil
from sirius_data._constants import pol_codes_RL, pol_codes_XY, pol_str
from sirius._sirius_utils._array_utils import _is_subset, _calc_baseline_indx_pair
from sirius._sirius_utils._cngi_io import read_ms, write_ms
import daskms


def make_time_xda(
    time_start="2019-10-03T19:00:00.000", time_delta=3600, n_samples=10, n_chunks=4
):
    """
    Create a time series xarray array.
    Parameters
    ----------
    -------
    time_xda : xarray.DataArray
    """
    ts = np.array(
        TimeSeries(
            time_start=time_start, time_delta=time_delta * u.s, n_samples=n_samples
        ).time.value
    )
    chunksize = int(np.ceil(n_samples / n_chunks))
    time_da = da.from_array(ts, chunks=chunksize)
    print("Number of chunks ", len(time_da.chunks[0]))

    time_xda = xr.DataArray(
        data=time_da, dims=["time"], attrs={"time_delta": float(time_delta)}
    )

    return time_xda


def make_chan_xda(
    spw_name="sband",
    freq_start=3 * 10**9,
    freq_delta=0.4 * 10**9,
    freq_resolution=0.01 * 10**9,
    n_channels=3,
    n_chunks=3,
):
    """
    Create a channel frequencies xarray array.
    Parameters
    ----------
    -------
    chan_xda : xarray.DataArray
    """
    freq_chan = (np.arange(0, n_channels) * freq_delta + freq_start).astype(
        float
    )  # astype(float) needed for interfacing with CASA simulator.
    chunksize = int(np.ceil(n_channels / n_chunks))
    chan_da = da.from_array(freq_chan, chunks=chunksize)
    print("Number of chunks ", len(chan_da.chunks[0]))

    chan_xda = xr.DataArray(
        data=chan_da,
        dims=["chan"],
        attrs={
            "freq_resolution": float(freq_resolution),
            "spw_name": spw_name,
            "freq_delta": float(freq_delta),
        },
    )
    return chan_xda


def write_to_ms_daskms(
    vis_xds,
    time_xda,
    chan_xda,
    pol,
    tel_xds,
    phase_center_names,
    phase_center_ra_dec,
    auto_corr,
    save_parms,
):
    """
    Write out a MeasurementSet to disk using dask-ms

    Parameters
    ----------
    vis_xds : xarray.Dataset
    time_xda : xarray.DataArray
    chan_xda : xarray.DataArray
    pol : list
    tel_xds : xarray.Dataset
    phase_center_names : numpy.array
    phase_center_ra_dec : numpy.array
    auto_corr : bool
    save_parms : dict
    -------
    xarray.Dataset
    """

    # n_time, n_baseline, n_chan, n_pol = vis_xds.DATA.shape
    # ant_pos = tel_xds.ANT_POS.values
    ms_table_name = save_parms["ms_name"]

    ### using simple_sim3.ms as an output template

    # creating skeleton of the new MS, deleting if already exists
    try:
        os.remove(ms_table_name)
    except IsADirectoryError:
        shutil.rmtree(ms_table_name)
    except FileNotFoundError:
        pass

    ### Building first graph, the main table

    # master list to contain datasets for writing into the MS
    datasets = []

    # define a chunking schema
    n_row = vis_xds.sizes["time"] * vis_xds.sizes["baseline"]
    n_chan = vis_xds.sizes["chan"]
    n_pol = vis_xds.sizes["pol"]
    #chunks = {"row": (n_row,), "chan": (n_chan,), "corr": (n_pol,), "uvw": (3,)}
    
    print('dims',n_row,n_chan,n_pol)

    # This code will most probably be moved into simulation if we get rid of row time baseline split.
    vis_data_reshaped = vis_xds.DATA.data.reshape((n_row, n_chan, n_pol))
    uvw_reshaped = vis_xds.UVW.data.reshape((n_row, 3))
    weight_reshaped = vis_xds.WEIGHT.data.reshape((n_row, n_pol))
    sigma_reshaped = vis_xds.SIGMA.data.reshape((n_row, n_pol))
    
    chunks = (vis_data_reshaped.chunks[0][0],vis_data_reshaped.chunks[1][0],vis_data_reshaped.chunks[2][0])
    print('chunks',vis_data_reshaped.chunks)
    print(vis_data_reshaped.chunksize)
    print(vis_data_reshaped.chunks[0][0],vis_data_reshaped.chunks[1][0],vis_data_reshaped.chunks[2][0])
    chunks = vis_data_reshaped.chunks
    print('****',chunks)
    print('vis_data_reshaped',vis_data_reshaped)
    chunks = {"row": (vis_data_reshaped.chunks[0][0],), "chan": (vis_data_reshaped.chunks[1][0],), "corr": (vis_data_reshaped.chunks[2][0],), "uvw": (3,)}

    # generate an antenna index for each time step
    ant1_arr = da.from_array(np.array([], dtype="int32"))
    ant2_arr = da.from_array(np.array([], dtype="int32"))
    for tt in range(0, vis_xds.sizes["time"]):
        ant1, ant2 = _calc_baseline_indx_pair(tel_xds.sizes["ant_name"], auto_corr)
        ant1_arr = da.append(ant1_arr, ant1)
        ant2_arr = da.append(ant2_arr, ant2)
        ant1s = ant1_arr.rechunk(chunks=chunks["row"])
        ant2s = ant2_arr.rechunk(chunks=chunks["row"])

    # we run this function on only a single DDI at a time
    ddid = da.zeros(n_row, chunks=chunks["row"], dtype="int32")

    # don't flag any of the data yet
    flags = da.zeros_like(vis_data_reshaped, dtype=bool)
    flag_rows = da.zeros_like(ddid, dtype=bool)
    # can we get away with not specifying flag_category ([0,0,0 Boolean])?

    # currently don't support subarrays, so only one array ID assigned
    array_ids = da.zeros_like(ddid, dtype="int32")

    # fill with input in units of the input array, which we expect to be SI (s)
    exposures = da.full_like(ddid, time_xda.time_delta, dtype="float64")
    # interval maps to exposure in perfect simulation conditions
    intervals = exposures

    # not supporting different feed types
    feeds = da.zeros_like(ddid, "int32")

    # index the strings in phase_center_names (a function of the time dimension)
    field_index = da.from_array(np.unique(phase_center_names, return_index=True)[1])
    field_ids = da.repeat(field_index, (ddid.size // field_index.size))

    # this function is also only run for a single observation at once
    observation_ids = da.zeros_like(ddid)

    # currently don't support group processing
    processor_ids = da.zeros_like(ddid)

    # WIP: since it doesn't affect data can be 0s for now, function tbc later to derive from time_xda
    scan_numbers = da.ones_like(ddid)

    # unsupported - table for semi-obscure calibration indexing (e.g., temperature loads for solar)
    state_ids = da.zeros_like(ddid)

    # fill time col input object explicitly match row chunking, expect units in SI (s)
    times = da.repeat(time_xda.data, repeats=vis_xds.sizes["baseline"]).rechunk(
        chunks=chunks["row"]
    )
    # this gave us an array of strings, but we need seconds since epoch in float to cram into the MS
    # convert to datetime64[ms], ms since epoch, seconds since epoch, then apply correction
    # NB: difference between Unix origin (1970-01-01) and what CASA expects (1858-11-17) is +/-3506716800 seconds
    times = times.astype(np.datetime64).astype(float) / 10**3 + 3506716800.0

    # match the time column for now, ephemeris support can come later
    time_centroids = times

    # only fill the data and model columns to ensure fair comparison between write times
    empty_data_column = da.zeros_like(vis_data_reshaped)
   
    datasets.append(
        daskms.Dataset(
            {
                "DATA": (
                    ("row", "chan", "corr"),
                    vis_data_reshaped.astype("complex"),
                ),
                "MODEL_DATA": (
                    ("row", "chan", "corr"),
                    empty_data_column.astype("complex"),
                ),
                "CORRECTED_DATA": (
                    ("row", "chan", "corr"),
                    vis_data_reshaped.astype("complex"),
                ),
                "FLAG": (("row", "chan", "corr"), flags.astype("bool")),
                "UVW": (("row", "uvw"), uvw_reshaped.astype("float")),
                "SIGMA": (("row", "pol"), sigma_reshaped.astype("float")),
                "WEIGHT": (("row", "pol"), weight_reshaped.astype("float")),
                "FLAG_ROW": (("row"), flag_rows.astype("bool")),
                "DATA_DESC_ID": (("row"), ddid.astype("int")),
                "ANTENNA1": (("row"), ant1s.astype("int")),
                "ANTENNA2": (("row"), ant2s.astype("int")),
                "ARRAY_ID": (("row"), array_ids.astype("int")),
                "EXPOSURE": (("row"), exposures.astype("float")),
                "FEED1": (("row"), feeds.astype("int")),
                "FEED2": (("row"), feeds.astype("int")),
                "FIELD_ID": (("row"), field_ids.astype("int")),
                "INTERVAL": (("row"), intervals.astype("float")),
                "OBSERVATION_ID": (("row"), observation_ids.astype("int")),
                "PROCESSOR_ID": (("row"), processor_ids.astype("int")),
                "SCAN_NUMBER": (("row"), scan_numbers.astype("int")),
                "STATE_ID": (("row"), state_ids.astype("int")),
                "TIME": (("row"), times.astype("float")),
                "TIME_CENTROID": (("row"), time_centroids.astype("float")),
                #'WEIGHT_SPECTRUM': (("row","chan","pol"), weight_spectrum_reshaped),
            }
        )
    )
    
    print('datasets',datasets)

    ### then, pass or construct the arrays needed to populate each subtable

    # ANTENNA
    ant_subtable = []
    ds = daskms.Dataset(
        data_vars=dict(
            NAME=(
                ("row"),
                da.from_array(tel_xds.ant_name.data, chunks=tel_xds.dims["ant_name"]),
            ),
            DISH_DIAMETER=(("row"), tel_xds.DISH_DIAMETER.data),
            POSITION=(("row", "xyz"), tel_xds.ANT_POS.data),
            # not yet supporting space-based interferometers
            TYPE=(
                ("row"),
                da.full(tel_xds.ant_name.shape, "GROUND-BASED", dtype="<U12"),
            ),
            FLAG_ROW=(("row"), da.zeros(tel_xds.ant_name.shape, dtype="bool")),
            # when this input is available from tel.zarr then we can infer it, til then assume alt-az
            MOUNT=(("row"), da.full(tel_xds.ant_name.shape, "alt-az", dtype="<U6")),
            # likewise, although this seems like it should be pulled from the cfg files
            STATION=(("row"), da.full(tel_xds.ant_name.shape, "P", dtype="<U1")),
            # until we have some input with OFFSET specified, no conditional
            OFFSET=(
                ("row", "xyz"),
                da.zeros((tel_xds.dims["ant_name"], 3), dtype=np.float),
            ),
        )
    )
    ant_subtable.append(ds)

    # DATA_DESCRIPTION
    ddi_subtable = []
    ds = daskms.Dataset(
        data_vars=dict(
            # this function operates on a single DDI at once, so this should reduce to length-1 arrays = 0
            # we could also enumerate the ds list if we were reading from existing MS and pass the index
            SPECTRAL_WINDOW_ID=(("row"), da.zeros(1, dtype="int")),
            FLAG_ROW=(("row"), da.zeros(1, dtype="bool")),
            POLARIZATION_ID=(("row"), da.zeros(1, dtype="int")),
        ),
    )
    ddi_subtable.append(ds)

    # FEED
    if np.all(np.isin(pol, [5, 6, 7, 8])):
        poltype_arr = da.broadcast_to(
            da.asarray(["R", "L"]), (tel_xds.ant_name.size, 2)
            )
    elif np.all(np.isin(pol, [9, 10, 11, 12])):
        # it's clunky to assume linear feeds...
        poltype_arr = da.broadcast_to(
            da.asarray(["X", "Y"]), (tel_xds.ant_name.size, 2)
        )

    '''
    poltype_arr = da.broadcast_to(
            da.asarray(pol.astype(np.int32)), (tel_xds.ant_name.size, len(pol))
        )
    '''
    '''
    poltype_arr = da.broadcast_to(
            da.asarray(pol_str[pol]), (tel_xds.ant_name.size, len(pol))
        )
    '''
        
    
    feed_subtable = []
    ds = daskms.Dataset(
        data_vars=dict(
            ANTENNA_ID=(("row"), da.arange(0, tel_xds.dims["ant_name"], dtype="int")),
            # -1 fill value indicates that we're not using the optional BEAM subtable
            BEAM_ID=(("row"), da.ones(tel_xds.ant_name.shape, dtype="int") * -1),
            INTERVAL=(
                ("row"),
                da.full(tel_xds.dims["ant_name"], fill_value=1e30, dtype="float"),
            ),
            # we're not supporting offset feeds yet
            POSITION=(
                ("row", "xyz"),
                da.zeros((tel_xds.dims["ant_name"], 3), dtype=np.float),
            ),
            # indexed from FEEDn in the MAIN table
            FEED_ID=(("row"), da.zeros(tel_xds.dims["ant_name"], dtype="int")),
            # "Polarization reference angle. Converts into parallactic angle in the sky domain."
            RECEPTOR_ANGLE=(
                ("row", "receptors"),
                da.zeros((tel_xds.dims["ant_name"], poltype_arr.shape[1])),
            ),
            # "Polarization response at the center of the beam for this feed expressed
            # in a linearly polarized basis (e→x,e→y) using the IEEE convention."
            # practically, broadcast a POLxPOL complex identity matrix along a new N_antenna dim
            POL_RESPONSE=(
                ("row", "receptors", "receptors-2"),
                da.broadcast_to(
                    da.eye(poltype_arr.shape[1], dtype="complex"),
                    (tel_xds.dims["ant_name"], poltype_arr.shape[1], poltype_arr.shape[1]),
                ),
            ),
            # A value of -1 indicates the row is valid for all spectral windows
            SPECTRAL_WINDOW_ID=(
                ("row"),
                da.ones(tel_xds.dims["ant_name"], dtype="int") * -1,
            ),
            NUM_RECEPTORS=(
                ("row"),
                da.full(tel_xds.dims["ant_name"], fill_value=poltype_arr.shape[1], dtype="int"),
            ),
            POLARIZATION_TYPE=(("row", "receptors"), poltype_arr),
            # "the same measure reference used for the TIME column of the MAIN table must be used"
            # in practice this appears to be 0 since the conversion to casa-expected epoch is done
            TIME=(("row"), da.zeros(tel_xds.dims["ant_name"], dtype="float")),
            # "Beam position oﬀset, as deﬁned on the sky but in the antenna reference frame."
            # the third dimension size could also be taken from phase_center_ra_dec in theory
            BEAM_OFFSET=(
                ("row", "receptors", "radec"),
                da.zeros(shape=(tel_xds.dims["ant_name"], poltype_arr.shape[1], 2), dtype="float"),
            ),
        ),
    )
    feed_subtable.append(ds)

    # FLAG_CMD
    # we're not flagging our sim so this subtable has no rows

    # FIELD
    field_subtable = []
    ds = daskms.Dataset(
        data_vars=dict(
            NAME=(("row"), da.array(phase_center_names)),
            SOURCE_ID=(("row"), da.indices(phase_center_names.shape)[0]),
            # may need to wrap the RA at 180deg to make the MS happy
            REFERENCE_DIR=(
                ("row", "field-poly", "field-dir"),
                # expand_dims was added to dask.array in version 2022.02.0
                da.expand_dims(da.array(phase_center_ra_dec, dtype="double"), axis=0),
            ),
            PHASE_DIR=(
                ("row", "field-poly", "field-dir"),
                da.expand_dims(da.array(phase_center_ra_dec, dtype="double"), axis=0),
            ),
            DELAY_DIR=(
                ("row", "field-poly", "field-dir"),
                da.expand_dims(da.array(phase_center_ra_dec, dtype="double"), axis=0),
            ),
            CODE=(
                ("row"),
                da.full(phase_center_names.shape, fill_value="", dtype="<U1").astype(
                    "object"
                ),
            ),
            # "Required to use the same TIME Measure reference as in MAIN."
            # in practice this appears to be 0 since the conversion to casa-expected epoch is done
            TIME=(("row"), da.zeros(phase_center_names.shape, dtype="float")),
            FLAG_ROW=(("row"), da.zeros(phase_center_names.shape, dtype="bool")),
            # Series order for the *_DIR columns
            NUM_POLY=(("row"), da.zeros(phase_center_names.shape, dtype="int")),
        ),
    )
    field_subtable.append(ds)

    # HISTORY
    # the libraries for which we care about providing history don't have __version__
    # using pkg_resources.get_distribution fails for 2/3
    # we don't want to stay pegged to 3.8 (for importlib.metadata)
    # and version numbers seems like the only really useful info downstream
    # it's unclear if populating this subtable is even helpful
    his_subtable = []
    ds = daskms.Dataset(
        data_vars=dict(
            MESSAGE=(
                ("row"),
                da.array(["taskname=sirius.dio.write_to_ms"], dtype="object"),
            ),
            APPLICATION=(("row"), da.array(["ms"], dtype="object")),
            # "Required to have the same TIME Measure reference as used in MAIN."
            # but unlike some subtables with ^that^ in the spec, this is actual timestamps
            # NB: difference between Unix origin (1970-01-01) and what CASA expects (1858-11-17) is +/-3506716800 seconds
            TIME=(
                ("row"),
                (da.array([time.time()], dtype="float") / 10**3 + 3506716800.0),
            ),
            PRIORITY=(("row"), da.array(["NORMAL"], dtype="object")),
            ORIGIN=(("row"), da.array(["dask-ms"], dtype="object")),
            OBJECT_ID=(("row"), da.array([0], dtype="int")),
            OBSERVATION_ID=(("row"), da.array([-1], dtype="int")),
            # The MSv2 spec says there is "an adopted project-wide format."
            # which is big if true... appears to have shape expand_dims(MESSAGE)
            APP_PARAMS=(
                ("row", "APP_PARAMS-1"),
                da.array([[""], [""]], dtype="object").transpose(),
            ),
            CLI_COMMAND=(
                ("row", "CLI_COMMAND-1"),
                da.array([[""], [""]], dtype="object").transpose(),
            ),
        ),
    )
    his_subtable.append(ds)

    # OBSERVATION
    obs_subtable = []
    ds = daskms.Dataset(
        data_vars=dict(
            TELESCOPE_NAME=(
                ("row"),
                da.array([tel_xds.telescope_name], dtype="object"),
            ),
            RELEASE_DATE=(("row"), da.zeros(1, dtype="float")),
            SCHEDULE_TYPE=(("row"), da.array([""], dtype="object")),
            PROJECT=(("row"), da.array(["SiRIUS simulation"], dtype="object")),
            # first and last value
            TIME_RANGE=(
                ("row", "obs-exts"),
                da.array([da.take(times, [0, -1]).astype("float")]),
            ),
            # could try to be clever about this to get uname w/ os or psutil
            OBSERVER=(("row"), da.array(["SiRIUS"], dtype="object")),
            FLAG_ROW=(("row"), da.zeros(1, dtype="bool")),
        ),
    )
    obs_subtable.append(ds)
    
    print(tel_xds.ant_name.size , time_xda.size)
    
    ANTENNA_ID=(
                ("row"),
                da.tile(da.arange(0, tel_xds.ant_name.size), reps=10).rechunk(
                    chunks=tel_xds.ant_name.size * time_xda.size
                ),
            )
            
    print(ANTENNA_ID)

    # POINTING
    pnt_subtable = []
    ds = daskms.Dataset(
        data_vars=dict(
            # is this general enough for the case where phase_center_ra_dec has size > 1 ?
            TARGET=(
                ("row", "point-poly", "radec"),
                da.broadcast_to(
                    da.array(phase_center_ra_dec),
                    shape=(tel_xds.ant_name.size * time_xda.size, 1, 2),
                ),
            ),
            # set time origin for polynomial expansions to beginning of the observation
            TIME_ORIGIN=(
                ("row"),
                da.repeat(
                    da.take(times, [0]), repeats=tel_xds.ant_name.size * time_xda.size
                ),
            ),
            INTERVAL=(
                ("row"),
                da.repeat(
                    da.asarray([time_xda.time_delta]),
                    repeats=tel_xds.ant_name.size * time_xda.size,
                ),
            ),
            # True if tracking the nominal pointing position
            TRACKING=(
                ("row"),
                da.ones(shape=tel_xds.ant_name.size * time_xda.size, dtype="bool"),
            ),
            ANTENNA_ID=(
                ("row"),
                da.tile(da.arange(0, tel_xds.ant_name.size), reps=time_xda.size).rechunk(
                    chunks=tel_xds.ant_name.size * time_xda.size
                ),
            ),
            DIRECTION=(
                ("row", "point-poly", "radec"),
                da.broadcast_to(
                    da.array(phase_center_ra_dec),
                    shape=(tel_xds.ant_name.size * time_xda.size, 1, 2),
                ),
            ),
            # only supporting first order polynomials at present
            NUM_POLY=(
                ("row"),
                da.zeros(shape=tel_xds.ant_name.size * time_xda.size, dtype="int"),
            ),
            # could fill with phase_center_names; the reference implementation is empty
            NAME=(
                ("row"),
                da.full(
                    tel_xds.ant_name.size * time_xda.size, fill_value="", dtype="<U1"
                ).astype("object"),
            ),
            # another different use of this same column name:
            # "Mid-point of the time interval for which the information in this row is valid."
            # NB: difference between Unix origin (1970-01-01) and what CASA expects (1858-11-17) is +/-3506716800 seconds
            TIME=(
                ("row"),
                # must drop from the xr.DataArray to a raw dask.array then make expected shape
                da.repeat(
                    (
                        time_xda.astype(np.datetime64).astype(float) / 10**3
                        + 3506716800.0
                    ).data,
                    repeats=tel_xds.ant_name.size,
                ).rechunk(chunks=tel_xds.ant_name.size * time_xda.size),
            ),
        ),
    )
    pnt_subtable.append(ds)

    # POLARIZATION
    # Surely there is a more elegant way to build this strange index
    pol_index = []
    for pp in pol:
        if pp == 5 or pp == 9:
            pol_index.append([0, 0])
        if pp == 6 or pp == 10:
            pol_index.append([0, 1])
        if pp == 7 or pp == 11:
            pol_index.append([1, 0])
        if pp == 8 or pp == 12:
            pol_index.append([1, 1])

    pol_subtable = []
    ds = daskms.Dataset(
        data_vars=dict(
            NUM_CORR=(("row"), da.asarray([len(pol)], dtype="int")),
            CORR_TYPE=(("row", "corr"), da.asarray([pol], dtype="int")),
            FLAG_ROW=(("row"), da.zeros(shape=1).astype("bool")),
            # "Pair of integers for each correlation product, specifying the receptors from which the signal originated."
            CORR_PRODUCT=(
                ("row", "corr", "corrprod_idx"),
                da.asarray([pol_index], dtype="int"),
            ),
        ),
    )
    pol_subtable.append(ds)

    # PROCESSOR
    # we only support a single processor, thus this subtable will remain empty

    # SPECTRAL_WINDOW
    # this function will be operating on a single DDI and therefore SPW at once
    spw_subtable = []
    ds = daskms.Dataset(
        data_vars=dict(
            FREQ_GROUP=(("row"), da.zeros(shape=1).astype("int")),
            FLAG_ROW=(("row"), da.zeros(shape=1).astype("bool")),
            NET_SIDEBAND=(("row"), da.ones(shape=1).astype("int")),
            # if only everything were consistently indexed...
            # maybe it would be better to use chan_xda.spw_name but that might break something downstream
            FREQ_GROUP_NAME=(
                ("row"),
                da.full(shape=1, fill_value="Group 1", dtype="<U7").astype("object"),
            ),
            # NB: a naive chan_xda.sum() is high by an order of magnitude!
            TOTAL_BANDWIDTH=(
                ("row"),
                da.asarray([chan_xda.freq_delta * chan_xda.size]),
            ),
            # "frequency representative of this spw, usually the sky frequency corresponding to the DC edge of the baseband."
            # until "reference" in chan.xda.attrs use 1st channel
            REF_FREQUENCY=(("row"), da.take(chan_xda.data, [0])),
            # obscure measures tool keyword for Doppler tracking
            MEAS_FREQ_REF=(("row"), da.ones(shape=1).astype("int")),
            # "Identiﬁcation of the electronic signal path for the case of multiple (simultaneous) IFs.
            # (e.g. VLA: AC=0, BD=1, ATCA: Freq1=0, Freq2=1)"
            IF_CONV_CHAIN=(("row"), da.zeros(shape=1).astype("int")),
            NAME=(("row"), da.array([chan_xda.spw_name]).astype("object")),
            NUM_CHAN=(("row"), da.array([chan_xda.size]).astype("int")),
            # the following share shape (1,chans)
            # "it is more efficient to keep a separate reference to this information"
            CHAN_WIDTH=(
                ("row", "chan"),
                da.broadcast_to([chan_xda.freq_delta], shape=(1, chan_xda.size)).astype(
                    "float"
                ),
            ),
            # the assumption that input channel frequencies are central will hold for a while
            CHAN_FREQ=(
                ("row", "chan"),
                da.broadcast_to(
                    da.asarray(chan_xda.data), shape=(1, chan_xda.size)
                ).astype("float"),
            ),
            RESOLUTION=(
                ("row", "chan"),
                da.broadcast_to(
                    # note that this is not what we call chan.xda.freq_resolution
                    [chan_xda.freq_delta],
                    shape=(1, chan_xda.size),
                ).astype("float"),
            ),
            # we may eventually want to infer this by instrument, e.g., ALMA correlator binning
            # but until "effective_bw" in chan_xda.attrs,
            EFFECTIVE_BW=(
                ("row", "chan"),
                da.broadcast_to([chan_xda.freq_delta], shape=(1, chan_xda.size)).astype(
                    "float"
                ),
            ),
        ),
    )
    spw_subtable.append(ds)

    # STATE
    state_subtable = []
    ds = daskms.Dataset(
        data_vars=dict(
            FLAG_ROW=(("row"), da.zeros(shape=1).astype("bool")),
            SIG=(("row"), da.ones(shape=1).astype("bool")),
            CAL=(("row"), da.zeros(shape=1).astype("float")),
            # some subset of observing modes e.g., solar will require this
            LOAD=(("row"), da.zeros(shape=1).astype("float")),
            # reference phase if available
            REF=(("row"), da.zeros(shape=1).astype("bool")),
            # relative to SCAN_NUMBER in MAIN, better support TBD
            SUB_SCAN=(("row"), da.zeros(shape=1).astype("int")),
            OBS_MODE=(
                ("row"),
                da.full(
                    shape=1, fill_value="OBSERVE_TARGET.ON_SOURCE", dtype="<U24"
                ).astype("object"),
            ),
        ),
    )
    state_subtable.append(ds)

    # other subtables, e.g., SOURCE, SYSCAL, and WEATHER are not yet supported!

    # In general we should pass specific values to columns kwargs, but since
    # we deleted any existing file to begin, should be no risk of spurious writes

    # the main table object should be added to the graph first to avoid RuntimeErrors
    print('datasets',datasets)
    ms_writes = daskms.xds_to_table(datasets, save_parms["ms_name"], columns="ALL")

    sub_ant = daskms.xds_to_table(
        ant_subtable,
        "::".join((save_parms["ms_name"], "ANTENNA")),
        columns="ALL",
    )

    sub_ddi = daskms.xds_to_table(
        ddi_subtable,
        "::".join((save_parms["ms_name"], "DATA_DESCRIPTION")),
        columns="ALL",
    )

    sub_feed = daskms.xds_to_table(
        feed_subtable,
        "::".join((save_parms["ms_name"], "FEED")),
        columns="ALL",
    )

    sub_field = daskms.xds_to_table(
        field_subtable,
        "::".join((save_parms["ms_name"], "FIELD")),
        columns="ALL",
    )

    sub_his = daskms.xds_to_table(
        his_subtable,
        "::".join((save_parms["ms_name"], "HISTORY")),
        columns="ALL",
    )

    sub_obs = daskms.xds_to_table(
        obs_subtable,
        "::".join((save_parms["ms_name"], "OBSERVATION")),
        columns="ALL",
    )

    sub_point = daskms.xds_to_table(
        pnt_subtable,
        "::".join((save_parms["ms_name"], "POINTING")),
        columns="ALL",
    )

    sub_pol = daskms.xds_to_table(
        pol_subtable,
        "::".join((save_parms["ms_name"], "POLARIZATION")),
        columns="ALL",
    )

    sub_spw = daskms.xds_to_table(
        spw_subtable,
        "::".join((save_parms["ms_name"], "SPECTRAL_WINDOW")),
        columns="ALL",
    )

    sub_state = daskms.xds_to_table(
        state_subtable,
        "::".join((save_parms["ms_name"], "STATE")),
        columns="ALL",
    )

    ### execute the graphs

    if save_parms["DAG_name_write"]:
        dask.visualize(ms_writes, filename=save_parms["DAG_name_write"])

    start = time.time()
    dask.compute(ms_writes)
    print("*** Dask compute time (main table)", time.time() - start)
    start = time.time()
    dask.compute(
        sub_ant,
        sub_ddi,
        sub_feed,
        sub_field,
        sub_his,
        sub_obs,
        sub_pol,
        sub_spw,
        sub_state,
    )
    print("*** Dask compute time (subtables)", time.time() - start)
    start = time.time()
    dask.compute(sub_point)
    print("*** Dask compute time (pointing table)", time.time() - start)

    return daskms.xds_from_ms(save_parms["ms_name"])
#################### END  write_to_ms_daskms ####################

def write_to_ms_daskms_and_sim_tool(
    vis_xds,
    time_xda,
    chan_xda,
    pol,
    tel_xds,
    phase_center_names,
    phase_center_ra_dec,
    auto_corr,
    save_parms,
):
    """
    Write out a MeasurementSet to disk using dask-ms

    This first implementation is kept only temporarily, until performance comparisons are completed.
    """

    start = time.time()
    from casatools import simulator
    from casatasks import mstransform

    n_time, n_baseline, n_chan, n_pol = vis_xds.DATA.shape
    
    print(n_time, n_baseline, n_chan, n_pol)
    print(vis_xds)

    sm = simulator()

    ant_pos = tel_xds.ANT_POS.values
    os.system("rm -rf " + save_parms["ms_name"])
    sm.open(ms=save_parms["ms_name"])

    ###########################################################################################################################
    ## Set the antenna configuration
    sm.setconfig(
        telescopename=tel_xds.telescope_name,
        x=ant_pos[:, 0],
        y=ant_pos[:, 1],
        z=ant_pos[:, 2],
        dishdiameter=tel_xds.DISH_DIAMETER.values,
        mount=["alt-az"],
        antname=list(
            tel_xds.ant_name.values
        ),  # CASA can't handle an array of antenna names.
        coordsystem="global",
        referencelocation=tel_xds.site_pos[0],
    )

    ## Set the polarization mode (this goes to the FEED subtable)
    from sirius_data._constants import pol_codes_RL, pol_codes_XY, pol_str
    from sirius._sirius_utils._array_utils import _is_subset

    if _is_subset(pol_codes_RL, pol):  # ['RR','RL','LR','LL']
        sm.setfeed(mode="perfect R L", pol=[""])
    elif _is_subset(pol_codes_XY, pol):  # ['XX','XY','YX','YY']
        sm.setfeed(mode="perfect X Y", pol=[""])
    else:
        assert False, print(
            "Pol selection invalid, must either be subset of [5,6,7,8] or [9,10,11,12] but is ",
            pol,
        )

    sm.setspwindow(
        spwname=chan_xda.spw_name,
        freq=chan_xda.data[0].compute(),
        deltafreq=chan_xda.freq_delta,
        freqresolution=chan_xda.freq_resolution,
        nchannels=len(chan_xda),
        refcode="LSRK",
        stokes=" ".join(pol_str[pol]),
    )

    if auto_corr:
        sm.setauto(autocorrwt=1.0)
    else:
        sm.setauto(autocorrwt=0.0)

    mjd = Time(time_xda.data[0:2].compute(), scale="utc")
    integration_time = (mjd[1] - mjd[0]).to("second")

    start_time = (mjd[0] - (integration_time / 2 + 37 * u.second)).mjd
    start_time_dict = {
        "m0": {"unit": "d", "value": start_time},
        "refer": "UTC",
        "type": "epoch",
    }

    sm.settimes(
        integrationtime=integration_time.value,
        usehourangle=False,
        referencetime=start_time_dict,
    )

    fields_set = []
    field_time_count = Counter(phase_center_names)

    # print(field_time_count,phase_center_names)
    if len(phase_center_names) == 1:  # Single field case
        field_time_count[list(field_time_count.keys())[0]] = n_time

    start_time = 0
    for i, ra_dec in enumerate(
        phase_center_ra_dec
    ):  # In future make phase_center_ra_dec a unique list
        if phase_center_names[i] not in fields_set:
            dir_dict = {
                "m0": {"unit": "rad", "value": ra_dec[0]},
                "m1": {"unit": "rad", "value": ra_dec[1]},
                "refer": "J2000",
                "type": "direction",
            }
            sm.setfield(sourcename=phase_center_names[i], sourcedirection=dir_dict)
            fields_set.append(phase_center_names[i])

            stop_time = (
                start_time
                + integration_time.value * field_time_count[phase_center_names[i]]
            )
            sm.observe(
                sourcename=phase_center_names[i],
                spwname=chan_xda.spw_name,
                starttime=str(start_time) + "s",
                stoptime=str(stop_time) + "s",
            )
            start_time = stop_time

    n_row = n_time * n_baseline

    print("Meta data creation ", time.time() - start)

    # print(vis_data.shape)
    # print(n_row,n_time, n_baseline, n_chan, n_pol)

    start = time.time()
    # This code will most probably be moved into simulation if we get rid of row time baseline split.
    vis_data_reshaped = vis_xds.DATA.data.reshape((n_row, n_chan, n_pol))
    uvw_reshaped = vis_xds.UVW.data.reshape((n_row, 3))
    weight_reshaped = vis_xds.WEIGHT.data.reshape((n_row, n_pol))
    sigma_reshaped = vis_xds.SIGMA.data.reshape((n_row, n_pol))

    print("reshape time ", time.time() - start)
    # weight_spectrum_reshaped = np.tile(weight_reshaped[:,None,:],(1,n_chan,1))

    #    print(weight_reshaped.compute().shape)
    #    print(sigma_reshaped.compute().shape)
    #    print(weight_reshaped)
    #    print(sigma_reshaped)

    # dask_ddid = da.full(n_row, 0, chunks=chunks['row'], dtype=np.int32)

    # print('vis_data_reshaped',vis_data_reshaped)

    start = time.time()
    from daskms import xds_to_table, xds_from_ms, Dataset

    # print('vis_data_reshaped.chunks',vis_data_reshaped.chunks)
    row_id = da.arange(n_row, chunks=vis_data_reshaped.chunks[0], dtype="int32")

    dataset = Dataset(
        {
            "DATA": (("row", "chan", "corr"), vis_data_reshaped),
            "CORRECTED_DATA": (("row", "chan", "corr"), vis_data_reshaped),
            "UVW": (("row", "uvw"), uvw_reshaped),
            "SIGMA": (("row", "pol"), sigma_reshaped),
            "WEIGHT": (("row", "pol"), weight_reshaped),
            "ROWID": (("row",), row_id),
        }
    )
    # ,'WEIGHT_SPECTRUM': (("row","chan","pol"), weight_spectrum_reshaped)
    ms_writes = xds_to_table(dataset, save_parms["ms_name"], columns="ALL")

    if save_parms["DAG_name_write"]:
        dask.visualize(ms_writes, filename=save_parms["DAG_name_write"])

    start = time.time()
    dask.compute(ms_writes)
    print("*** Dask compute time", time.time() - start)

    sm.close()

    from casatasks import flagdata

    flagdata(vis=save_parms["ms_name"], mode="unflag")

    return xds_from_ms(save_parms["ms_name"])

#################### END write_to_ms_daskms_and_sim_tool ####################

#################################################################
def write_to_ms_cngi(vis_xds,
    time_xda,
    chan_xda,
    pol,
    tel_xds,
    phase_center_names,
    phase_center_ra_dec,
    auto_corr,
    save_parms,):
    
    import numpy as np
    import xarray as xr
    n_time, n_baseline, n_chan, n_pol = vis_xds.DATA.shape
    n_row = n_time*n_baseline

    #Create mxds
    mxds = xr.Dataset()
    coords = {'polarization_ids':np.array([0]), 'spw_ids': np.array([0])}
    mxds = mxds.assign_coords(coords)

    ###############################
    #Create xds0: main table
    ###############################
    coords = {'chan':chan_xda.values, 'pol': pol, 'polarization_ids':np.array([0]), 'spw_ids': np.array([0])}

    vis_data_reshaped = vis_xds.DATA.data.reshape((n_row, n_chan, n_pol))
    uvw_reshaped = vis_xds.UVW.data.reshape((n_row, 3))
    weight_reshaped = vis_xds.WEIGHT.data.reshape((n_row, n_pol))
    sigma_reshaped = vis_xds.SIGMA.data.reshape((n_row, n_pol))
    
    chunks = {"row": (vis_data_reshaped.chunks[0][0],), "chan": (vis_data_reshaped.chunks[1][0],), "corr": (vis_data_reshaped.chunks[2][0],), "uvw": (3,)}

    # generate an antenna index for each time step
    ant1_arr = da.from_array(np.array([], dtype="int32"))
    ant2_arr = da.from_array(np.array([], dtype="int32"))
    for tt in range(0, vis_xds.sizes["time"]):
        ant1, ant2 = _calc_baseline_indx_pair(tel_xds.sizes["ant_name"], auto_corr)
        ant1_arr = da.append(ant1_arr, ant1)
        ant2_arr = da.append(ant2_arr, ant2)
        ant1s = ant1_arr.rechunk(chunks=chunks["row"])
        ant2s = ant2_arr.rechunk(chunks=chunks["row"])

    # we run this function on only a single DDI at a time
    ddid = da.zeros(n_row, chunks=chunks["row"], dtype="int32")

    # don't flag any of the data yet
    flags = da.zeros_like(vis_data_reshaped, dtype=bool)
    flag_rows = da.zeros_like(ddid, dtype=bool)
    # can we get away with not specifying flag_category ([0,0,0 Boolean])?

    # currently don't support subarrays, so only one array ID assigned
    array_ids = da.zeros_like(ddid, dtype="int32")

    # fill with input in units of the input array, which we expect to be SI (s)
    exposures = da.full_like(ddid, time_xda.time_delta, dtype="float64")
    # interval maps to exposure in perfect simulation conditions
    intervals = exposures

    # not supporting different feed types
    feeds = da.zeros_like(ddid, "int32")

    # index the strings in phase_center_names (a function of the time dimension)
    field_index = da.from_array(np.unique(phase_center_names, return_index=True)[1])
    field_ids = da.repeat(field_index, (ddid.size // field_index.size))

    # this function is also only run for a single observation at once
    observation_ids = da.zeros_like(ddid)

    # currently don't support group processing
    processor_ids = da.zeros_like(ddid)

    # WIP: since it doesn't affect data can be 0s for now, function tbc later to derive from time_xda
    scan_numbers = da.ones_like(ddid)

    # unsupported - table for semi-obscure calibration indexing (e.g., temperature loads for solar)
    state_ids = da.zeros_like(ddid)

    # fill time col input object explicitly match row chunking, expect units in SI (s)
    times = da.repeat(time_xda.data, repeats=vis_xds.sizes["baseline"]).rechunk(
        chunks=chunks["row"]
    )
    # this gave us an array of strings, but we need seconds since epoch in float to cram into the MS
    # convert to datetime64[ms], ms since epoch, seconds since epoch, then apply correction
    # NB: difference between Unix origin (1970-01-01) and what CASA expects (1858-11-17) is +/-3506716800 seconds
    #times = times.astype(np.datetime64).astype(float) / 10**3 + 3506716800.0
    times = times.astype(dtype='datetime64[ns]')
    print(times)

    # match the time column for now, ephemeris support can come later
    time_centroids = times

    # only fill the data and model columns to ensure fair comparison between write times
    empty_data_column = da.zeros_like(vis_data_reshaped)
    
    row_id = da.arange(n_row, chunks=vis_data_reshaped.chunks[0], dtype="int32")
   
    xds0 = xr.Dataset(
            {
                "DATA": (
                    ("row", "chan", "pol"),
                    vis_data_reshaped.astype("complex"),
                ),
                "MODEL_DATA": (
                    ("row", "chan", "pol"),
                    empty_data_column.astype("complex"),
                ),
                "CORRECTED_DATA": (
                    ("row", "chan", "pol"),
                    vis_data_reshaped.astype("complex"),
                ),
                "FLAG": (("row", "chan", "pol"), flags.astype("bool")),
                "UVW": (("row", "uvw_index"), uvw_reshaped.astype("float")),
                "SIGMA": (("row", "pol"), sigma_reshaped.astype("float")),
                "WEIGHT": (("row", "pol"), weight_reshaped.astype("float")),
                "FLAG_ROW": (("row"), flag_rows.astype("bool")),
                "DATA_DESC_ID": (("row"), ddid.astype("int")),
                "ANTENNA1": (("row"), ant1s.astype("int")),
                "ANTENNA2": (("row"), ant2s.astype("int")),
                "ARRAY_ID": (("row"), array_ids.astype("int")),
                "EXPOSURE": (("row"), exposures.astype("float")),
                "FEED1": (("row"), feeds.astype("int")),
                "FEED2": (("row"), feeds.astype("int")),
                "FIELD_ID": (("row"), field_ids.astype("int")),
                "INTERVAL": (("row"), intervals.astype("float")),
                "OBSERVATION_ID": (("row"), observation_ids.astype("int")),
                "PROCESSOR_ID": (("row"), processor_ids.astype("int")),
                "SCAN_NUMBER": (("row"), scan_numbers.astype("int")),
                "STATE_ID": (("row"), state_ids.astype("int")),
                "TIME": (("row"), times),
                "TIME_CENTROID": (("row"), time_centroids.astype("float")),
                "ROWID": (("row",), row_id),
                #'WEIGHT_SPECTRUM': (("row","chan","pol"), weight_spectrum_reshaped),
            }
        )
    
    
    coords = {'chan':chan_xda.values, 'pol': pol, 'pol_id':np.array([0]), 'spw_id': np.array([0])}
    xds0 = xds0.assign_coords(coords)
    
    
    xds0.attrs['MS_VERSION'] = 2.0
    from sirius_data._ms_column_descriptions_dicts import main_column_description
    xds0.attrs['column_descriptions'] = main_column_description
    xds0.attrs['info'] = {'type': 'Measurement Set', 'subType': 'simulator', 'readme': 'This is a MeasurementSet Table holding measurements from a Telescope\nThis is a MeasurementSet Table holding simulated astronomical observations\n'}
    xds0.attrs['bad_cols'] = ['FLAG_CATEGORY']
    mxds.attrs['xds0'] = xds0
    
    ###############################
    #Create SPECTRAL_WINDOW
    ###############################

    spw_xds = xr.Dataset({
            "FREQ_GROUP":(("row"), np.zeros(shape=1).astype("int")),
            "FLAG_ROW":(("row"), np.zeros(shape=1).astype("bool")),
            "NET_SIDEBAND":(("row"), np.ones(shape=1).astype("int")),
            # if only everything were consistently indexed...
            # maybe it would be better to use chan_xda.spw_name but that might break something downstream
            "FREQ_GROUP_NAME":(
                ("row"),
                np.array([chan_xda.spw_name]),
            ),
            # NB: a naive chan_xda.sum() is high by an order of magnitude!
            "TOTAL_BANDWIDTH":(
                ("row"),
                np.asarray([chan_xda.freq_delta * chan_xda.size]),
            ),
            # "frequency representative of this spw, usually the sky frequency corresponding to the DC edge of the baseband."
            # until "reference" in chan.xda.attrs use 1st channel
            "REF_FREQUENCY":(("row"), np.take(chan_xda.data, [0])),
            # obscure measures tool keyword for Doppler tracking
            "MEAS_FREQ_REF":(("row"), np.ones(shape=1).astype("int")),
            # "Identiﬁcation of the electronic signal path for the case of multiple (simultaneous) IFs.
            # (e.g. VLA: AC=0, BD=1, ATCA: Freq1=0, Freq2=1)"
            "IF_CONV_CHAIN":(("row"), np.zeros(shape=1).astype("int")),
            "NAME":(("row"), np.array([chan_xda.spw_name])),
            "NUM_CHAN":(("row"), np.array([chan_xda.size]).astype("int")),
            # the following share shape (1,chans)
            # "it is more efficient to keep a separate reference to this information"
            "CHAN_WIDTH":(
                ("row", "chan"),
                np.broadcast_to([chan_xda.freq_delta], shape=(1, chan_xda.size)).astype(
                    "float"
                ),
            ),
            # the assumption that input channel frequencies are central will hold for a while
            "CHAN_FREQ":(
                ("row", "chan"),
                np.broadcast_to(
                    np.asarray(chan_xda.data), shape=(1, chan_xda.size)
                ).astype("float"),
            ),
            "RESOLUTION":(
                ("row", "chan"),
                np.broadcast_to(
                    # note that this is not what we call chan.xda.freq_resolution
                    [chan_xda.freq_delta],
                    shape=(1, chan_xda.size),
                ).astype("float"),
            ),
            # we may eventually want to infer this by instrument, e.g., ALMA correlator binning
            # but until "effective_bw" in chan_xda.attrs,
            "EFFECTIVE_BW":(
                ("row", "chan"),
                np.broadcast_to([chan_xda.freq_delta], shape=(1, chan_xda.size)).astype(
                    "float"
                ),
            ),
            }
        )
        
    
    from sirius_data._ms_column_descriptions_dicts import spectral_window_column_description
    spw_xds.attrs['column_descriptions'] = spectral_window_column_description
    spw_xds.attrs['info'] = {'type': '', 'subType': '', 'readme': ''}
    spw_xds.attrs['bad_cols'] = []

    mxds.attrs['SPECTRAL_WINDOW'] = spw_xds
    
    ###############################
    #Create POLARIZATION
    ###############################
    # POLARIZATION
    # Surely there is a more elegant way to build this strange index
    pol_index = []
    for pp in pol:
        if pp == 5 or pp == 9:
            pol_index.append([0, 0])
        if pp == 6 or pp == 10:
            pol_index.append([0, 1])
        if pp == 7 or pp == 11:
            pol_index.append([1, 0])
        if pp == 8 or pp == 12:
            pol_index.append([1, 1])
    
    pol_xds = xr.Dataset({
            "NUM_CORR":(("row"), np.asarray([len(pol)], dtype="int")),
            "CORR_TYPE":(("row", "corr"), np.asarray([pol], dtype="int")),
            "FLAG_ROW":(("row"), np.zeros(shape=1).astype("bool")),
            # "Pair of integers for each correlation product, specifying the receptors from which the signal originated."
            "CORR_PRODUCT":(
                ("row", "corr", "corrprod_idx"),
                np.asarray([pol_index], dtype="int"),
            ),
        }
    )
    
    from sirius_data._ms_column_descriptions_dicts import polarization_column_description
    pol_xds.attrs['column_descriptions'] = polarization_column_description
    pol_xds.attrs['info'] = {'type': '', 'subType': '', 'readme': ''}
    pol_xds.attrs['bad_cols'] = []
    
    mxds.attrs['POLARIZATION'] = pol_xds
    
    ###############################
    #DATA_DESCRIPTION
    ###############################

    ddi_xds = xr.Dataset(
         {
            # this function operates on a single DDI at once, so this should reduce to length-1 arrays = 0
            # we could also enumerate the ds list if we were reading from existing MS and pass the index
            "SPECTRAL_WINDOW_ID":(("row"), np.zeros(1, dtype="int")),
            "FLAG_ROW":(("row"), np.zeros(1, dtype="bool")),
            "POLARIZATION_ID":(("row"), np.zeros(1, dtype="int")),
        },
    )
    
    from sirius_data._ms_column_descriptions_dicts import data_description_column_description
    ddi_xds.attrs['column_descriptions'] = data_description_column_description
    ddi_xds.attrs['info'] = {'type': '', 'subType': '', 'readme': ''}
    ddi_xds.attrs['bad_cols'] = []
    
    mxds.attrs['DATA_DESCRIPTION'] = ddi_xds
    
    
    ###############################
    #ANTENNA
    ###############################
    ant_xds = xr.Dataset(
            {
            "NAME":(("row"),tel_xds.ant_name.data),
            "DISH_DIAMETER":(("row"), tel_xds.DISH_DIAMETER.data),
            "POSITION":(("row", "xyz"), tel_xds.ANT_POS.data),
            # not yet supporting space-based interferometers
            "TYPE":(
                ("row"),
                np.full(tel_xds.ant_name.shape, "GROUND-BASED", dtype="<U12"),
            ),
            "FLAG_ROW":(("row"), np.zeros(tel_xds.ant_name.shape, dtype="bool")),
            # when this input is available from tel.zarr then we can infer it, til then assume alt-az
            "MOUNT":(("row"), np.full(tel_xds.ant_name.shape, "alt-az", dtype="<U6")),
            # likewise, although this seems like it should be pulled from the cfg files
            "STATION":(("row"), np.full(tel_xds.ant_name.shape, "P", dtype="<U1")),
            # until we have some input with OFFSET specified, no conditional
            "OFFSET":(
                ("row", "xyz"),
                np.zeros((tel_xds.dims["ant_name"], 3), dtype=np.float),
            ),
        }
    )
    
    from sirius_data._ms_column_descriptions_dicts import antenna_column_description
    ant_xds.attrs['column_descriptions'] = antenna_column_description
    ant_xds.attrs['info'] = {'type': '', 'subType': '', 'readme': ''}
    ant_xds.attrs['bad_cols'] = []
    
    mxds.attrs['ANTENNA'] = ant_xds
    
    ###############################
    # Feed
    ###############################
    if np.all(np.isin(pol, [5, 6, 7, 8])):
        poltype_arr = np.broadcast_to(
            np.asarray(["R", "L"]), (tel_xds.ant_name.size, 2)
            )
    elif np.all(np.isin(pol, [9, 10, 11, 12])):
        # it's clunky to assume linear feeds...
        poltype_arr = np.broadcast_to(
            np.asarray(["X", "Y"]), (tel_xds.ant_name.size, 2)
        )
        
    feed_xds = xr.Dataset(
        data_vars=dict(
            ANTENNA_ID=(("row"), np.arange(0, tel_xds.dims["ant_name"], dtype="int")),
            # -1 fill value indicates that we're not using the optional BEAM subtable
            BEAM_ID=(("row"), np.ones(tel_xds.ant_name.shape, dtype="int") * -1),
            INTERVAL=(
                ("row"),
                np.full(tel_xds.dims["ant_name"], fill_value=1e30, dtype="float"),
            ),
            # we're not supporting offset feeds yet
            POSITION=(
                ("row", "xyz"),
                np.zeros((tel_xds.dims["ant_name"], 3), dtype=np.float),
            ),
            # indexed from FEEDn in the MAIN table
            FEED_ID=(("row"), np.zeros(tel_xds.dims["ant_name"], dtype="int")),
            # "Polarization reference angle. Converts into parallactic angle in the sky domain."
            RECEPTOR_ANGLE=(
                ("row", "receptors"),
                np.zeros((tel_xds.dims["ant_name"], poltype_arr.shape[1])),
            ),
            # "Polarization response at the center of the beam for this feed expressed
            # in a linearly polarized basis (e→x,e→y) using the IEEE convention."
            # practically, broadcast a POLxPOL complex identity matrix along a new N_antenna dim
            POL_RESPONSE=(
                ("row", "receptors", "receptors-2"),
                np.broadcast_to(
                    np.eye(poltype_arr.shape[1], dtype="complex"),
                    (tel_xds.dims["ant_name"], poltype_arr.shape[1], poltype_arr.shape[1]),
                ),
            ),
            # A value of -1 indicates the row is valid for all spectral windows
            SPECTRAL_WINDOW_ID=(
                ("row"),
                np.ones(tel_xds.dims["ant_name"], dtype="int") * -1,
            ),
            NUM_RECEPTORS=(
                ("row"),
                np.full(tel_xds.dims["ant_name"], fill_value=poltype_arr.shape[1], dtype="int"),
            ),
            POLARIZATION_TYPE=(("row", "receptors"), poltype_arr),
            # "the same measure reference used for the TIME column of the MAIN table must be used"
            # in practice this appears to be 0 since the conversion to casa-expected epoch is done
            TIME=(("row"), np.zeros(tel_xds.dims["ant_name"], dtype="float")),
            # "Beam position oﬀset, as deﬁned on the sky but in the antenna reference frame."
            # the third dimension size could also be taken from phase_center_ra_dec in theory
            BEAM_OFFSET=(
                ("row", "receptors", "radec"),
                np.zeros(shape=(tel_xds.dims["ant_name"], poltype_arr.shape[1], 2), dtype="float"),
            ),
        ),
    )
    
    from sirius_data._ms_column_descriptions_dicts import feed_column_description
    feed_xds.attrs['column_descriptions'] = feed_column_description
    feed_xds.attrs['info'] = {'type': '', 'subType': '', 'readme': ''}
    feed_xds.attrs['bad_cols'] = []
    
    mxds.attrs['FEED'] = feed_xds
    
    ###############################
    # Field
    ###############################
    field_xds = xr.Dataset(
        data_vars=dict(
            NAME=(("row"), np.array(phase_center_names)),
            SOURCE_ID=(("row"), np.indices(phase_center_names.shape)[0]),
            # may need to wrap the RA at 180deg to make the MS happy
            REFERENCE_DIR=(
                ("row", "field-poly", "field-dir"),
                # expand_dims was added to dask.array in version 2022.02.0
                np.expand_dims(np.array(phase_center_ra_dec, dtype="double"), axis=0),
            ),
            PHASE_DIR=(
                ("row", "field-poly", "field-dir"),
                np.expand_dims(np.array(phase_center_ra_dec, dtype="double"), axis=0),
            ),
            DELAY_DIR=(
                ("row", "field-poly", "field-dir"),
                np.expand_dims(np.array(phase_center_ra_dec, dtype="double"), axis=0),
            ),
            CODE=(
                ("row"),
                np.full(phase_center_names.shape, fill_value="", dtype="<U1").astype(
                    "object"
                ),
            ),
            # "Required to use the same TIME Measure reference as in MAIN."
            # in practice this appears to be 0 since the conversion to casa-expected epoch is done
            TIME=(("row"), np.zeros(phase_center_names.shape, dtype="float")),
            FLAG_ROW=(("row"), np.zeros(phase_center_names.shape, dtype="bool")),
            # Series order for the *_DIR columns
            NUM_POLY=(("row"), np.zeros(phase_center_names.shape, dtype="int")),
        ),
    )
    
    
    from sirius_data._ms_column_descriptions_dicts import field_column_description
    field_xds.attrs['column_descriptions'] = field_column_description
    field_xds.attrs['info'] = {'type': '', 'subType': '', 'readme': ''}
    field_xds.attrs['bad_cols'] = []
    
    mxds.attrs['FIELD'] = field_xds
    
    ###############################
    # History
    ###############################
    his_xds = xr.Dataset(
        data_vars=dict(
            MESSAGE=(
                ("row"),
                np.array(["taskname=sirius.dio.write_to_ms"], dtype="object"),
            ),
            APPLICATION=(("row"), np.array(["ms"], dtype="object")),
            # "Required to have the same TIME Measure reference as used in MAIN."
            # but unlike some subtables with ^that^ in the spec, this is actual timestamps
            # NB: difference between Unix origin (1970-01-01) and what CASA expects (1858-11-17) is +/-3506716800 seconds
            TIME=(
                ("row"),
                (np.array([time.time()], dtype="float") / 10**3 + 3506716800.0),
            ),
            PRIORITY=(("row"), np.array(["NORMAL"], dtype="object")),
            ORIGIN=(("row"), np.array(["dask-ms"], dtype="object")),
            OBJECT_ID=(("row"), np.array([0], dtype="int")),
            OBSERVATION_ID=(("row"), np.array([-1], dtype="int")),
            # The MSv2 spec says there is "an adopted project-wide format."
            # which is big if true... appears to have shape expand_dims(MESSAGE)
            APP_PARAMS=(
                ("row", "APP_PARAMS-1"),
                np.array([[""], [""]], dtype="object").transpose(),
            ),
            CLI_COMMAND=(
                ("row", "CLI_COMMAND-1"),
                np.array([[""], [""]], dtype="object").transpose(),
            ),
        ),
    )
    
    from sirius_data._ms_column_descriptions_dicts import history_column_description
    his_xds.attrs['column_descriptions'] = history_column_description
    his_xds.attrs['info'] = {'type': '', 'subType': '', 'readme': ''}
    his_xds.attrs['bad_cols'] = []
    
    mxds.attrs['HISTORY'] = his_xds
    
    ###############################
    # Observation
    ###############################

    obs_xds = xr.Dataset(
        data_vars=dict(
            TELESCOPE_NAME=(
                ("row"),
                np.array([tel_xds.telescope_name], dtype="object"),
            ),
            RELEASE_DATE=(("row"), np.zeros(1, dtype="float")),
            SCHEDULE_TYPE=(("row"), np.array([""], dtype="object")),
            PROJECT=(("row"), np.array(["SiRIUS simulation"], dtype="object")),
            # first and last value
            TIME_RANGE=(
                ("row", "obs-exts"),
                np.array([np.take(times, [0, -1]).astype("float")]),
            ),
            # could try to be clever about this to get uname w/ os or psutil
            OBSERVER=(("row"), np.array(["SiRIUS"], dtype="object")),
            FLAG_ROW=(("row"), np.zeros(1, dtype="bool")),
        ),
    )
    
    from sirius_data._ms_column_descriptions_dicts import observation_column_description
    obs_xds.attrs['column_descriptions'] = observation_column_description
    obs_xds.attrs['info'] = {'type': '', 'subType': '', 'readme': ''}
    obs_xds.attrs['bad_cols'] = []
    
    mxds.attrs['OBSERVATION'] = obs_xds

    ###############################
    # Pointing
    ###############################
    
    pnt_xds = xr.Dataset(
        data_vars=dict(
            # is this general enough for the case where phase_center_ra_dec has size > 1 ?
            TARGET=(
                ("row", "point-poly", "radec"),
                np.broadcast_to(
                    np.array(phase_center_ra_dec),
                    shape=(tel_xds.ant_name.size * time_xda.size, 1, 2),
                ),
            ),
            # set time origin for polynomial expansions to beginning of the observation
            TIME_ORIGIN=(
                ("row"),
                np.repeat(
                    np.take(times, [0]), repeats=tel_xds.ant_name.size * time_xda.size
                ),
            ),
            INTERVAL=(
                ("row"),
                np.repeat(
                    np.asarray([time_xda.time_delta]),
                    repeats=tel_xds.ant_name.size * time_xda.size,
                ),
            ),
            # True if tracking the nominal pointing position
            TRACKING=(
                ("row"),
                np.ones(shape=tel_xds.ant_name.size * time_xda.size, dtype="bool"),
            ),
            ANTENNA_ID=(
                ("row"),
                np.tile(np.arange(0, tel_xds.ant_name.size), reps=time_xda.size),
            ),
            DIRECTION=(
                ("row", "point-poly", "radec"),
                np.broadcast_to(
                    np.array(phase_center_ra_dec),
                    shape=(tel_xds.ant_name.size * time_xda.size, 1, 2),
                ),
            ),
            # only supporting first order polynomials at present
            NUM_POLY=(
                ("row"),
                np.zeros(shape=tel_xds.ant_name.size * time_xda.size, dtype="int"),
            ),
            # could fill with phase_center_names; the reference implementation is empty
            NAME=(
                ("row"),
                np.full(
                    tel_xds.ant_name.size * time_xda.size, fill_value="", dtype="<U1"
                ).astype("object"),
            ),
            # another different use of this same column name:
            # "Mid-point of the time interval for which the information in this row is valid."
            # NB: difference between Unix origin (1970-01-01) and what CASA expects (1858-11-17) is +/-3506716800 seconds
            TIME=(
                ("row"),
                # must drop from the xr.DataArray to a raw dask.array then make expected shape
                np.repeat(
                    (
                        time_xda.astype(np.datetime64).astype(float) / 10**3
                        + 3506716800.0
                    ).data,
                    repeats=tel_xds.ant_name.size,
                ).rechunk(chunks=tel_xds.ant_name.size * time_xda.size),
            ),
        ),
    )
    
    from sirius_data._ms_column_descriptions_dicts import pointing_column_description
    pnt_xds.attrs['column_descriptions'] = pointing_column_description
    pnt_xds.attrs['info'] = {'type': '', 'subType': '', 'readme': ''}
    pnt_xds.attrs['bad_cols'] = []
    
    mxds.attrs['POINTING'] = pnt_xds
    
    ###############################
    # State
    ###############################
    
    state_xds = xr.Dataset(
        data_vars=dict(
            FLAG_ROW=(("row"), np.zeros(shape=1).astype("bool")),
            SIG=(("row"), np.ones(shape=1).astype("bool")),
            CAL=(("row"), np.zeros(shape=1).astype("float")),
            # some subset of observing modes e.g., solar will require this
            LOAD=(("row"), np.zeros(shape=1).astype("float")),
            # reference phase if available
            REF=(("row"), np.zeros(shape=1).astype("bool")),
            # relative to SCAN_NUMBER in MAIN, better support TBD
            SUB_SCAN=(("row"), np.zeros(shape=1).astype("int")),
            OBS_MODE=(
                ("row"),
                np.full(
                    shape=1, fill_value="OBSERVE_TARGET.ON_SOURCE", dtype="<U24"
                ).astype("object"),
            ),
        ),
    )

    from sirius_data._ms_column_descriptions_dicts import state_column_description
    state_xds.attrs['column_descriptions'] = state_column_description
    state_xds.attrs['info'] = {'type': '', 'subType': '', 'readme': ''}
    state_xds.attrs['bad_cols'] = []
    
    mxds.attrs['STATE'] = state_xds
    
    if save_parms['mode'] == 'lazy':
        return mxds
    else:
        write_ms(mxds, save_parms["ms_name"], subtables=True)
        return read_ms(save_parms["ms_name"], subtables=True)

#################### END write_to_ms_cngi ####################
        
        
def read_zarr(
    infile,
    sel_xds=None,
    chunks=None,
    consolidated=True,
    overwrite_encoded_chunks=True,
    **kwargs,
):
    """
    Read zarr format Visibility data from disk to xarray Dataset

    Parameters
    ----------
    infile : str
        input Visibility filename
    sel_xds : string or list
        Select the ddi to open, for example ['xds0','xds1'] will open the first two ddi. Default None returns everything
    chunks : dict
        sets specified chunk size per dimension. Dict is in the form of 'dim':chunk_size, for example {'time':100, 'baseline':400, 'chan':32, 'pol':1}.
        Default None uses the original zarr chunking.
    consolidated : bool
        use zarr consolidated metadata capability. Only works for stores that have already been consolidated. Default True works with datasets
        produced by convert_ms which automatically consolidates metadata.
    overwrite_encoded_chunks : bool
        drop the zarr chunks encoded for each variable when a dataset is loaded with specified chunk sizes.  Default True, only applies when chunks
        is not None.
        
    Returns
    -------
    xarray.core.dataset.Dataset
        New xarray Dataset of Visibility data contents
    """
    import os
    import numpy as np
    import cngi._utils._io as xdsio
    from xarray import open_zarr

    if chunks is None:
        chunks = "auto"
        #overwrite_encoded_chunks = False
    #print('overwrite_encoded_chunks',overwrite_encoded_chunks)

    infile = os.path.expanduser(infile)
    if sel_xds is None:
        sel_xds = os.listdir(infile)
    sel_xds = list(np.atleast_1d(sel_xds))
    
    
    #print(os.path.join(infile, 'DDI_INDEX'))
    mxds = open_zarr(os.path.join(infile, 'DDI_INDEX'), chunks=chunks,consolidated=consolidated,overwrite_encoded_chunks=overwrite_encoded_chunks)

    for part in os.listdir(os.path.join(infile, "global")):
        xds_temp = open_zarr(os.path.join(infile, 'global/'+part), chunks=chunks,
                                                                     consolidated=consolidated,
                                                                     overwrite_encoded_chunks=overwrite_encoded_chunks)
        xds_temp = _fix_dict_for_ms(part,xds_temp)
        mxds.attrs[part] = xds_temp

    for part in os.listdir(infile):
        if ('xds' in part) and (part in sel_xds):
            xds_temp = open_zarr(os.path.join(infile, part), chunks=chunks,
                                                                     consolidated=consolidated,
                                                                     overwrite_encoded_chunks=overwrite_encoded_chunks)
            xds_temp = _fix_dict_for_ms(part,xds_temp)
            mxds.attrs[part] = xds_temp
            
    return mxds
 
    
def write_zarr(mxds, outfile, chunks_on_disk=None, partition=None, consolidated=True, compressor=None, overwrite=True, graph_name='write_zarr'):
    """
    Write xarray dataset to zarr format on disk. When chunks_on_disk is not specified the chunking in the input dataset is used.
    When chunks_on_disk is specified that dataset is saved using that chunking.

    Parameters
    ----------
    mxds : xarray.core.dataset.Dataset
        Dataset of dataset to write to disk
    outfile : str
        outfile filename, generally ends in .zarr
    chunks_on_disk : dict of int
        A dictionary with the chunk size that will be used when writing to disk. For example {'time': 20, 'chan': 6}.
        If chunks_on_disk is not specified the chunking of dataset will be used.
    partition : str or list
        Name of partition xds to write into outfile (from the mxds attributes section). Overwrites existing partition of same name.
        Default None writes entire mxds
    compressor : numcodecs.blosc.Blosc
        The blosc compressor to use when saving the converted data to disk using zarr.
        If None the zstd compression algorithm used with compression level 2.
    graph_name : string
        The time taken to execute the graph and save the dataset is measured and saved as an attribute in the zarr file.
        The graph_name is the label for this timing information.

    Returns
    -------
    """
    import xarray as xr
    import zarr
    import time
    from numcodecs import Blosc
    from itertools import cycle
    import os
    import numpy as np

    if compressor is None:
        compressor = Blosc(cname='zstd', clevel=2, shuffle=0)

    if partition is None:
        partition = list(mxds.attrs.keys())
    partition = list(np.atleast_1d(partition))
        
    if overwrite:
        try:
            os.remove(outfile)
        except IsADirectoryError:
            shutil.rmtree(outfile)
        except FileNotFoundError:
            pass
        os.system('mkdir ' + outfile)
    else:
        assert not os.path.isfile(outfile), 'vis.zarr folder already exists. Set overwrite to True.'
        
    ddi_indx_xds = xr.Dataset()
    ddi_indx_xds['polarization_ids'] = mxds['polarization_ids']
    ddi_indx_xds['spw_ids'] = mxds['spw_ids']
    encoding = dict(zip(list(ddi_indx_xds.data_vars), cycle([{'compressor': compressor}])))
    xr.Dataset.to_zarr(ddi_indx_xds, store=outfile+'/DDI_INDEX', mode='w', encoding=encoding,consolidated=consolidated)
        
    for xds_name in partition:
        if "xds" in xds_name:
            xds_outfile = outfile + '/' + xds_name
            xds_for_disk = mxds.attrs[xds_name]
            if chunks_on_disk is not None:
                xds_for_disk = xds_for_disk.chunk(chunks=chunks_on_disk)
        else:
            xds_outfile = outfile + '/global/' + xds_name
            xds_for_disk = mxds.attrs[xds_name]
            
        xds_for_disk = _fix_dict_for_zarr(xds_name, xds_for_disk)
            
        # Create compression encoding for each datavariable
        encoding = dict(zip(list(xds_for_disk.data_vars), cycle([{'compressor': compressor}])))
        start = time.time()

        # Consolidated is set to False so that the timing information is included in the consolidate metadata.
        xr.Dataset.to_zarr(xds_for_disk, store=xds_outfile, mode='w', encoding=encoding,consolidated=False)
        time_to_calc_and_store = time.time() - start
        #print('Time to store and execute graph for ', xds_name, graph_name, time_to_calc_and_store)

        #Add timing information
        dataset_group = zarr.open_group(xds_outfile, mode='a')
        dataset_group.attrs[graph_name+'_time'] = time_to_calc_and_store
            
        if consolidated == True:
            zarr.consolidate_metadata(xds_outfile)


def _fix_dict_for_ms(name, xds):
    xds.attrs['column_descriptions'] = xds.attrs['column_descriptions'][0]
    xds.attrs['info'] = xds.attrs['info'][0]

    if "xds" in name:
        xds.column_descriptions['UVW']['shape'] = np.array(xds.column_descriptions['UVW']['shape'].split(',')).astype(int)

    if "SPECTRAL_WINDOW" == name:
        xds.column_descriptions['CHAN_FREQ']['keywords']['MEASINFO']['TabRefCodes'] = np.array(xds.column_descriptions['CHAN_FREQ']['keywords']['MEASINFO']['TabRefCodes'].split(',')).astype(int)
        xds.column_descriptions['REF_FREQUENCY']['keywords']['MEASINFO']['TabRefCodes'] =  np.array(xds.column_descriptions['REF_FREQUENCY']['keywords']['MEASINFO']['TabRefCodes'].split(',')).astype(int)
        
    if "ANTENNA" == name:
        xds.column_descriptions['OFFSET']['shape'] = np.array(xds.column_descriptions['OFFSET']['shape'].split(',')).astype(int)
        xds.column_descriptions['POSITION']['shape'] = np.array(xds.column_descriptions['POSITION']['shape'].split(',')).astype(int)
    
    if "FEED" == name:
        xds.column_descriptions['POSITION']['shape'] = np.array(xds.column_descriptions['POSITION']['shape'].split(',')).astype(int)

    if "OBSERVATION" == name:
        xds.column_descriptions['TIME_RANGE']['shape'] = np.array(xds.column_descriptions['TIME_RANGE']['shape'].split(',')).astype(int)

    return xds
    
def _fix_dict_for_zarr(name, xds):
    xds.attrs['column_descriptions'] = [xds.attrs['column_descriptions']]
    xds.attrs['info'] = [xds.attrs['info']]
    
    if "xds" in name:
        xds.column_descriptions[0]['UVW']['shape'] = ','.join(map(str, xds.column_descriptions[0]['UVW']['shape']))

    if "SPECTRAL_WINDOW" == name:
        xds.column_descriptions[0]['CHAN_FREQ']['keywords']['MEASINFO']['TabRefCodes'] = ','.join(map(str, xds.column_descriptions[0]['CHAN_FREQ']['keywords']['MEASINFO']['TabRefCodes']))
        xds.column_descriptions[0]['REF_FREQUENCY']['keywords']['MEASINFO']['TabRefCodes'] = ','.join(map(str, xds.column_descriptions[0]['REF_FREQUENCY']['keywords']['MEASINFO']['TabRefCodes']))
    
    if "ANTENNA" == name:
        xds.column_descriptions[0]['OFFSET']['shape'] = ','.join(map(str, xds.column_descriptions[0]['OFFSET']['shape']))
        xds.column_descriptions[0]['POSITION']['shape'] = ','.join(map(str, xds.column_descriptions[0]['POSITION']['shape']))
    
    if "FEED" == name:
        xds.column_descriptions[0]['POSITION']['shape'] = ','.join(map(str, xds.column_descriptions[0]['POSITION']['shape']))

    if "OBSERVATION" == name:
        xds.column_descriptions[0]['TIME_RANGE']['shape'] = ','.join(map(str, xds.column_descriptions[0]['TIME_RANGE']['shape']))
        
    return xds


