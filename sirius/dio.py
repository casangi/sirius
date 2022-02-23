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
from daskms import xds_to_table, xds_from_table, xds_from_ms, Dataset


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


def write_to_ms(
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

    if save_parms["write_to_ms"]:
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
    chunks = {"row": (n_row,), "chan": (n_chan,), "corr": (n_pol,), "uvw": (3,)}

    # This code will most probably be moved into simulation if we get rid of row time baseline split.
    vis_data_reshaped = vis_xds.DATA.data.reshape((n_row, n_chan, n_pol))
    uvw_reshaped = vis_xds.UVW.data.reshape((n_row, 3))
    weight_reshaped = vis_xds.WEIGHT.data.reshape((n_row, n_pol))
    sigma_reshaped = vis_xds.SIGMA.data.reshape((n_row, n_pol))

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
        Dataset(
            {
                "DATA": (
                    ("row", "chan", "corr"),
                    vis_data_reshaped.astype("complex"),
                ),
                "MODEL_DATA": (
                    ("row", "chan", "corr"),
                    vis_data_reshaped.astype("complex"),
                ),
                "CORRECTED_DATA": (
                    ("row", "chan", "corr"),
                    empty_data_column.astype("complex"),
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

    ### then, pass or construct the arrays needed to populate each subtable

    # ANTENNA
    ant_subtable = []
    ds = Dataset(data_vars=dict(
        NAME=(("row"), da.from_array(tel_xds.ant_name.data, chunks=tel_xds.dims['ant_name'])),
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
            da.zeros((tel_xds.dims['ant_name'], 3), dtype=np.float)),
        )
    )
    ant_subtable.append(ds)
    
    # DATA_DESCRIPTION
    ddi_subtable = xr.Dataset(
        data_vars=dict(
            # this function operates on a single DDI at once, so this should reduce to length-1 arrays = 0
            # we could also enumerate the ds list if we were reading from existing MS and pass the index
            SPECTRAL_WINDOW_ID=(("row"), da.zeros(1, dtype="int")),
            FLAG_ROW=(("row"), da.zeros(1, dtype="bool")),
            POLARIZATION_ID=(("row"), da.zeros(1, dtype="int")),
        ),
    )

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

    feed_subtable = xr.Dataset(
        data_vars=dict(
            ANTENNA_ID=(("row"), da.arange(0, tel_xds.dims['ant_name'], dtype="int")),
            # -1 fill value indicates that we're not using the optional BEAM subtable
            BEAM_ID=(("row"), da.ones(tel_xds.ant_name.shape, dtype="int") * -1),
            INTERVAL=(
                ("row"),
                da.full(tel_xds.dims['ant_name'], fill_value=1e30, dtype="float"),
            ),
            # we're not supporting offset feeds yet
            POSITION=(
                ("row", "xyz"),
                da.zeros((tel_xds.dims['ant_name'], 3), dtype=np.float),
            ),
            # indexed from FEEDn in the MAIN table
            FEED_ID=(("row"), da.zeros(tel_xds.dims['ant_name'], dtype="int")),
            # "Polarization reference angle. Converts into parallactic angle in the sky domain."
            RECEPTOR_ANGLE=(
                ("row", "receptors"),
                da.zeros((tel_xds.dims['ant_name'], len(pol))),
            ),
            # "Polarization response at the center of the beam for this feed expressed
            # in a linearly polarized basis (e→x,e→y) using the IEEE convention."
            # practically, broadcast a POLxPOL complex identity matrix along a new N_antenna dim
            POL_RESPONSE=(
                ("row", "receptors", "receptors-2"),
                da.broadcast_to(
                    da.eye(len(pol), dtype="complex"),
                    (tel_xds.dims['ant_name'], len(pol), len(pol)),
                ),
            ),
            # A value of -1 indicates the row is valid for all spectral windows
            SPECTRAL_WINDOW_ID=(
                ("row"),
                da.ones(tel_xds.dims['ant_name'], dtype="int") * -1,
            ),
            NUM_RECEPTORS=(
                ("row"),
                da.full(tel_xds.dims['ant_name'], fill_value=len(pol), dtype="int"),
            ),
            POLARIZATION_TYPE=(("row", "receptors"), poltype_arr),
            # "the same measure reference used for the TIME column of the MAIN table must be used"
            # in practice this appears to be 0 since the conversion to casa-expected epoch is done
            TIME=(("row"), da.zeros(tel_xds.dims['ant_name'], dtype="float")),
            # "Beam position oﬀset, as deﬁned on the sky but in the antenna reference frame."
            # the third dimension size could also be taken from phase_center_ra_dec in theory
            BEAM_OFFSET=(
                ("row", "receptors", "radec"),
                da.zeros(shape=(tel_xds.dims['ant_name'], len(pol), 2), dtype="float"),
            ),
        ),
    )

    # FLAG_CMD
    # we're not flagging our sim so this subtable has no rows

    # FIELD
    field_subtable = xr.Dataset(
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

    # HISTORY
    # the libraries for which we care about providing history don't have __version__
    # using pkg_resources.get_distribution fails for 2/3
    # we don't want to stay pegged to 3.8 (for importlib.metadata)
    # and version numbers seems like the only really useful info downstream
    # it's unclear if populating this subtable is even helpful
    his_subtable = xr.Dataset(
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
            APP_PARAMS=(("row", "APP_PARAMS-1"), da.array([[""], [""]], dtype="object").transpose()),
            CLI_COMMAND=(("row", "CLI_COMMAND-1"), da.array([[""], [""]], dtype="object").transpose()),
        ),
    )

    # OBSERVATION
    obs_subtable = xr.Dataset(
        data_vars=dict(
            TELESCOPE_NAME=(
                ("row"),
                da.array([tel_xds.telescope_name], dtype="object"),
            ),
            RELEASE_DATE=(("row"), da.zeros(1, dtype="float")),
            SCHEDULE_TYPE=(("row"), da.array([""], dtype="object")),
            PROJECT=(("row"), da.array(["SiRIUS simulation"], dtype="object")),
            # first and last value
            TIME_RANGE=(("row", "obs-exts"), da.array([da.take(times, [0, -1]).astype("float")])),
            # could try to be clever about this to get uname w/ os or psutil
            OBSERVER=(("row"), da.array(["SiRIUS"], dtype="object")),
            FLAG_ROW=(("row"), da.zeros(1, dtype="bool")),
        ),
    )

    # POINTING
    pnt_subtable = xr.Dataset(
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
                da.tile(da.arange(0, tel_xds.ant_name.size), reps=10).rechunk(
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
                da.repeat((time_xda.astype(np.datetime64).astype(float) / 10**3 + 3506716800.0).data, repeats=tel_xds.ant_name.size)
            ),
        ),
    )

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
    pol_subtable = xr.Dataset(
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

    # PROCESSOR
    # we only support a single processor, thus this subtable will remain empty

    # SPECTRAL_WINDOW
    # this function will be operating on a single DDI and therefore SPW at once
    spw_subtable = xr.Dataset(
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
            CHAN_FREQ=(("row", "chan"), 
                       da.broadcast_to(da.asarray(chan_xda.data), shape=(1, chan_xda.size)).astype(
                           "float"),
                   ),
            RESOLUTION=(
                ("row", "chan"),
                da.broadcast_to(
                    # note that this is not what we call chan.xda.freq_resolution
                    [chan_xda.freq_delta], shape=(1, chan_xda.size)
                ).astype("float"),
            ),
            # we may eventually want to infer this by instrument, e.g., ALMA correlator binning
            # but until "effective_bw" in chan_xda.attrs,
            EFFECTIVE_BW=(
                ("row", "chan"),
                da.broadcast_to(
                    [chan_xda.freq_delta], shape=(1, chan_xda.size)
                ).astype("float"),
            ),
        ),
    )

    # STATE
    state_subtable = xr.Dataset(
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

    # SOURCE

    # A lot of this depends on how sophisticated a point_source_skycoord is accepted
    # Until a consistent set of attributes are specified for this input,
    # try to set sensible defaults

    # Not handling source lists or source frame motion yet
    n_sources = 1

    # Unclear how it's being calculated by the reference implementation but
    # it's 1840s (~30m) earlier than the first time value...
    # we can choose not to, but for now it's matched exactly
    base_time = time_xda.astype(np.datetime64).astype(float) / 10**3 + 3506716800.0
    adj_time = da.take(base_time, [0]) - 1840.0

    source_subtable = xr.Dataset(
        data_vars=dict(
            SYSVEL=(["row", "lines"], da.zeros(shape=(1, n_sources)).astype("float")),
            CODE=(
                "row",
                da.full(shape=(n_sources,), fill_value="", dtype="<U1").astype(
                    "object"
                ),
            ),
            CALIBRATION_GROUP=("row", da.zeros(n_sources).astype("int")),
            # seems this was filled wrong (1e+30) by the reference implementation
            INTERVAL=("row", da.array([time_xda.time_delta]).astype("float")),
            # as in FIELD subtable
            SOURCE_ID=("row", da.indices((n_sources,))[0]),
            # function acting on a single DDI (and thus SPW) at once
            SPECTRAL_WINDOW_ID=("row", da.zeros(n_sources).astype("int")),
            # until variable num_lines is handled, desired transition == first channel
            REST_FREQUENCY=(
                ["row", "lines"],
                da.broadcast_to(
                    da.take(chan_xda.data, [0]), shape=(n_sources, 1)
                ).astype("float"),
            ),
            TRANSITION=(
                ["row", "lines"],
                da.broadcast_to(
                    da.full(shape=(n_sources,), fill_value="X", dtype="<U1"),
                    shape=(n_sources, 1),
                ).astype("object"),
            ),
            # index only used by optional PULSAR subtable, which we won't support yet
            PULSAR_ID=("row", da.zeros(n_sources).astype("int32")),
            # note - this is specific to the provided TIME
            DIRECTION=(["row", "radec"], da.array(phase_center_ra_dec).astype("float")),
            # not supporting proper motion yet
            PROPER_MOTION=(
                ["row", "radec_per_sec"],
                da.zeros(shape=(n_sources, 2)).astype("float"),
            ),
            NUM_LINES=("row", da.ones(n_sources).astype("int")),
            # since we have named fields and not sources, mosaics will need some work here
            NAME=(
                "row",
                da.full(
                    shape=(n_sources,),
                    fill_value=str(phase_center_names.squeeze()),
                    dtype="U20",
                ).astype("object"),
            ),
            # "Mid-point of the time interval for which the data in this row is valid."
            TIME=("row", adj_time.data),
        ),
    )

    # other subtables, e.g., SYSCAL and WEATHER are not yet supported!
    
    # In general we should pass specific values to columns kwargs, but since
    # we deleted any existing file to begin, should be no risk of spurious writes

    # the main table object should be added to the graph first to avoid RuntimeErrors
    ms_writes = xds_to_table(datasets, save_parms["ms_name"], columns="ALL")

    sub_ant = xds_to_table(
        ant_subtable,
        "::".join((save_parms["ms_name"], "ANTENNA")),
        columns="ALL",
    )

    sub_ddi = xds_to_table(
        ddi_subtable,
        "::".join((save_parms["ms_name"], "DATA_DESCRIPTION")),
        columns="ALL",
    )

    sub_feed = xds_to_table(
        feed_subtable,
        "::".join((save_parms["ms_name"], "FEED")),
        columns="ALL",
    )
    
    sub_field = xds_to_table(
        field_subtable,
        "::".join((save_parms["ms_name"], "FIELD")),
        columns="ALL",
    )
    
    sub_his = xds_to_table(
        his_subtable,
        "::".join((save_parms["ms_name"], "HISTORY")),
        columns="ALL",
    )
    
    sub_obs = xds_to_table(
        obs_subtable,
        "::".join((save_parms["ms_name"], "OBSERVATION")),
        columns="ALL",
    )
    """
    sub_point = xds_to_table(
        pnt_subtable,
        "::".join((save_parms["ms_name"], "POINTING")),
        columns="ALL",
    )
    """
    sub_pol = xds_to_table(
        pol_subtable,
        "::".join((save_parms["ms_name"], "POLARIZATION")),
        columns="ALL",
    )

    sub_spw = xds_to_table(
        spw_subtable,
        "::".join((save_parms["ms_name"], "SPECTRAL_WINDOW")),
        columns="ALL",
    )

    sub_state = xds_to_table(
        state_subtable,
        "::".join((save_parms["ms_name"], "STATE")),
        columns="ALL",
    )
    
    sub_source = xds_to_table(
        source_subtable,
        "::".join((save_parms["ms_name"], "SOURCE")),
        columns="ALL",
    )

    ### execute the graphs

    if save_parms["DAG_name_write"]:
        dask.visualize(ms_writes, filename=save_parms["DAG_name_write"])

    if save_parms["write_to_ms"]:
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
            sub_source,
        )
        print("*** Dask compute time (subtables)", time.time() - start)
        #start = time.time()
        #dask.compute(sub_point)
        #print("*** Dask compute time (pointing table)", time.time() - start)

    return xds_from_ms(save_parms["ms_name"])
