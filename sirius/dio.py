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
from daskms import xds_to_table, xds_from_ms, Dataset


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
        # convert to datetime64[ms]
        times = times.astype(np.datetime64)
        # then into seconds since epoch
        times = da.from_array(times.astype(float) / 10**3 + 3506716800.0)
        # NB: difference between Unix origin (1970-01-01) and what CASA expects (1858-11-17) is +/-3506716800 seconds

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

        ### perform the actual saving to the MeasurementSet using dask-ms
        ms_writes = xds_to_table(datasets, save_parms["ms_name"], columns="ALL")
        # In general we should pass specific values to columns kwargs, but since
        # we deleted any existing file to begin, should be no risk of spurious writes

        if save_parms["DAG_name_write"]:
            dask.visualize(ms_writes, filename=save_parms["DAG_name_write"])

        if save_parms["write_to_ms"]:
            start = time.time()
            dask.compute(ms_writes)
            print("*** Dask compute time", time.time() - start)

        print("compute and save time ", time.time() - start)

        ### WIP: write the subtables

        # pass or construct the arrays needed to populate each subtable

        ant_subtable = xds_from_table(save_parms["ms_name"] + "::ANTENNA")
        for i, ds in enumerate(ant_subtable):
            ant_subtable[i].NAME = tel_xds.ant_name
            ant_subtable[i].DISH_DIAMETER = tel_xds.DISH_DIAMETER
            ant_subtable[i].POSITION = tel_xds.ANT_POS
            # not yet supporting space-based interferometers
            ant_subtable[i].TYPE = da.full(
                tel_xds.ant_name.shape, "GROUND-BASED", dtype="<U12"
            )
            ant_subtable[i].FLAG_ROW = da.zeros(tel_xds.ant_name.shape, dtype="bool")
            # when this input is available from tel.zarr then we can infer it, til then assume alt-az
            ant_subtable[i].MOUNT = da.full(
                tel_xds.ant_name.shape, "alt-az", dtype="<U6"
            )
            # likewise, although this seems like it should be pulled from the cfg files
            ant_subtable[i].STATION = da.full(tel_xds.ant_name.shape, "P", dtype="<U1")
            # until we have some input with OFFSET specified, no conditional
            ant_subtable[i].OFFSET = da.zeros(
                (tel_xds.ant_name.shape, 3), dtype="f,f,f"
            )

        ddi_subtable = xds_from_table(save_parms["ms_name"] + "::DATA_DESCRIPTION")
        for i, ds in enumerate(ddi_subtable):
            # this function operates on a single DDI at once, so this should reduce to length-1 arrays = 0
            ddi_subtable[i].SPECTRAL_WINDOW_ID = da.array([i], dtype="int")
            ddi_subtable[i].FLAG_ROW = da.zeros(1, dtype="bool")
            ddi_subtable[i].POLARIZATION_ID = da.array([i], dtype="int")

        feed_subtable = xds_from_table(save_parms["ms_name"] + "::FEED")
        for i, ds in enumerate(feed_subtable):
            feed_subtable[i].ANTENNA_ID = da.arange(
                0, tel_xds.ant_name.size, dtype="int"
            )
            # -1 fill value indicates that we're not using the optional BEAM subtable
            feed_subtable[i].BEAM_ID = da.ones(tel_xds.ant_name.shape, dtype="int") * -1
            # ugh...
            feed_subtable[i].INTERVAL = da.full(
                tel_xds.ant_name.shape, fill_value=1e30, dtype="float"
            )
            # we're not supporting offset feeds yet
            feed_subtable[i].POSITION = da.zeros(
                (tel_xds.ant_name.shape, 3), dtype="f,f,f"
            )
            # indexed from FEEDn in the MAIN table
            feed_subtable[i].FEED_ID = da.zeros(tel_xds.ant_name.shape, dtype="int")
            # "Polarization reference angle. Converts into parallactic angle in the sky domain."
            feed_subtable[i].RECEPTOR_ANGLE = da.zeros(
                (tel_xds.ant_name.size, len(pol))
            )
            # "Polarization response at the center of the beam for this feed expressed
            # in a linearly polarized basis (e→x,e→y) using the IEEE convention."
            # practically, broadcast a POLxPOL complex identity matrix along a new N_antenna dim
            feed_subtable[i].POL_RESPONSE = da.broadcast_to(
                da.eye(len(pol), dtype="complex"),
                (tel_xds.ant_name.size, len(pol), len(pol)),
            )
            # A value of -1 indicates the row is valid for all spectral windows
            feed_subtable[i].SPECTRAL_WINDOW_ID = (
                da.ones(tel_xds.ant_name.shape, dtype="int") * -1
            )
            feed_subtable[i].NUM_RECEPTORS = da.full(
                tel_xds.ant_name.shape, fill_value=len(pol), dtype="int"
            )
            if np.all(np.isin(pol, [5, 6, 7, 8])):
                feed_subtable[i].POLARIZATION_TYPE = da.broadcast_to(
                    da.asarray(["R", "L"]), (tel_xds.ant_name.size, 2)
                )
            elif np.all(np.isin(pol, [9, 10, 11, 12])):
                # it's clunky to assume linear feeds
                feed_subtable[i].POLARIZATION_TYPE = da.broadcast_to(
                    da.asarray(["X", "Y"]), (tel_xds.ant_name.size, 2)
                )

            # "the same measure reference used for the TIME column of the MAIN table must be used"
            # in practice this appears to be 0 since the conversion to casa-expected epoch is done
            feed_subtable[i].TIME = da.zeros(tel_xds.ant_name.shape, dtype="float")
            # "Beam position oﬀset, as deﬁned on the sky but in the antenna reference frame."
            # the third dimension size could also be taken from phase_center_ra_dec in theory
            feed_subtable[i].BEAM_OFFSET = da.zeros(
                shape=(tel_xds.ant_name.size, len(pol), 2), dtype="float"
            )

        fcmd_subtable = xds_from_table(save_parms["ms_name"] + "::FLAG_CMD")
        for i, ds in enumerate(fcmd_subtable):
            # we're not flagging our sim so this subtable has no rows
            fcmd_subtable[i].INTERVAL = da.array([], dtype="float")
            fcmd_subtable[i].APPLIED = da.array([], dtype="bool")
            fcmd_subtable[i].COMMAND = da.array([], dtype="object")
            fcmd_subtable[i].SEVERITY = da.array([], dtype="int")
            fcmd_subtable[i].TYPE = da.array([], dtype="object")
            fcmd_subtable[i].REASON = da.array([], dtype="object")
            fcmd_subtable[i].TIME = da.array([], dtype="float")
            fcmd_subtable[i].LEVEL = da.array([], dtype="int")

        field_subtable = xds_from_table(save_parms["ms_name"] + "::FIELD")
        for i, ds in enumerate(field_subtable):
            field_subtable[i].NAME = da.array(phase_center_names)
            field_subtable[i].SOURCE_ID = da.indices(phase_center_names.shape)[0]
            # may need to wrap the RA at 180deg to make the MS happy
            field_subtable[i].REFERENCE_DIR = da.expand_dims(
                da.array(phase_center_ra_dec, dtype="double"), axis=0
            )
            field_subtable[i].PHASE_DIR = da.expand_dims(
                da.array(phase_center_ra_dec, dtype="double"), axis=0
            )
            field_subtable[i].DELAY_DIR = da.expand_dims(
                da.array(phase_center_ra_dec, dtype="double"), axis=0
            )
            field_subtable[i].CODE = da.full(
                phase_center_names.shape, fill_value="", dtype="<U1"
            ).astype("object")
            # "Required to use the same TIME Measure reference as in MAIN."
            # in practice this appears to be 0 since the conversion to casa-expected epoch is done
            field_subtable[i].TIME = da.zeros(phase_center_names.shape, dtype="float")
            field_subtable[i].FLAG_ROW = da.zeros(
                phase_center_names.shape, dtype="bool"
            )
            # Series order for the *_DIR columns
            field_subtable[i].NUM_POLY = da.zeros(phase_center_names.shape, dtype="int")

        his_subtable = xds_from_table(save_parms["ms_name"] + "::HISTORY")
        for i, ds in enumerate(his_subtable):
            # the libraries for which we care about providing history don't have __version__
            # using pkg_resources.get_distribution fails for 2/3
            # we don't want to stay pegged to 3.8 (for importlib.metadata)
            # and version numbers is the only really useful info for downstream
            # it's unclear if populating this subtable is even helpful
            his_subtable[i].MESSAGE = da.array(
                ["taskname=sirius.dio.write_to_ms"], dtype="object"
            )
            his_subtable[i].APPLICATION = da.array(["ms"], dtype="object")
            # "Required to have the same TIME Measure reference as used in MAIN."
            # but unlike some subtables with ^that^ in the spec, this is actual timestamps
            # NB: difference between Unix origin (1970-01-01) and what CASA expects (1858-11-17) is +/-3506716800 seconds
            his_subtable[i].TIME = (
                da.array([time.time()], dtype="float") / 10**3 + 3506716800.0
            )
            his_subtable[i].PRIORITY = da.array(["NORMAL"], dtype="object")
            his_subtable[i].ORIGIN = da.array(["dask-ms"], dtype="object")
            his_subtable[i].OBJECT_ID = da.array([0], dtype="int")
            his_subtable[i].OBSERVATION_ID = da.array([-1], dtype="int")
            # The MSv2 spec says there is "an adopted project-wide format."
            # which is big if true... appears to have shape expand_dims(MESSAGE)
            his_subtable[i].APP_PARAMS = da.array(["", ""], dtype="object")
            his_subtable[i].CLI_COMMAND = da.array(["", ""], dtype="object")

        obs_subtable = xds_from_table(save_parms["ms_name"] + "::OBSERVATION")
        for i, ds in enumerate(obs_subtable):
            obs_subtable.TELESCOPE_NAME = da.array(
                [tel_xds.telescope_name], dtype="object"
            )
            obs_subtable.RELEASE_DATE = da.zeros(1, dtype="float")
            obs_subtable.SCHEDULE_TYPE = da.array([""], dtype="object")
            obs_subtable.PROJECT = da.array(["SiRIUS simulation"], dtype="object")
            # first and last value
            obs_subtable.TIME_RANGE = da.take(times, [0, -1])
            obs_subtable.OBSERVER = da.array(["SiRIUS"], dtype="object")
            obs_subtable.FLAG_ROW = da.zeros(1, dtype="bool")

        pnt_subtable = xds_from_table(save_parms["ms_name"] + "::POINTING")
        for i, ds in enumerate(pnt_subtable):
            # is this general enough for the case where phase_center_ra_dec has size > 1 ?
            pnt_subtable.TARGET = da.broadcast_to(
                da.array(phase_center_ra_dec),
                shape=(tel_xds.ant_name.size * time_xda.size, 1, 2),
            )
            # set time origin for polynomial expansions to beginning of the observation
            pnt_subtable.TIME_ORIGIN = da.repeat(
                da.take(times, [0]), repeats=tel_xds.ant_name.size * time_xda.size
            )
            pnt_subtable.INTERVAL = da.repeat(
                da.asarray([time_xda.time_delta]),
                repeats=tel_xds.ant_name.size * time_xda.size,
            )
            # True if tracking the nominal pointing position
            pnt_subtable.TRACKING = da.ones(
                shape=tel_xds.ant_name.size * time_xda.size, dtype="bool"
            )
            pnt_subtable.ANTENNA_ID = da.tile(
                da.arange(0, tel_xds.ant_name.size), reps=10
            ).rechunk(chunks=tel_xds.ant_name.size * time_xda.size)
            pnt_subtable.DIRECTION = da.broadcast_to(
                da.array(phase_center_ra_dec),
                shape=(tel_xds.ant_name.size * time_xda.size, 1, 2),
            )
            # only supporting first order polynomials at present
            pnt_subtable.NUM_POLY = da.zeros(
                shape=tel_xds.ant_name.size * time_xda.size, dtype="int"
            )
            # could fill with phase_center_names; the reference implementation is empty
            pnt_subtable.NAME = da.full(
                tel_xds.ant_name.size * time_xda.size, fill_value="", dtype="<U1"
            ).astype("object")
            # another different use of this same column name:
            # "Mid-point of the time interval for which the information in this row is valid."
            # NB: difference between Unix origin (1970-01-01) and what CASA expects (1858-11-17) is +/-3506716800 seconds
            pnt_subtable.TIME = (
                time_xda.astype(np.datetime64).astype(float) / 10**3 + 3506716800.0
            )

        pol_subtable = xds_from_table(save_parms["ms_name"] + "::POLARIZATION")
        for i, ds in enumerate(pol_subtable):
            # "Pair of integers for each correlation product, specifying the receptors from which the signal originated."
            # Not sure the most elegant way to build this strange index
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
            pol_subtable[i].CORR_PRODUCT = da.asarray(pol_index, dtype="int")
            pol_subtable[i].NUM_CORR = da.asarray([len(pol)], dtype="int")
            pol_subtable[i].CORR_TYPE = da.asarray(pol, dtype="int")
            pol_subtable[i].FLAG_ROW = da.zeros(shape=1).astype("bool")

        pro_subtable = xds_from_table(save_parms["ms_name"] + "::PROCESSOR")
        for i, ds in enumerate(pro_subtable):
            # we only support a single processor, thus this subtable will remain empty
            pro_subtable.SUB_TYPE = da.array([], dtype="object")
            pro_subtable.TYPE_ID = da.array([], dtype="int")
            pro_subtable.FLAG_ROW = da.array([], dtype="bool")
            pro_subtable.TYPE = da.array([], dtype="object")
            pro_subtable.MODE_ID = da.array([], dtype="int")

        spw_subtable = xds_from_table(save_parms["ms_name"] + "::SPECTRAL_WINDOW")
        for i, ds in enumerate(spw_subtable):
            # this function will be operating on a single DDI and therefore SPW at once
            spw_subtable[i].FREQ_GROUP = da.zeros(shape=1).astype("int")
            spw_subtable[i].FLAG_ROW = da.zeros(shape=1).astype("bool")
            spw_subtable[i].NET_SIDEBAND = da.ones(shape=1).astype("int")
            # if only everything were consistently indexed...
            # maybe it would be better to use chan_xda.spw_name but that might break something downstream
            spw_subtable[i].FREQ_GROUP_NAME = da.full(
                shape=1, fill_value="Group 1", dtype="<U7"
            ).astype("object")
            # NB: a naive chan_xda.sum() is high by an order of magnitude!
            spw_subtable[i].TOTAL_BANDWIDTH = da.asarray(
                [chan_xda.freq_delta * chan_xda.size]
            )
            # "A frequency representative of this spectral window, usually the sky frequency corresponding to the DC edge of the baseband."
            if "reference" not in chan_xda.attrs:
                # until it's provided consistently, use 1st channel
                spw_subtable[i].REF_FREQUENCY = da.take(chan_xda.data, [0])
            # obscure measures tool keyword for Doppler tracking
            spw_subtable[i].MEAS_FREQ_REF = da.ones(shape=1).astype("int")
            # "Identiﬁcation of the electronic signal path for the case of multiple (simultaneous) IFs.
            # (e.g. VLA: AC=0, BD=1, ATCA: Freq1=0, Freq2=1)"
            spw_subtable[i].IF_CONV_CHAIN = da.zeros(shape=1).astype("int")
            spw_subtable[i].NAME = da.array(chan_xda.spw_name).astype("object")
            spw_subtable[i].NUM_CHAN = da.array([chan_xda.size]).astype("int")
            # shape (1,chans)
            # "it is more efficient to keep a separate reference to this information"
            spw_subtable[i].CHAN_WIDTH = da.broadcast_to(
                [chan_xda.freq_delta], shape=(chan_xda.size,)
            ).astype("float")
            # the assumption that input channel frequencies are central will hold for a while
            spw_subtable[i].CHAN_FREQ = da.asarray(chan_xda.data)
            spw_subtable[i].RESOLUTION = da.broadcast_to(
                [chan_xda.freq_resolution], shape=(chan_xda.size,)
            ).astype("float")
            # we may eventually want to infer this by instrument, e.g., ALMA correlator binning
            # but for now, just check if it's provided separately by the inputs
            if "effective_bw" not in chan_xda.attrs:
                spw_subtable[i].EFFECTIVE_BW = da.broadcast_to(
                    [chan_xda.freq_resolution], shape=(chan_xda.size,)
                ).astype("float")

        state_subtable = xds_from_table(save_parms["ms_name"] + "::STATE")
        for i, ds in enumerate(state_subtable):
            state_subtable[i].FLAG_ROW = da.zeros(shape=1).astype("bool")
            state_subtable[i].SIG = da.ones(shape=1).astype("bool")
            state_subtable[i].CAL = da.zeros(shape=1).astype("float")
            # some subset of observing modes e.g., solar will require this
            state_subtable[i].LOAD = da.zeros(shape=1).astype("float")
            # reference phase
            state_subtable[i].REF = da.zeros(shape=1).astype("bool")
            # relative to SCAN_NUMBER in MAIN, better support TBD
            state_subtable[i].SUB_SCAN = da.zeros(shape=1).astype("int")
            state_subtable[i].OBS_MODE = da.full(
                shape=1, fill_value="OBSERVE_TARGET.ON_SOURCE", dtype="<U24"
            ).astype("object")

        # next the SOURCE subtable which is clunky because it's not created by default

        # A lot of this depends on how sophisticated a point_source_skycoord is accepted
        # Until a consistent set of attributes are specified for this input,
        # try to set sensible defaults

        # Not handling source lists or source frame motion yet
        n_sources = 1
        sysvel = da.zeros(shape=(n_sources,)).astype("float")
        code = da.full(shape=(n_sources,), fill_value="", dtype="<U1").astype("object")
        calgroup = da.zeros(n_sources).astype("int")
        # seems this was filled wrong (1e+30) by the reference implementation
        interval = da.array([time_xda.time_delta]).astype("float")
        # as in FIELD subtable
        sourceid = da.indices((n_sources,))[0]
        # function acting on a single DDI (and thus SPW) at once
        spwid = da.zeros(n_sources).astype("int")
        # until variable num_lines is handled, desired transition == first channel
        restfreq = da.broadcast_to(
            da.take(chan_xda.data, [0]), shape=(n_sources, 1)
        ).astype("float")
        transition = da.broadcast_to(
            da.full(shape=(n_sources,), fill_value="X", dtype="<U1"),
            shape=(n_sources, 1),
        ).astype("object")
        # index only used by optional PULSAR subtable, which we won't support yet
        pulsarid = da.zeros(n_sources).astype("int")
        # note - this is TIME-specific and not necessarily the phase center
        direction = da.array(point_source_ra_dec).astype("float")
        # not supporting proper motion yet
        proper = da.zeros(shape=(n_sources, 2)).astype("float")
        numlines = da.ones(n_sources).astype("int")
        # since we have named fields and not sources, mosaics will need some work here
        name = da.full(
            shape=(n_sources,),
            fill_value=str(phase_center_names.squeeze()),
            dtype="U20",
        ).astype("object")
        # "Mid-point of the time interval for which the data in this row is valid."
        # Not sure how it's being calculated by the reference implementation but
        # it's 1840s (~30m) earlier than the first time value...
        # we can choose to not to, but for now it's matched exactly
        base_time = (
            time_xda.astype(np.datetime64).astype(float) / 10**3 + 3506716800.0
        )
        time = da.take(base_time, [0]) - 1840.0

        # now the extra step for this subtable of populating the dataset, just in case it's somehow exists
        try:
            source_subtable = xds_from_table(save_parms["ms_name"] + "::SOURCE")
            source_subtable[i].SYSVEL = sysvel
            source_subtable[i].CODE = code
            source_subtable[i].CALIBRATION_GROUP = calgroup
            source_subtable[i].INTERVAL = interval
            source_subtable[i].SOURCE_ID = sourceid
            source_subtable[i].SPECTRAL_WINDOW_ID = spwid
            source_subtable[i].REST_FREQUENCY = restfreq
            source_subtable[i].TRANSITION = transition
            source_subtable[i].PULSAR_ID = pulsarid
            source_subtable[i].DIRECTION = direction
            source_subtable[i].PROPER_MOTION = proper
            source_subtable[i].NUM_LINES = numlines
            source_subtable[i].NAME = name
            source_subtable[i].TIME = time
        except ValueError:
            # the table doesn't exist in the MS so we must create it
            source_subtable = xr.Dataset(
                data_vars=dict(
                    SYSVEL=(["row", "lines"], sysvel),
                    CODE=("row", code),
                    CALIBRATION_GROUP=("row", calgroup),
                    INTERVAL=("row", interval),
                    SOURCE_ID=("row", sourceid),
                    SPECTRAL_WINDOW_ID=("row", spwid),
                    REST_FREQUENCY=(["row", "lines"], restfreq),
                    TRANSITION=(["row", "lines"], transition),
                    PULSAR_ID=("row", pulsarid),
                    DIRECTION=(["row", "radec"], direction),
                    PROPER_MOTION=(["row", "radec_per_sec"], proper),
                    NUM_LINES=("row", numlines),
                    NAME=("row", name),
                    TIME=("row", time),
                ),
                coords=dict(ROWID=("row", np.zeros(1).astype("int"))),
            )

        # other subtables, e.g., SYSCAL and WEATHER are not yet supported!

        ### next, perform the actual saving to the MeasurementSet using dask-ms

        xds_to_table(
            ant_subtable,
            save_parms["ms_name"] + "::ANTENNA",
            [
                "OFFSET",
                "DISH_DIAMETER",
                "NAME",
                "MOUNT",
                "POSITION",
                "FLAG_ROW",
                "TYPE",
                "STATION",
            ],
        )
        xds_to_table(
            ddi_subtable,
            save_parms["ms_name"] + "::DATA_DESCRIPTION",
            ["SPECTRAL_WINDOW_ID", "FLAG_ROW", "POLARIZATION_ID"],
        )
        xds_to_table(
            feed_subtable,
            save_parms["ms_name"] + "::FEED",
            [
                "FEED_ID",
                "TIME",
                "NUM_RECEPTIORS",
                "POLARIZATION_TYPE",
                "POL_RESPONSE",
                "RECEPTOR_ANGLE",
                "POSITION",
                "INTERVAL",
                "ANTENNA_ID",
                "SPECTRAL_WINDOW_ID",
                "BEAM_ID",
            ],
        )
        xds_to_table(
            fcmd_subtable,
            save_parms["ms_name"] + "::FLAG_CMD",
            [
                "LEVEL",
                "APPLIED",
                "TIME",
                "REASON",
                "SEVERITY",
                "COMMAND",
                "INTERVAL",
                "TYPE",
            ],
        )
        xds_to_table(
            field_subtable,
            save_parms["ms_name"] + "::FIELD",
            [
                "CODE",
                "TIME",
                "NAME",
                "REFERENCE_DIR",
                "PHASE_DIR",
                "DELAY_DIR",
                "NUM_POLY",
                "FLAG_ROW",
                "SOURCE_ID",
            ],
        )
        xds_to_table(
            his_subtable,
            save_parms["ms_name"] + "::HISTORY",
            [
                "APP_PARAMS",
                "APPLICATION",
                "TIME",
                "MESSAGE",
                "PRIORITY",
                "ORIGIN",
                "CLI_COMMAND",
                "OBJECT_ID",
                "OBSERVATION_ID",
            ],
        )
        xds_to_table(
            obs_subtable,
            save_parms["ms_name"] + "::OBSERVATION",
            [
                "TIME_RANGE",
                "PROJECT",
                "SCHEDULE_TYPE",
                "FLAG_ROW",
                "TELESCOPE_NAME",
                "OBSERVER",
                "RELEASE_DATE",
            ],
        )
        xds_to_table(
            pnt_subtable,
            save_parms["ms_name"] + "::POINTING",
            [
                "DIRECTION",
                "TARGET",
                "TIME",
                "TIME_ORIGIN",
                "NAME",
                "TRACKING",
                "NUM_POLY",
                "INTERVAL",
                "ANTENNA_ID",
            ],
        )
        xds_to_table(
            datasets,
            "test.ms::POLARIZATION",
            ["CORR_PRODUCT", "CORR_TYPE", "FLAG_ROW", "NUM_CORR"],
        )
        xds_to_table(
            pro_subtable,
            save_parms["ms_name"] + "::PROCESSOR",
            ["SUB_TYPE", "TYPE_ID", "FLAG_ROW", "TYPE", "MODE_ID"],
        )
        xds_to_table(
            spw_subtable,
            save_parms["ms_name"] + "::SPECTRAL_WINDOW",
            [
                "FREQ_GROUP",
                "FLAG_ROW",
                "NET_SIDEBAND",
                "CHAN_WIDTH",
                "CHAN_FREQ",
                "FREQ_GROUP_NAME",
                "TOTAL_BANDWIDTH",
                "REF_FREQUENCY",
                "MEAS_FREQ_REF",
                "RESOLUTION",
                "IF_CONV_CHAIN",
                "NAME",
                "EFFECTIVE_BW",
                "NUM_CHAN",
            ],
        )
        xds_to_table(
            state_subtable,
            save_parms["ms_name"] + "::STATE",
            ["REF", "SUB_SCAN", "CAL", "SIG", "LOAD", "FLAG_ROW", "OBS_MODE"],
        )
        xds_to_table(
            source_subtable,
            save_parms["ms_name"] + "::SOURCE",
            [
                "CALIBRATION_GROUP",
                "CODE",
                "DIRECTION",
                "INTERVAL",
                "NAME",
                "NUM_LINES",
                "PROPER_MOTION",
                "SOURCE_ID",
                "SPECTRAL_WINDOW_ID",
                "TIME",
                "POSITION",
                "PULSAR_ID",
                "REST_FREQUENCY",
                "SYSVEL",
                "TRANSITION",
            ],
        )

        return xds_from_ms(save_parms["ms_name"])
