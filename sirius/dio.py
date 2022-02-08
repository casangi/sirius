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
from astropy.timeseries import TimeSeries
from astropy.time import Time
from astropy import units as u
import dask.array as da
from collections import Counter
import time
import dask
import os
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
    freq_start=3 * 10 ** 9,
    freq_delta=0.4 * 10 ** 9,
    freq_resolution=0.01 * 10 ** 9,
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
        ant1_arr = da.from_array([], dtype="int32")
        ant2_arr = da.from_array([], dtype="int32")
        for tt in range(0, vis_xds.sizes["time"]):
            ant1, ant2 = _calc_baseline_indx_pair(tel_xds.sizes["ant_name"], auto_corr)
            ant1_arr.append(ant1)
            ant2_arr.append(ant2)

        # we run this function on only a single DDI at a time
        ddid = da.zeros(n_row, chunks=chunks["row"], dtype="int32")

        row_ids = da.arange(n_row, chunks=(vis_data_reshaped.chunks[0]), dtype="int32")

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
        field_index = da.unique(phase_center_names, return_index=True)[1]
        field_ids = da.repeat(
            field_index, (ddid.size // field_index.size), dtype="int32"
        )

        # this function is also only run for a single observation at once
        observation_ids = da.zeros_like(ddid)

        # currently don't support group processing
        processor_ids = da.zeros_like(ddid)

        # WIP: since it doesn't affect data can be 0s for now, function tbc later to derive from time_xda
        scan_numbers = da.ones_like(ddid)

        # unsupported - table for semi-obscure calibration indexing (e.g., temperature loads for solar)
        state_ids = da.zeros_like(ddid)

        # fill time col input object explicitly match row chunking, expect units in SI (s)
        times = da.rechunk(time_xda.data, chunks=chunks["row"])
        # match these columns for now, ephemeris support can come later
        time_centroids = times

        # only fill the data and model columns to ensure fair comparison between write times
        empty_data_column = da.zeros_like(vis_data_reshaped)

        datasets.append(
            Dataset(
                {
                    "DATA": (("row", "chan", "corr"), vis_data_reshaped),
                    "MODEL_DATA": (("row", "chan", "corr"), vis_data_reshaped),
                    "CORRECTED_DATA": (("row", "chan", "corr"), empty_data_column),
                    "FLAG": (("row", "chan", "corr"), flags),
                    "UVW": (("row", "uvw"), uvw_reshaped),
                    "SIGMA": (("row", "pol"), sigma_reshaped),
                    "WEIGHT": (("row", "pol"), weight_reshaped),
                    "FLAG_ROW": (("row"), flag_rows),
                    "DATA_DESC_ID": (("row"), ddid),
                    "ROWID": (("row"), row_ids),
                    "ANTENNA1": (("row"), da.from_array(ant1_arr)),
                    "ANTENNA2": (("row"), da.from_array(ant2_arr)),
                    "ARRAY_ID": (("row"), array_ids),
                    "EXPOSURE": (("row"), exposures),
                    "FEED1": (("row"), feeds),
                    "FEED2": (("row"), feeds),
                    "FIELD_ID": (("row"), field_ids),
                    "INTERVAL": (("row"), intervals),
                    "OBSERVATION_ID": (("row"), observation_ids),
                    "PROCESSOR_ID": (("row"), processor_ids),
                    "SCAN_NUMBER": (("row"), scan_numbers),
                    "STATE_ID": (("row"), state_ids),
                    "TIME": (("row"), times),
                    "TIME_CENTROID": (("row"), time_centroids),
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

        # still TBD: write the subtables, e.g., FEED, SOURCE

        return xds_from_ms(save_parms["ms_name"])
