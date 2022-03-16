import pytest

import numpy as np
import dask.array as da
import xarray as xr
from sirius.dio import make_time_xda, make_chan_xda


# First, the creation of time arrays


@pytest.fixture()
def time_inputs():
    time_xda = make_time_xda(
        time_start="2022-02-02T14:02:22.000", time_delta=3600, n_samples=2, n_chunks=2
    )
    return time_xda


@pytest.fixture()
def time_template():
    tmp_xda = xr.DataArray(
        data=da.from_array(
            np.array(
                ["2022-02-02T14:02:22.000", "2022-02-02T15:02:22.000"],
                dtype="<U23",
            ),
            chunks=1,
        ),
        dims=["time"],
        coords=dict(),
        attrs=dict(time_delta=3600.0),
    )
    return tmp_xda


def test_time_inputs_dims(time_inputs, time_template):
    assert time_inputs.dims == time_template.dims


@pytest.mark.parametrize("key", ["time_delta"])
def test_time_inputs_attrs(time_inputs, key):
    assert key in time_inputs.attrs.keys()


# Next, the creation of frequency arrays


@pytest.fixture()
def chan_inputs():
    chan_xda = make_chan_xda(
        freq_start=3 * 10**9,
        freq_delta=0.4 * 10**9,
        freq_resolution=0.01 * 10**9,
        n_channels=3,
        n_chunks=1,
    )
    return chan_xda


@pytest.fixture()
def chan_template():
    tmp_xda = xr.DataArray(
        data=da.from_array(np.array([3.0e09, 3.4e09, 3.8e09])),
        dims=["chan"],
        coords=dict(),
        attrs=dict(
            freq_resolution=10000000.0,
            spw_name="sband",
            freq_delta=400000000.0,
        ),
    )
    return tmp_xda


def test_chan_inputs_dims(chan_inputs, chan_template):
    assert chan_inputs.dims == chan_template.dims


@pytest.mark.parametrize("key", ["freq_resolution", "spw_name", "freq_delta"])
def test_chan_inputs_attrs(chan_inputs, key):
    assert key in chan_inputs.attrs.keys()
