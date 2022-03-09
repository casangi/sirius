import pytest

import numpy as np
import dask.array as da
import xarray as xr
from sirius.dio import make_time_xda


@pytest.fixture()
def layout():
    time_xda = make_time_xda(
        time_start="2022-02-02T14:02:22.000", time_delta=3600, n_samples=2, n_chunks=2
    )
    return time_xda


@pytest.fixture()
def template():
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


def test_layout_dims(layout, template):
    assert layout.dims == template.dims


@pytest.mark.parametrize("key", ["time_delta"])
def test_attrs(layout, key):
    assert key in layout.attrs.keys()
