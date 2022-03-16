import pytest

import pkg_resources
import numpy as np
import dask.array as da
import xarray as xr


@pytest.fixture()
def layout():
    tel_dir = pkg_resources.resource_filename(
        "sirius_data", "telescope_layout/data/vla.d.tel.zarr"
    )
    tel_xds = xr.open_zarr(tel_dir, consolidated=False)
    return tel_xds


@pytest.fixture()
def template():
    tmp_xds = xr.Dataset(
        data_vars=dict(
            ANT_POS=(
                ("ant_name", "pos_coord"),
                da.array(
                    [
                        [-1601188.989351, -5042000.518599, 3554843.38448],
                        [-1601225.230987, -5041980.39073, 3554855.657987],
                        [-1601265.110332, -5041982.563379, 3554834.816409],
                        [-1601315.874282, -5041985.324465, 3554808.263784],
                        [-1601376.950042, -5041988.68289, 3554776.344871],
                        [-1601447.176774, -5041992.529191, 3554739.647266],
                        [-1601526.335275, -5041996.876364, 3554698.284889],
                        [-1601614.061201, -5042001.676547, 3554652.455603],
                        [-1601709.987416, -5042006.942534, 3554602.306306],
                        [-1601192.424192, -5042022.883542, 3554810.383317],
                        [-1601150.02746, -5042000.630731, 3554860.703495],
                        [-1601114.318178, -5042023.187696, 3554844.922416],
                        [-1601068.771188, -5042051.92937, 3554824.767363],
                        [-1601014.405657, -5042086.261585, 3554800.76897],
                        [-1600951.545716, -5042125.92728, 3554772.987195],
                        [-1600880.545264, -5042170.376845, 3554741.425036],
                        [-1600801.880602, -5042219.386677, 3554706.382285],
                        [-1600715.918854, -5042273.14215, 3554668.128757],
                        [-1601185.55397, -5041978.191573, 3554876.382645],
                        [-1601180.820941, -5041947.459898, 3554921.573373],
                        [-1601177.368455, -5041925.069104, 3554954.532566],
                        [-1601173.903632, -5041902.679083, 3554987.485762],
                        [-1601168.735762, -5041869.062707, 3555036.885577],
                        [-1601162.553007, -5041829.021602, 3555095.854771],
                        [-1601155.593706, -5041783.860938, 3555162.327771],
                        [-1601147.885235, -5041733.855114, 3555235.914849],
                        [-1601139.483292, -5041679.021042, 3555316.478099],
                    ]
                ),
            ),
            DISH_DIAMETER=("ant_name", da.full(shape=(27,), fill_value=25.0)),
        ),
        coords=dict(
            pos_coord=da.array([0, 1, 2], dtype="int"),
            ant_name=da.array(
                [
                    "W01",
                    "W02",
                    "W03",
                    "W04",
                    "W05",
                    "W06",
                    "W07",
                    "W08",
                    "W09",
                    "E01",
                    "E02",
                    "E03",
                    "E04",
                    "E05",
                    "E06",
                    "E07",
                    "E08",
                    "E09",
                    "N01",
                    "N02",
                    "N03",
                    "N04",
                    "N05",
                    "N06",
                    "N07",
                    "N08",
                    "N09",
                ],
                dtype="<U3",
            ),
        ),
        attrs=dict(
            telescope_name="VLA",
            site_pos=[
                {
                    "m0": {"unit": "m", "value": -1601185.3650000016},
                    "m1": {"unit": "m", "value": -5041977.546999999},
                    "m2": {"unit": "m", "value": 3554875.8700000006},
                    "refer": "ITRF",
                    "type": "position",
                }
            ],
        ),
    )
    return tmp_xds


def test_layout_dims(layout, template):
    assert layout.dims == template.dims


@pytest.mark.parametrize("coord_name", ["pos_coord", "ant_name"])
def test_coords(layout, template, coord_name):
    # np.all instead of raw assert because the indices are multidimensional
    assert np.all((layout.get_index(coord_name)) == (template.get_index(coord_name)))


@pytest.mark.parametrize("key", ["telescope_name", "site_pos"])
def test_attrs(layout, key):
    assert key in layout.attrs.keys()
