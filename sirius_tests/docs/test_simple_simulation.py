import os

import numpy as np
import pkg_resources
import pytest
import xarray as xr
from astropy.coordinates import SkyCoord

from sirius import simulation
from sirius.dio import make_chan_xda, make_time_xda
from sirius_data.beam_1d_func_models.airy_disk import vla


@pytest.fixture()
def source_radec_template():
    source_skycoord = SkyCoord(ra="19h59m50.51793355s", dec="+40d48m11.3694551s", frame="fk5")
    source_ra_dec = np.array([source_skycoord.ra.rad, source_skycoord.dec.rad])[None, None, :]
    return source_ra_dec


@pytest.fixture()
def phase_center_radec_template():
    phase_center = SkyCoord(ra="19h59m28.5s", dec="+40d44m01.5s", frame="fk5")
    phase_center_radec = np.array([phase_center.ra.rad, phase_center.dec.rad])[None, :]
    return phase_center_radec


@pytest.fixture()
def tel_xds_template():
    tel_dir = pkg_resources.resource_filename("sirius_data", "telescope_layout/data/vla.d.tel.zarr")
    tel_xds = xr.open_zarr(tel_dir, consolidated=False)
    return tel_xds


@pytest.fixture()
def time_xda_template():
    time_xda = make_time_xda(
        time_start="2019-10-03T19:00:00.000",
        time_delta=3600,
        n_samples=10,
        n_chunks=1,
    )
    return time_xda


@pytest.fixture()
def chan_xda_template():
    chan_xda = make_chan_xda(
        freq_start=3 * 10**9,
        freq_delta=0.4 * 10**9,
        freq_resolution=0.01 * 10**9,
        n_channels=3,
        n_chunks=1,
    )
    return chan_xda


@pytest.mark.skip(reason="xfail due to https://github.com/casangi/sirius/issues/2 but skipping is faster w/ fixtures")
@pytest.mark.parametrize("to_disk", [(False, True)])
def test_simple_simulation(
    source_radec_template,
    phase_center_radec_template,
    tel_xds_template,
    time_xda_template,
    chan_xda_template,
    to_disk,
):
    """
    Adapted from docs/simple_simulation.ipynb
    """
    xds = simulation(
        # source flux
        np.array([1.0, 0, 0, 1.0])[None, None, None, :],
        source_radec_template,
        # pointing_radec
        None,
        phase_center_radec_template,
        # phase center names
        np.array(["field1"]),
        # beam_params
        dict(),
        # beam_models
        [vla],
        # beam_model_map
        np.zeros(27, dtype=int),
        # uvw_params
        {"calc_method": "casa", "auto_corr": False},
        tel_xds_template,
        time_xda_template,
        chan_xda_template,
        # pol
        [5, 8],
        # noise_params
        None,
        # save_params
        {
            "ms_name": "simple_sim.ms",
            "write_to_ms": bool(to_disk),
        },
    )
    xds.compute()
    assert os.path.exists("simple_sim.ms")
