import pytest

import numpy as np
from astropy.coordinates import SkyCoord
from sirius._sirius_utils._coord_transforms import _sin_project


@pytest.fixture()
def point_source_template():
    """
    point_source_ra_dec.shape is expected to be (n_time[singleton], n_point_sources, 2)
    """
    point_source_skycoord = SkyCoord(
        ra="19h59m50.51793355s", dec="+40d48m11.3694551s", frame="fk5"
    )
    point_source_ra_dec = np.array(
        [point_source_skycoord.ra.rad, point_source_skycoord.dec.rad]
    )[None, None, :]
    return point_source_ra_dec


@pytest.fixture()
def phase_center_template():
    """
    phase_center_ra_dec.shape is expected to be (n_time[singleton], 2)
    """
    # NB: point source was created offset from this reference location
    phase_center = SkyCoord(ra="19h59m28.5s", dec="+40d44m01.5s", frame="fk5")
    phase_center_ra_dec = np.array([phase_center.ra.rad, phase_center.dec.rad])[None, :]
    return phase_center_ra_dec


@pytest.mark.parametrize("ra,dec", [(0.00121203, 0.00121203)])
def test_sin_project(point_source_template, phase_center_template, ra, dec):
    lm_sin = _sin_project(
        phase_center_template[0, :], 
        point_source_template[0, :, :]
    )[0, :]
    assert np.allclose(lm_sin, np.array([ra, dec]))
