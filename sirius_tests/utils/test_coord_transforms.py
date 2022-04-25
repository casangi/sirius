import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from sirius._sirius_utils._coord_transforms import _compute_rot_coords, _rot_coord, _sin_project


def test_rot_coord():
    assert np.allclose(_rot_coord(1, 2, 2), (1.402448017104221, -1.7415910999199666))


def test_compute_rot_coords():
    assert np.allclose(
        (
            np.array(
                [
                    [-1.97260236, -0.15400751, 1.66458735, 3.4831822, 5.30177705],
                    [-2.80489603, -0.98630118, 0.83229367, 2.65088853, 4.46948338],
                    [-3.63718971, -1.81859485, 0.0, 1.81859485, 3.63718971],
                    [-4.46948338, -2.65088853, -0.83229367, 0.98630118, 2.80489603],
                    [-5.30177705, -3.4831822, -1.66458735, 0.15400751, 1.97260236],
                ]
            ),
            np.array(
                [
                    [5.30177705, 4.46948338, 3.63718971, 2.80489603, 1.97260236],
                    [3.4831822, 2.65088853, 1.81859485, 0.98630118, 0.15400751],
                    [1.66458735, 0.83229367, -0.0, -0.83229367, -1.66458735],
                    [-0.15400751, -0.98630118, -1.81859485, -2.65088853, -3.4831822],
                    [-1.97260236, -2.80489603, -3.63718971, -4.46948338, -5.30177705],
                ]
            ),
        ),
        _compute_rot_coords(np.array([5, 5]), np.array([2, 2]), 2),
    )


@pytest.fixture()
def point_source_template():
    """
    point_source_ra_dec.shape is expected to be (n_time[singleton], n_point_sources, 2)
    """
    point_source_skycoord = SkyCoord(ra="19h59m50.51793355s", dec="+40d48m11.3694551s", frame="fk5")
    point_source_ra_dec = np.array([point_source_skycoord.ra.rad, point_source_skycoord.dec.rad])[None, None, :]
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
    lm_sin = _sin_project(phase_center_template[0, :], point_source_template[0, :, :])[0, :]
    assert np.allclose(lm_sin, np.array([ra, dec]))
