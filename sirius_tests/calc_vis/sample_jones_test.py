import numpy as np
from sirius.calc_vis import sample_J, sample_J_analytic
from sirius._sirius_utils._direction_rotate import _directional_cosine


def test_sample_J():
    x = np.arange(0, 201, 1)
    y = np.arange(0, 201, 1)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx ** 2 + (yy + 1) ** 2)
    bm_J = np.array([[[z, z], [z, z], [z, z]], [[z, z], [z, z], [z, z]]])
    bm_pa = np.array([0, 1])
    bm_chan = np.array([0, 1, 2])
    bm_pol = np.array([0, 1, 2])
    lmn = _directional_cosine((2.1, 3.2))
    freq = 1.1
    pa = 0.8
    delta_l = 4
    delta_m = 4
    assert (
        np.allclose(
            np.array([[0.52961546 + 0.0j], [0.52961546 + 0.0j]], dtype="complex128"),
            sample_J(bm_J, bm_pa, bm_chan, bm_pol, delta_l, delta_m, pa, freq, lmn),
        )
        == True
    )


def test_sample_J_analytic_airy():
    lmn = _directional_cosine((2.1, 3.2))
    assert (
        np.allclose(
            np.array([-0.00019941 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.00019941 + 0.0j]),
            sample_J_analytic("airy", 25, 2, lmn, 1.2e9, 1),
        )
        == True
    )


def test_sample_J_analytic_CASA():
    lmn = _directional_cosine((2.1, 3.2))
    assert (
        np.allclose(
            np.array([-0.00025785 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.00025785 + 0.0j]),
            sample_J_analytic("casa_airy", 25, 2, lmn, 1.2e9, 1),
        )
        == True
    )
