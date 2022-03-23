import numpy as np
import pytest

from sirius._sirius_utils._beam_utils import _sample_J_func, _sample_J_image
from sirius._sirius_utils._coord_transforms import _directional_cosine


@pytest.mark.skip(reason="Known to be SiRIUSly broken, API refactor is WIP")
def test_sample_J():
    x = np.arange(0, 201, 1)
    y = np.arange(0, 201, 1)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx**2 + (yy + 1) ** 2)
    bm_J = np.array([[[z, z], [z, z], [z, z]], [[z, z], [z, z], [z, z]]]).astype("complex")
    bm_pa = np.array([0, 1])
    bm_chan = np.array([0, 1, 2])
    bm_pol = np.array([0, 1, 2])
    lmn = _directional_cosine(np.asarray([[2.1, 3.2]]))
    freq = 1.1
    pa = 0.8
    delta_l = 4
    delta_m = 4
    test1 = np.array(
        [[0.529161845 + 0.0j], [0.529161845 + 0.0j], [1.21749605e165 + 1.35507324e248j], [0.0 + 0.0j]],
        dtype="complex128",
    )
    test2 = _sample_J_image(bm_J, bm_pa, bm_chan, bm_pol, delta_l, delta_m, pa, freq, lmn[0, :])
    assert np.allclose(test1, test2, rtol=1e-8)


def test_sample_J_analytic_airy():
    lmn = _directional_cosine(np.asarray([[2.1, 3.2]]))
    assert np.allclose(
        np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j]),
        _sample_J_func("casa", np.array([25.0, 2.0]), 0.03113667385557884, lmn[0, :], 1.2e9, 1),
    )


def test_sample_J_analytic_CASA():
    lmn = _directional_cosine(np.asarray([[2.1, 3.2]]))
    assert np.allclose(
        np.array([-0.00026466 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.00026466 + 0.0j]),
        _sample_J_func("casa_airy", np.array([25.0, 2.0]), 0.03113667385557884, lmn[0, :], 1.2e9, 1),
        rtol=1e-5,
    )
