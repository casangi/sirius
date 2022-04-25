import numpy as np
from scipy.interpolate import interp2d

from sirius._sirius_utils._math_utils import _bilinear_interpolate, _interp_array, _powl, _powl2, mat_dis


def test_2d_interpolation():
    x = np.arange(0, 201, 1)
    y = np.arange(0, 201, 1)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx**2 + (yy + 1) ** 2)
    f = interp2d(x, y, z, kind="linear")

    assert np.allclose(f(45.5, 51.5), _bilinear_interpolate(z, np.array([45.5]), np.array([51.5])))


def test_array_interpolation():
    x = np.arange(0, 201, 1)
    y = np.arange(0, 201, 1)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx**2 + (yy + 1) ** 2)
    z_c = np.sin(xx**2 + (yy + 1) ** 2) * 0
    assert np.allclose(
        np.array([[0.56429664 + 0.0j], [1.12859327 + 0.0j], [1.69288991 + 0.0j]]),
        _interp_array(
            np.array([z + 1j * z_c, 2 * z + 1j * z_c, 3 * z + 1j * z_c]), np.array([2.0]), np.array([2.0]), 4.0, 4.0
        ),
    )


def test_array_interpolation_complex():
    x = np.arange(0, 201, 1)
    y = np.arange(0, 201, 1)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx**2 + (yy + 1) ** 2)
    assert np.allclose(
        np.array([[0.56429666 + 1.1285933j], [0.56429666 + 0.56429666j], [0.56429666 + 0.0j]], dtype="complex128"),
        _interp_array(np.array([z + 2j * z, z + 1j * z, z]), np.array([2.0]), np.array([2.0]), 4.0, 4.0),
    )


def test_powl2():
    assert np.allclose(_powl2(np.array([[5.0]]), 5), np.array([[5.0**5]]))


def test_powl2_array():
    assert np.allclose(_powl2(np.array([[5.0, 6.0, 7.0]]), 5), np.array([[5.0, 6.0, 7.0]]) ** 5)


def test_powl2_arrays():
    assert np.allclose(
        _powl2(np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0]]), 5), np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0]]) ** 5
    )


def test_powl():
    assert np.allclose(_powl(5.0, 5), 5**5)


def test_mat_dis():
    A = np.array([np.linspace(0, 4, 4), np.linspace(0, 4, 4)])
    B = np.array([np.linspace(0, 4, 4) + 1, np.linspace(0, 4, 4) + 1])
    assert np.allclose(np.sum(np.abs(A - B)), mat_dis(A, B))
