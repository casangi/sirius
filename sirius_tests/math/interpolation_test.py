import numpy as np
from scipy.interpolate import interp2d
from sirius._sirius_utils._math import bilinear_interpolate, interp_array


def test_2d_interpolation():
    x = np.arange(0, 201, 1)
    y = np.arange(0, 201, 1)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx ** 2 + (yy + 1) ** 2)
    f = interp2d(x, y, z, kind="linear")

    assert (
        np.allclose(
            f(45.5, 51.5), bilinear_interpolate(z, np.array([45.5]), np.array([51.5]))
        )
        == True
    )


def test_array_interpolation():
    x = np.arange(0, 201, 1)
    y = np.arange(0, 201, 1)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx ** 2 + (yy + 1) ** 2)
    assert (
        np.allclose(
            np.array([[0.56429664 + 0.0j], [1.12859327 + 0.0j], [1.69288991 + 0.0j]]),
            interp_array(
                np.array([z, 2 * z, 3 * z]), np.array([2]), np.array([2]), 4, 4
            ),
        )
        == True
    )


def test_array_interpolation_complex():
    x = np.arange(0, 201, 1)
    y = np.arange(0, 201, 1)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx ** 2 + (yy + 1) ** 2)
    assert (
        np.allclose(
            np.array(
                [
                    [0.56429666 + 1.1285933j],
                    [0.56429666 + 0.56429666j],
                    [0.56429666 + 0.0j],
                ],
                dtype="complex128",
            ),
            interp_array(
                np.array([z + 2j * z, z + 1j * z, z]),
                np.array([2]),
                np.array([2]),
                4,
                4,
            ),
        )
        == True
    )
