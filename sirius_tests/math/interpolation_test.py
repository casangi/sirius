import numpy as np
from scipy.interpolate import interp2d
from sirius._sirius_utils._math_utils import _bilinear_interpolate, _interp_array

def test_2d_interpolation():
    x = np.arange(0, 201, 1)
    y = np.arange(0, 201, 1)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx**2+(yy+1)**2)
    f = interp2d(x, y, z, kind='linear')
    
    assert np.allclose(f(45.5, 51.5), _bilinear_interpolate(z, np.array([45.5]), np.array([51.5]))) == True
    
def test_array_interpolation():
    x = np.arange(0, 201, 1)
    y = np.arange(0, 201, 1)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx**2+(yy+1)**2)
    z_c = np.sin(xx**2+(yy+1)**2)*0
    assert np.allclose(np.array([[0.56429664+0.j], [1.12859327+0.j], [1.69288991+0.j]]), _interp_array(np.array([z+1j*z_c, 2*z+1j*z_c, 3*z+1j*z_c]), np.array([2.]), np.array([2.]), 4., 4.)) == True
    
def test_array_interpolation_complex():
    x = np.arange(0, 201, 1)
    y = np.arange(0, 201, 1)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx**2+(yy+1)**2)
    assert np.allclose(np.array([[0.56429666+1.1285933j ], [0.56429666+0.56429666j], [0.56429666+0.j]], dtype="complex128"), _interp_array(np.array([z+2j*z, z+1j*z, z]), np.array([2.]), np.array([2.]), 4., 4.)) == True