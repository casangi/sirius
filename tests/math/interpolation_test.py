import numpy as np
from scipy.interpolate import interp2d
from sirius._sirius_utils._math import bilinear_interpolate

def test_2d_interpolation():
    x = np.arange(0, 201, 1)
    y = np.arange(0, 201, 1)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx**2+(yy+1)**2)
    f = interp2d(x, y, z, kind='linear')
    
    assert np.allclose(f(45.5, 51.5), bilinear_interpolate(z, np.array([45.5]), np.array([51.5]))) == True