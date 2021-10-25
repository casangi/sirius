import numpy as np
from sirius._sirius_utils._math_utils import _powl2, _powl

def test_powl2():
    assert np.allclose(_powl2(np.array([[5.]]), 5), np.array([[5.**5]])) == True
    
def test_powl2_array():
    assert np.allclose(_powl2(np.array([[5., 6., 7.]]), 5), np.array([[5., 6., 7.]])**5) == True
    
def test_powl2_arrays():
    assert np.allclose(_powl2(np.array([[5., 6., 7.], [5., 6., 7.]]), 5), np.array([[5., 6., 7.], [5., 6., 7.]])**5) == True
    
def test_powl():
    assert np.allclose(_powl(5., 5), 5**5) == True