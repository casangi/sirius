import numpy as np
from sirius._sirius_utils._beam_utils import _sample_J_image, _sample_J_func
from sirius._sirius_utils._coord_transforms import _directional_cosine
''' SiRIUSly broken
def test_sample_J():
    x = np.arange(0, 201, 1)
    y = np.arange(0, 201, 1)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx**2+(yy+1)**2)
    bm_J = np.array([[[z, z], [z, z], [z, z]], [[z, z], [z, z], [z, z]]]).astype("complex")
    bm_pa = np.array([0, 1])
    bm_chan = np.array([0, 1, 2])
    bm_pol = np.array([0, 1, 2])
    lmn = _directional_cosine(np.asarray([[2.1, 3.2]]))
    freq = 1.1
    pa = 0.8
    delta_l = 4
    delta_m = 4
    test1 = np.array([[0.529161845+0.j],[0.529161845+0.j], [1.21749605e+165+1.35507324e+248j], [0.+0.j]], dtype="complex128")
    test2 = _sample_J_image(bm_J, bm_pa, bm_chan, bm_pol, delta_l, delta_m, pa, freq, lmn[0,:])
    print('******',test2)
    print(test2.shape)
    assert np.allclose(test1, test2, rtol = 1e-8) == True
'''
    
def test_sample_J_analytic_airy():
    lmn = _directional_cosine(np.asarray([[2.1, 3.2]]))
    assert np.allclose(np.array([1.0+0.j, 0.+0.j, 0.+0.j, 1.0+0.j]), _sample_J_func("casa", np.array([25., 2.]), 0.03113667385557884, lmn[0,:], 1.2e9, 1)) == True
    
def test_sample_J_analytic_CASA():
    lmn = _directional_cosine(np.asarray([[2.1, 3.2]]))
    #assert np.allclose(np.array([-0.00025785+0.j, 0.+0.j, 0.+0.j, -0.00025785+0.j]), sample_J_analytic("casa_airy", 25, 2, lmn, 1.2e9, 1)) == True
    assert np.allclose(np.array([-0.00026466+0.j,  0.        +0.j,  0.        +0.j, -0.00026466+0.j]), _sample_J_func("casa_airy", np.array([25., 2.]), 0.03113667385557884, lmn[0,:], 1.2e9, 1), rtol = 1e-8) == True
