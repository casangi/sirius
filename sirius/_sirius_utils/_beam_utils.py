 #   Copyright 2019 AUI, Inc. Washington DC, USA
 #
 #   Licensed under the Apache License, Version 2.0 (the "License");
 #   you may not use this file except in compliance with the License.
 #   You may obtain a copy of the License at
 #
 #       http://www.apache.org/licenses/LICENSE-2.0
 #
 #   Unless required by applicable law or agreed to in writing, software
 #   distributed under the License is distributed on an "AS IS" BASIS,
 #   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 #   See the License for the specific language governing permissions and
 #   limitations under the License.
 
import numpy as np
import itertools
import time
import xarray as xr
import dask.array as da
import numba
from numba import jit
from scipy import signal
from sirius._sirius_utils._beam_funcs import _casa_airy_beam, _airy_beam, _casa_airy_beam_njit, _airy_beam_njit, _poly_beam
from sirius._sirius_utils._math_utils import _interp_array
from sirius._sirius_utils._coord_transforms import  _rot_coord, _compute_rot_coords
from sirius._sirius_utils._zernike_polynomials import _generate_zernike_surface
from sirius._sirius_utils._array_utils import _find_angle_indx, _find_val_indx
from sirius_data._constants import map_mueler_to_pol, c


def _calc_resolution(beam_freq,dish_diameter,beam_parms):
    # Ensures that the beam lies within the image.
    fov = beam_parms['fov_scaling']*(1.22 * c / (dish_diameter*beam_freq))
    
    print(fov)
    max_delta = max(max(fov/beam_parms['image_size'][0]),max(fov/beam_parms['image_size'][1]))
    return max_delta

def _beam_models_to_tuple(beam_models):
    '''
    Convert different beam models into a single format that consists of tupples and arrays.
    This is required by numba that does not support dictionaries.
    
    The function changes beam_models to a tuple where each element is tuple containing a beam model (tuple of tuples).
    The elements in the beam model tuple are strings, numbers and arrays describings the beam model.
    
    The beam model format:
    ( type: int
        0-image based beam model
        1-function based beam model
      func: str
        'airy'
        'casa_airy'
        'none'-image based
      J/par: complex np.array, [n_pa, n_chan, n_pol, n_l, n_m]/[1,1,1,1,n_par]
        The image or parameters for the beam function.
      pa: float np.array, [n_pa]
        Parallactic angles.
      chan: float np.array [n_chan]
        Channel frequencies.
      pol: int np.array [n_pol]
        Pol index (not pol code).
      delta_l: float
      delta_m: float
      max_rad_1GHz: float )
    '''
    

    beam_models_list = []
    
    for i,bm in enumerate(beam_models):
        if "J" in bm: #Image based beam
            bm_tuple = _exstract_arrays_from_img_bm_xds(bm)
            beam_models_list.append(bm_tuple)
        elif "func" in bm:
            bm_tuple = _exstract_vals_from_analytic_dict(bm)
            beam_models_list.append(bm_tuple)
        elif "BPC" in bm:
            bm_tuple = _exstract_arrays_from_poli_bm_xds(bm)
            beam_models_list.append(bm_tuple)
        else:
            assert False, print('Unsupported beam model:',bm)
        
    return tuple(beam_models_list)

    #(type: int, func: str,J/par: complex np.array, [n_pa, n_chan, n_pol, n_l, n_m]/[1,1,1,1,n_par], pa: float np.array, [n_pa], chan: float np.array [n_chan], pol: int np.array [n_pol], delta_l: float, delta_m: float ,max_rad_1GHz: float)
    
    

    
def check_bm(bm):
    '''
    Ensure bm is in correct format otherwise numba code will malfunction.
    '''
    assert isinstance(bm[0],np.int)
    assert isinstance(bm[1],str)
    assert isinstance(bm[2],np.ndarray)
    assert isinstance(bm[2][0,0,0,0,0],complex)
    assert len(bm[2].shape)==5
    assert isinstance(bm[3],np.ndarray)
    assert isinstance(bm[3][0],float)
    assert len(bm[3].shape)==1
    assert isinstance(bm[4],np.ndarray)
    assert isinstance(bm[4][0],float)
    assert len(bm[4].shape)==1
    assert isinstance(bm[5],np.ndarray)
    assert isinstance(bm[5][0],np.int64)
    assert len(bm[5].shape)==1
    assert isinstance(bm[6],float)
    assert isinstance(bm[7],float)
    assert isinstance(bm[8],float)
    
def _exstract_arrays_from_poli_bm_xds(bm):
    #(type: int, func: str,J: complex np.array, [n_pa, n_chan, n_pol, n_l, n_m], pa: float np.array, [n_pa], chan: float np.array [n_chan], pol: int np.array [n_pol], delta_l: float, delta_m: float ,max_rad_1GHz: float)
    bm_BPC = bm.BPC.values.astype(np.complex128)[None,:,:,None,:]
    #print("bm_J", bm_J.shape)
    chan = bm.chan.values
    pol = _pol_code_to_index(bm.pol.values)
    max_rad_1GHz = bm.attrs['max_rad_1GHz']
    func = 'poli'
    f_arr = np.array([0.0])
    i_arr = np.array([0])
    return (2,func,bm_BPC,f_arr,chan,pol,0.0,0.0,max_rad_1GHz)

def _exstract_arrays_from_img_bm_xds(bm):
    #(type: int, func: str,J: complex np.array, [n_pa, n_chan, n_pol, n_l, n_m], pa: float np.array, [n_pa], chan: float np.array [n_chan], pol: int np.array [n_pol], delta_l: float, delta_m: float ,max_rad_1GHz: float)
    bm_J = bm.J.values 
    #print("bm_J", bm_J.shape)
    pa = bm.pa.values
    chan = bm.chan.values
    pol = _pol_code_to_index(bm.pol.values)
    delta_l = (bm.l[1].values - bm.l[0].values).astype(float)
    delta_m = (bm.m[1].values - bm.m[0].values).astype(float)
    max_rad_1GHz = bm.attrs['max_rad_1GHz']
    func = 'image'
    return (0,func,bm_J,pa,chan,pol,delta_l,delta_m,max_rad_1GHz)
    
def _exstract_vals_from_analytic_dict(bm):
    #(type: int, func: str,J/par: complex np.array, [1,1,1,1,n_par], -: float np.array, [1], -: float np.array [1], -: int np.array [1], -: float, -: float,max_rad_1GHz: float)
    func = bm['func']
    #dish_diameter = bm['dish_diameter']
    #blockage_diameter = bm['blockage_diameter']
    dish_diameter = bm['dish_diam']
    blockage_diameter = bm['blockage_diam']
    max_rad_1GHz = bm['max_rad_1GHz']
    
    par_arr = np.zeros((1,1,1,1,2),np.complex)
    par_arr[0,0,0,0,0] = dish_diameter
    par_arr[0,0,0,0,1] = blockage_diameter
    f_arr = np.array([0.0])
    i_arr = np.array([0])
    return (1,func,par_arr,f_arr,f_arr,i_arr,0.0,0.0,max_rad_1GHz)

#@jit(nopython=True,cache=True,nogil=True)
@jit(nopython=True,nogil=True)
def _calc_pb_scale(flux, sep1, sep2,lmn1,lmn2,bm1_indx,bm2_indx,bm1,bm2,pa,freq,mueller_selection,do_pointing):
    # bm tuple: (type: int, func: str,J/par: complex np.array, [n_pa, n_chan, n_pol, n_l, n_m]/[1,1,1,1,n_par], pa: float np.array, [n_pa], chan: float np.array [n_chan], pol: int np.array [n_pol], delta_l: float, delta_m: float ,max_rad_1GHz: float)
    # bm[0]: 0 (zernike), 1 (func), 2 (polinomial)
    outside_beam = False
    max_rad_1 = bm1[8]/(freq/10**9) # scale max_rad_1GHz (bm[8]) to freq
    max_rad_2 = bm2[8]/(freq/10**9) # scale max_rad_1GHz (bm[8]) to freq
    
    #Create Jones matrix J for first antenna
    if (sep1 < max_rad_1) and (sep2 < max_rad_2):
        if bm1[0] == 0: #Image beam model
            J_sampled_1 = _sample_J_image(bm1[2],bm1[3],bm1[4],bm1[5],bm1[6],bm1[7],pa,freq,lmn1)
        elif bm1[0] == 1: #Function based beam model
            J_sampled_1 = _sample_J_func(bm1[1],bm1[2][0,0,0,0,:],bm1[8],lmn1,freq,1)
        elif bm1[0] == 2: #Polynomial based model
            J_sampled_1 = _sample_J_poli(bm_BPC=bm1[2][0,:,:,0,:],bm_chan=bm1[4],bm_pol=bm1[5],pa=pa,freq=freq,lmn=lmn1,ipower=1,max_rad_1GHz=bm1[8])
    else:
        outside_beam = True
        flux_scaled = np.zeros(4, dtype = numba.complex128)
        return flux_scaled, outside_beam
        
    #Create Jones matrix J for second antenna if different from first.
    if (bm1_indx == bm2_indx) and ~do_pointing:
        J_sampled_2 = J_sampled_1
    else:
        if bm2[0] == 0: #Image beam model
            J_sampled_2 = _sample_J_image(bm2[2],bm2[3],bm2[4],bm2[5],bm2[6],bm2[7],pa,freq,lmn2)
        elif bm2[0] == 1: #Function based beam model
            J_sampled_2 = _sample_J_func(bm2[1],bm2[2][0,0,0,0,:],bm2[8],lmn2,freq,1)
        elif bm2[0] == 2: #Polynomial based model
            J_sampled_2 = _sample_J_poli(bm_BPC=bm2[2][0,:,:,0,:],bm_chan=bm2[4],bm_pol=bm2[5],pa=pa,freq=freq,lmn=lmn2,ipower=1,max_rad_1GHz=bm2[8])
    
    if (bm1[0] in [1,2]) and (bm2[0] in [1,2]): #The function and polynomial coefficient based beam models have no leakage terms.
        flux_scaled = flux*(J_sampled_1*J_sampled_2)
    else:
        M = _make_mueler_mat(J_sampled_1, J_sampled_2, mueller_selection)
        flux_scaled = np.dot(M,flux)
        
    return flux_scaled, outside_beam

@jit(nopython=True,cache=True,nogil=True)
def _pol_code_to_index(pol):
    if pol[0] in [5,6,7,8]:
        return pol-5
    if pol[0] in [9,10,11,12]:
        return pol-9
    if pol[0] in [0,1,2,3]:
        return pol
    assert False # "Unsupported pol " + str(pol) #Numba does not support an explicit error message.
    
@jit(nopython=True,cache=True,nogil=True)
def _index_to_pol_code(index,pol):
    if pol[0] in [5,6,7,8]:
        return index+5
    if pol[0] in [9,10,11,12]:
        return index+9
    assert False # "Unsupported pol " + str(pol) #Numba does not support an explicit error message.

    
    
#@jit(nopython=True,cache=True,nogil=True)
@jit(nopython=True,nogil=True)
def _sample_J_func(func, parms,max_rad_1GHz, lmn, freq, ipower):
    #beam_parms = bm
    #beam_parms['ipower'] = 1
    
    if func == 'casa_airy':
        J_sampled = _casa_airy_beam(lmn[0],lmn[1],freq,parms[0].real, parms[1].real,ipower,max_rad_1GHz)
        #J_sampled = 0.5
    elif func == 'airy':
        #J_sampled = 0.5
        J_sampled = _airy_beam(lmn[0],lmn[1],freq,parms[0].real,parms[1].real, ipower)
    else:
        J_sampled = 1
    
    J_sampled_array = np.zeros(4, dtype = numba.float64)
    J_sampled_array[0] = J_sampled
    J_sampled_array[3] = J_sampled
    #J_sampled = np.array([J_sampled,0,0,J_sampled])
    return J_sampled_array.astype(numba.complex128)
    
@jit(nopython=True,cache=True,nogil=True)
def _sample_J_image(bm_J,bm_pa,bm_chan, bm_pol, bm_delta_l,bm_delta_m,pa,freq,lmn):
    """
    Obtain Jones matrix at coordinates lmn.
    Dimensionality refers to the beam model (not visibilities).
    
    Parameters
    ----------
    bm_J: complex np.array, [n_pa, n_chan, n_pol, n_pixels_x, n_pixels_y]
        Input array of Jones matrices for various parallactic angles, channels, and polarizations. 
    bm_pa: int np.array, [n_pa], radians
        Array of available parallactic angle indices
    bm_chan: int np.array, [n_chan], Hertz
        Array of available frequency channel indices
    bm_pol: int np.array, [n_pol]
        Array of available Jones matrix indices (valid values are 0,1,2,3. Jones matrix is flattened row-wise.)
    bm_delta_l: float, radians/pixel
        Radians per pixel along x
    bm_delta_m: float, radians/pixel
        Radians per pixel along y
    pa: float, radians
        Parallactic angle of desired Jones matrix
    freq:  float, Hertz
        Value of frequency of desired Jones matrix
    lmn: float np.array, [3], radians
        l,m,n coordinates of desired Jones matrix 
    
    Returns
    -------
    J: float np.array, [4]
        Interpolated Jones matrix at desired parallactic angle and coordinates
    
    """
    pa_indx = _find_angle_indx(bm_pa,pa)
    chan_indx = _find_val_indx(bm_chan, freq)
    bm_J_sub = bm_J[pa_indx, chan_indx,:,:,:]
    
    freq_scaling_factor = freq/bm_chan[chan_indx] # freq of visibility/freq of beam model
    x_rot, y_rot  = _rot_coord(lmn[0]*freq_scaling_factor,lmn[1]*freq_scaling_factor,pa-bm_pa[pa_indx])
    #xrot = np.ones(1, dtype = numba.float64)
    xrot = np.ones(1, dtype = np.float64)
    xrot[0] = x_rot
    #yrot = np.ones(1, dtype = numba.float64)
    yrot = np.ones(1, dtype = np.float64)
    yrot[0] = y_rot
    #print((xrot/bm[4]) + len(bm_J_sub[0, :, 0])//2)
    #print((yrot/bm[5]) + len(bm_J_sub[0, 0, :])//2)
    J_temp = _interp_array(bm_J_sub, xrot, yrot, bm_delta_l, bm_delta_m)
    
    J = np.zeros((4,),dtype=numba.complex128)
    for i,p in enumerate(bm_pol):
        J[p] = J_temp[i,0]
        #print(i)
        
    return J

@jit(nopython=True,cache=True,nogil=True)
def _sample_J_poli(bm_BPC,bm_chan,bm_pol,pa,freq,lmn,ipower,max_rad_1GHz):
    '''
    bm_BPC: float np.array, [n_chan,n_pol,n_coef]
    '''
    
    chan_indx = _find_val_indx(bm_chan, freq)
    bm_BPC_sub = bm_BPC[chan_indx,:,:]
    
    #freq_scaling_factor = freq/bm_chan[chan_indx] # freq of visibility/freq of beam model
    #J_sampled = _poly_beam(lmn[0]*freq_scaling_factor,lmn[1]*freq_scaling_factor,bm_chan[chan_indx],bm_BPC_sub.real, ipower,max_rad_1GHz)
    J_sampled = _poly_beam(lmn[0],lmn[1],freq,bm_BPC_sub.real, ipower,max_rad_1GHz)
    
    J_sampled_array = np.zeros(4, dtype = numba.float64)
    J_sampled_array[0] = J_sampled
    J_sampled_array[3] = J_sampled
    #J_sampled = np.array([J_sampled,0,0,J_sampled])
    
    #print('In sample_J',J_sampled,bm_chan[chan_indx],freq,lmn)
    
    return J_sampled_array.astype(numba.complex128)

#Non-numba versions:
def _sample_J_func_og(bm,lmn,freq):
    beam_parms = bm
    beam_parms['ipower'] = 1
    
    if beam_parms['func'] == 'casa_airy':
        J_sampled = _casa_airy_beam_njit(lmn,freq,beam_parms)
    elif beam_parms['func'] == 'airy':
        J_sampled = _airy_beam_njit(lmn,freq,beam_parms)
    else:
        J_sampled = 1
    J_sampled = np.array([J_sampled,0,0,J_sampled])
    return J_sampled


def _sample_J_image_og(bm,lmn,freq,pa):
# sample_J(J,pa_,chan_,lmn,chan,pa)
    bm_sub = bm
    if len(bm.pa) > 1:
        pa_indx= _find_angle_indx(bm_sub.pa.values,pa)
        bm_sub = bm_sub.isel(pa=pa_indx)
    else:
        bm_sub = bm_sub.isel(pa=0)
    if len(bm.chan) > 1:
        bm_sub = bm.J.interp(chan=freq,method='nearest')
    else:
        bm_sub = bm_sub.isel(chan=0)
        
    #print('pa values',bm_sub)
    x_rot, y_rot  = _rot_coord(lmn[0],lmn[1],pa-bm_sub.pa.values)
    
#    print(bm_sub)
#    print(lmn,pa,bm_sub.pa.values )
#    print(x_rot,y_rot)
#    print(bm_sub.J.isel(model=0).interp(l=x_rot,m=y_rot,method='linear'))
#    print(bm_sub.J.interp(l=x_rot,m=y_rot,method='linear').values)
    
    return bm_sub.J.interp(l=x_rot,m=y_rot,method='linear').values[0]

@jit(nopython=True,cache=True,nogil=True)
def _make_mueler_mat(J1, J2, mueller_selection):

    #M = np.zeros((4,4),dtype=np.complex)
    M = np.zeros((4,4),dtype=numba.complex128)
    
    for m_flat_indx in mueller_selection:
        #print(m_flat_indx//4,m_flat_indx - 4*(m_flat_indx//4))
        #print(np.where(map_mueler_to_pol[m_flat_indx][0] == pol)[0][0])
        #print(pol, map_mueler_to_pol[m_flat_indx][0])
        #M[m_flat_indx//4,m_flat_indx - 4*(m_flat_indx//4)] = J1[np.where(map_mueler_to_pol[m_flat_indx][0] == pol)[0][0]]*np.conj(J2[np.where(map_mueler_to_pol[m_flat_indx][1] == pol)[0][0]])
        
        M[m_flat_indx//4,m_flat_indx - 4*(m_flat_indx//4)] = J1[map_mueler_to_pol[m_flat_indx,0]]*np.conj(J2[map_mueler_to_pol[m_flat_indx,1]])
            
    return M
    
    #map_mueler_to_pol = np.array([[0,0],[0,1],[1,0],[1,1],[0,2],[0,3],[1,2],[1,3],[2,0],[2,1],[3,0],[3,1],[2,2],[2,3],[3,2],[3,3]])

'''
    #print('pol',pol)
    if inv:
        map_mueler_to_pol = np.array([[3, 3],[3, 2],[2, 3],[2, 2],[3, 1],[3, 0],[2, 1],[2, 0],[1, 3],[1, 2],[0, 3],[0, 2],[1, 1],[1, 0],[0, 1],[0, 0]]) # np.flip(map_mueler_to_pol,axis=0)
        #map_mueler_to_pol = np.array([ [[3, 3],[3, 2],[2, 3],[2, 2]],[[3, 1],[3, 0],[2, 1],[2, 0]],[[1, 3],[1, 2],[0, 3],[0, 2]],[[1, 1],[1, 0],[0, 1],[0, 0]]])
    else:
        map_mueler_to_pol = np.array([[0,0],[0,1],[1,0],[1,1],[0,2],[0,3],[1,2],[1,3],[2,0],[2,1],[3,0],[3,1],[2,2],[2,3],[3,2],[3,3]])
        #map_mueler_to_pol = np.array([[[0,0],[0,1],[1,0],[1,1]],[[0,2],[0,3],[1,2],[1,3]],[[2,0],[2,1],[3,0],[3,1]],[[2,2],[2,3],[3,2],[3,3]]])
        
'''

# Do we need the abs for apeture_parms['zernike_size'].
#    apeture_parms['cell_size'] = 1/(beam_parms['cell_size']*beam_parms['image_size']*apeture_parms['oversampling'])
#    apeture_parms['zernike_size'] = np.floor(np.abs(Dish_Diameter*eta/(apeture_parms['cell_size']*lmbd))) #Why is abs used?
#2.) Currently in ARD-20 the Zernike grid parameters are caculated using (in ZernikeCalc.cc):
#size_zernike = floor((D*eta)/dx)
#delta_zernike = 2.0/size_zernike
#
#By reversing the order the delta of the zernike grid and uv grid can match exactly (no floor operation). The zernike grid will be slightly larger (all cells outside the r=1 will still be nulled).
#delta_zernike = (2.0*dx)/(D*eta)
#size_zernike = ceil((D*eta)/dx)
#Assume ETA is the same for all pol and coef. Only a function of freq. Need to update zpc.zarr format.
  
def _calc_ant_jones(zpc_dataset,j_freq,j_pa,beam_parms):
    pa_prev = -42.0
    freq_prev = -42.0
    
    j_planes = np.zeros((len(j_pa),len(j_freq),len(beam_parms['needed_pol']),beam_parms['image_size'][0],beam_parms['image_size'][1]),np.complex128)
    j_planes_shape = j_planes.shape
    iter_dims_indx = itertools.product(np.arange(j_planes_shape[0]),np.arange(j_planes_shape[1]))
    ic = beam_parms['image_size']//2 #image center pixel
    
    
    #print('j_planes_shape,ic',j_planes_shape,ic)
    
    for i_pa, i_chan in iter_dims_indx:
        #print(i_pa,i_chan)
        pa = j_pa[i_pa]
        beam = zpc_dataset
        freq = j_freq[i_chan]

        if(freq != freq_prev):
            beam_interp = beam.interp(chan=freq,method=beam_parms['zernike_freq_interp'])
            
        dish_diam = beam.dish_diam
        lmbd = c/freq
        eta = beam_interp.ETA[0,0].values #Assume ETA is the same for all pol and coef. Only a function of freq. Need to update zpc.zarr format.
        uv_cell_size = 1/(beam_parms['cell_size']*beam_parms['image_size'])
        zernike_cell = (2.0*uv_cell_size*lmbd)/(dish_diam*eta)
        
        
        if (pa != pa_prev) or (freq != freq_prev) :
            beam_parms['parallactic_angle'] = pa
            image_size = (np.ceil(np.abs(2.0/zernike_cell))).astype(int)
            x_grid, y_grid = _compute_rot_coords(image_size,zernike_cell,pa)
            
            r_grid = np.sqrt(x_grid**2 + y_grid**2)
            
            zernike_size = np.array(x_grid.shape)
        
            ic_z = zernike_size//2
            include_last = (zernike_size%2).astype(int)
        
        #assert zernike_size[0] < beam_parms['conv_size'][0] and zernike_size[1] < gcf_parms['conv_size'][1], "The convolution size " + str(gcf_parms['conv_size']) +" is smaller than the aperture image " + zernike_size + " . Increase conv_size"
        
        #print('x_grid',x_grid)
        #print('y_grid',y_grid)
        
        
        start = time.time()
        for i_pol,pol in enumerate(beam_parms['needed_pol']):
            a = _generate_zernike_surface(beam_interp.ZPC[pol,:].values,x_grid,y_grid)
            a[r_grid > 1] = 0
            j_planes[i_pa, i_chan,i_pol,ic[0]-ic_z[0]:ic[0]+ic_z[0]+include_last[0],ic[1]-ic_z[1]:ic[1]+ic_z[1]+include_last[1]] = a
            j_planes[i_pa, i_chan, i_pol,:,:] = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(j_planes[i_pa, i_chan,i_pol,:,:])))/(beam_parms['image_size'][0]*beam_parms['image_size'][1])
        #print('One pol set',time.time()-start)
        
        #Normalize Jones
        if 3 not in beam_parms['needed_pol']:
            #P_max = np.abs(j_planes[i_pa, i_chan, np.where(beam_parms['needed_pol']==0),j_planes_shape[3]//2,j_planes_shape[4]//2])
            P_max = np.max(np.abs(j_planes[i_pa, i_chan, np.where(beam_parms['needed_pol']==0),:,:]))
            Q_max = P_max
        elif 0 not in beam_parms['needed_pol']:
            #Q_max = np.abs(j_planes[i_pa, i_chan, np.where(beam_parms['needed_pol']==3),j_planes_shape[3]//2,j_planes_shape[4]//2])
            Q_max = np.max(np.abs(j_planes[i_pa, i_chan, np.where(beam_parms['needed_pol']==3),:,:]))
            P_max = Q_max
        else:
            #P_max = np.abs(j_planes[i_pa, i_chan, np.where(beam_parms['needed_pol']==0),j_planes_shape[3]//2,j_planes_shape[4]//2])
            P_max = np.max(np.abs(j_planes[i_pa, i_chan, np.where(beam_parms['needed_pol']==0),:,:]))
            #Q_max = np.abs(j_planes[i_pa, i_chan, np.where(beam_parms['needed_pol']==3),j_planes_shape[3]//2,j_planes_shape[4]//2])
            Q_max = np.max(np.abs(j_planes[i_pa, i_chan, np.where(beam_parms['needed_pol']==3),:,:]))
        j_planes[i_pa, i_chan,:,:,:] = j_planes[i_pa, i_chan,:,:,:]*2/(P_max+Q_max)
        
        pa_prev = pa
        freq_prev = freq
    return j_planes#np.zeros((1,4,2048,2048),dtype=np.complex128)







#####################################################################################

##################Currently Not used functions##################
#@jit(nopython=True,cache=True)
def _outer_product(B1,B2,norm,conj):
    '''
    Input
    B1 2 x 2 x m x n array
    B2 2 x 2 x m x n array
    Output
    M 4 x 4 x m x n
    '''
    
    #assert B1.shape==B2.shape
    
    s = B1.shape
    
    M = np.zeros((4,4,s[2],s[3]),dtype=np.complex128)
    
    indx_b1 = np.array([[[0,0],[0,0],[0,1],[0,1]],[[0,0],[0,0],[0,1],[0,1]],[[1,0],[1,0],[1,1],[1,1]],[[1,0],[1,0],[1,1],[1,1]]])
    indx_b2 = np.array([[[0,0],[0,1],[0,0],[0,1]],[[1,0],[1,1],[1,0],[1,1]],[[0,0],[0,1],[0,0],[0,1]],[[1,0],[1,1],[1,0],[1,1]]])
    #print(indx_b1.shape)
    
    
    for i in range(4):
        for j in range(4):
            #print(indx_b1[i,j,:], ',*,', indx_b2[i,j,:])
            if conj:
                M[i,j,:,:] = B1[indx_b1[i,j,0],indx_b1[i,j,1],:,:] * B2[indx_b2[i,j,0],indx_b2[i,j,1],:,:].conj().T
            else:
                M[i,j,:,:] = B1[indx_b1[i,j,0],indx_b1[i,j,1],:,:] * B2[indx_b2[i,j,0],indx_b2[i,j,1],:,:]
                
            if norm:
                M[i,j,:,:] = M[i,j,:,:]/np.max(np.abs(M[i,j,:,:]))
    
    #print(M.shape)
    return(M)


def _outer_product_conv(B1,B2):
    
#    Input
#    B1 2 x 2 x m x n array
#    B2 2 x 2 x m x n array
#    Output
#    M 4 x 4 x m x n
    
    #assert B1.shape==B2.shape
    
    s = B1.shape
    
    M = np.zeros((4,4,s[2],s[3]),dtype=np.complex128)
    
    indx_b1 = np.array([[[0,0],[0,0],[0,1],[0,1]],[[0,0],[0,0],[0,1],[0,1]],[[1,0],[1,0],[1,1],[1,1]],[[1,0],[1,0],[1,1],[1,1]]])
    indx_b2 = np.array([[[0,0],[0,1],[0,0],[0,1]],[[1,0],[1,1],[1,0],[1,1]],[[0,0],[0,1],[0,0],[0,1]],[[1,0],[1,1],[1,0],[1,1]]])
    
    for i in range(4):
        for j in range(4):
            M[i,j,:,:] = signal.fftconvolve(B1[indx_b1[i,j,0],indx_b1[i,j,1],:,:], B2[indx_b2[i,j,0],indx_b2[i,j,1],:,:],mode='same')
    
    #print(M.shape)
    return(M)

    
def _make_flat(B):
    '''
    B 2x2xmxn
    B_flat 2mx2n
    '''
    s = B.shape
    B_flat = np.zeros((s[2]*s[0],s[3]*s[1]),dtype=complex)
    
    
    for i in range(s[0]):
        for j in range(s[1]):
            i_start = i*s[2]
            i_end = (i+1)*s[3]
            j_start = j*s[2]
            j_end = (j+1)*s[3]
            B_flat[i_start:i_end,j_start:j_end] = B[i,j,:,:]
            #print(B[i,j,1024,1024],np.abs(B[i,j,1024,1024]))
    return B_flat
    
    
def _make_flat_casa(B):
    '''
    B mxnx16
    B_flat 4mx4n
    '''
    s = B.shape
    B_flat = np.zeros((s[0]*4,s[1]*4),dtype=complex)
    
    #indx = np.array([[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2],[0,3],[1,3],[2,3],[3,3]])
    indx = np.array([[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]]) #saved as rows
    
    for c,i in enumerate(indx):
        #print(c,i)
        i_start = i[0]*s[0]
        i_end = (i[0]+1)*s[0]
        j_start = i[1]*s[1]
        j_end = (i[1]+1)*s[1]
        B_flat[i_start:i_end,j_start:j_end] = B[:,:,c].T
        #print(B[1024,1024,c],np.abs(B[1024,1024,c]))
    return B_flat



def _make_flat_casa(B):
    '''
    B mxnx16
    B_flat 4mx4n
    '''
    s = B.shape
    B_flat = np.zeros((s[0]*4,s[1]*4),dtype=complex)
    
    #indx = np.array([[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2],[0,3],[1,3],[2,3],[3,3]])
    indx = np.array([[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]]) #saved as rows
    
    for c,i in enumerate(indx):
        #print(c,i)
        i_start = i[0]*s[0]
        i_end = (i[0]+1)*s[0]
        j_start = i[1]*s[1]
        j_end = (i[1]+1)*s[1]
        B_flat[i_start:i_end,j_start:j_end] = B[:,:,c].T
        #print(B[1024,1024,c],np.abs(B[1024,1024,c]))
    return B_flat
