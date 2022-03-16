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
import xarray as xr
import copy
from sirius._sirius_utils._beam_utils import _calc_ant_jones, _calc_resolution, _pol_code_to_index, _index_to_pol_code
from sirius._sirius_utils._calc_parallactic_angles import _calc_parallactic_angles_astropy, _find_optimal_set_angle
from sirius._parm_utils._check_beam_parms import _check_beam_parms
from sirius._sirius_utils._beam_funcs import _3d_casa_airy_beam, _3d_airy_beam, _3d_poly_beam
from sirius_data._constants import map_mueler_to_pol, c

def calc_zpc_beam(zpc_xds,parallactic_angles,freq_chan,beam_parms,check_parms=True):
    """
    Calculates an antenna apertures from Zernike polynomial coefficients, and then Fourier transform it to obtain the antenna beam image.
    The beam image dimensionality is [pa (paralactic angle), chan (channel), pol (polarization), l (orthographic/synthesis projection of directional cosine), m (orthographic/synthesis projection of directional cosine)].

    Parameters
    ----------
    zpc_xds: xr.Dataset
        A Zernike polynomial coefficient xr.Datasets. Available models can be found in sirius_data/zernike_dish_models/data.
    parallactic_angles: float np.array, [n_pa], radians
        An array of the parallactic angles for which to calculate the antenna beams.
    freq_chan: float np.array, [n_chan], Hz
        Channel frequencies.
    beam_parms: dict
    beam_parms['mueller_selection']: int np.array, default=np.array([ 0, 5, 10, 15])
        The elements in the 4x4 beam Mueller matrix to use. The elements are numbered row wise.
        For example [ 0, 5, 10, 15] are the diagonal elements.
    beam_parms['pa_radius']: float, default=0.2, radians
        The change in parallactic angle that will trigger the calculation of a new beam when using Zernike polynomial aperture models.
    beam_parms['image_size']: int np.array, default=np.array([1000,1000])
        This parameter should rarely be modified. Size of the beam image generated from the Zernike polynomial coefficients.
    beam_parms['fov_scaling']: int, default=1.2
        This parameter should rarely be modified. Used to determine the cell size of the beam image so that it lies within the image that is generated.
    beam_parms['zernike_freq_interp']: str, default='nearest', options=['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
        What interpolation method to use for Zernike polynomial coefficients.
    check_parms: bool
        Check input parameters and asign defaults.
        
    Returns
    -------
    J_xds: xr.Dataset
        An xds that contains the image of the per antenna beam as a function of [pa (paralactic angle), chan (channel), pol (polarization), l (orthographic/synthesis projection of directional cosine), m (orthographic/synthesis projection of directional cosine)]. Should not be confused with the primary beam, which is the beam for a baseline and is equal to the product of two antenna beams.
    """
    _beam_parms = copy.deepcopy(beam_parms)
    
    if check_parms: assert(_check_beam_parms(_beam_parms)), "######### ERROR: beam_parms checking failed."
    
    #min_delta = _calc_resolution(freq_chan,zpc_xds.dish_diam,_beam_parms)
    #_beam_parms['cell_size'] = np.array([-min_delta,min_delta]) #- sign?
    delta = (zpc_xds.attrs['max_rad_1GHz']/np.min(freq_chan/10**9))/_beam_parms['image_size']
    _beam_parms['cell_size'] = np.array([-delta[0],delta[1]])*_beam_parms['fov_scaling'] #- sign?
    
    map_mueler_to_pol = np.array([[0,0],[0,1],[1,0],[1,1],[0,2],[0,3],[1,2],[1,3],[2,0],[2,1],[3,0],[3,1],[2,2],[2,3],[3,2],[3,3]])
    _beam_parms['needed_pol'] = np.unique(np.ravel(map_mueler_to_pol[_beam_parms['mueller_selection']]))
    
    assert (0 in _beam_parms['mueller_selection']) or (15 in _beam_parms['mueller_selection']), "Mueller element 0 or 15 must be selected."
    
    J = _calc_ant_jones(zpc_xds,freq_chan,parallactic_angles,_beam_parms)
    
    image_size = _beam_parms['image_size']
    image_center = image_size//2
    cell_size = _beam_parms['cell_size']
    
    image_center = np.array(image_size)//2
    l = np.arange(-image_center[0], image_size[0]-image_center[0])*cell_size[0]
    m = np.arange(-image_center[1], image_size[1]-image_center[1])*cell_size[1]

    coords = {'chan':freq_chan, 'pa': parallactic_angles, 'pol': _index_to_pol_code(_beam_parms['needed_pol'],zpc_xds.pol.values),'l':l,'m':m}
    
    J_xds = xr.Dataset()
    J_xds = J_xds.assign_coords(coords)
    
    J_xds['J'] = xr.DataArray(J, dims=['pa','chan','pol','l','m'])
    
    return J_xds
    
def calc_beam(func_parms,freq_chan,beam_parms,check_parms=True):

    _beam_parms = copy.deepcopy(beam_parms)
    _func_parms = copy.deepcopy(func_parms)
    if check_parms: assert(_check_beam_parms(_beam_parms)), "######### ERROR: beam_parms checking failed."
    
    
    #min_delta = _calc_resolution(freq_chan,_func_parms['dish_diam'],_beam_parms)
    #_beam_parms['cell_size'] = np.array([-min_delta,min_delta]) #- sign?
    delta = (func_parms['max_rad_1GHz']/np.min(freq_chan/10**9))/_beam_parms['image_size']
    _beam_parms['cell_size'] = np.array([-delta[0],delta[1]])*_beam_parms['fov_scaling'] #- sign?
    print(_beam_parms['image_size'],_beam_parms['cell_size'],_beam_parms['fov_scaling'],func_parms['max_rad_1GHz'])
    
    image_size = _beam_parms['image_size']
    image_center = image_size//2
    cell_size = _beam_parms['cell_size']
    
    image_center = np.array(image_size)//2
    l = np.arange(-image_center[0], image_size[0]-image_center[0])*cell_size[0]
    m = np.arange(-image_center[1], image_size[1]-image_center[1])*cell_size[1]
    
    ipower = 1
    if _func_parms['func'] == 'casa_airy':
        J = _3d_casa_airy_beam(l,m,freq_chan,_func_parms['dish_diam'], _func_parms['blockage_diam'],ipower,_func_parms['max_rad_1GHz'])
    elif _func_parms['func'] == 'airy':
        J = _3d_airy_beam(l,m,freq_chan,_func_parms['dish_diam'], _func_parms['blockage_diam'], ipower)

    coords = {'chan':freq_chan, 'pa': [0], 'pol': [0],'l':l,'m':m}
    
    J_xds = xr.Dataset()
    J_xds = J_xds.assign_coords(coords)
    
    J_xds['J'] = xr.DataArray(J[None,:,None,:,:], dims=['pa','chan','pol','l','m'])
    
    return J_xds
    
def calc_bpc_beam(bpc_xds,freq_chan,beam_parms,check_parms=True):

    _beam_parms = copy.deepcopy(beam_parms)
    if check_parms: assert(_check_beam_parms(_beam_parms)), "######### ERROR: beam_parms checking failed. "
    
    
    #min_delta = _calc_resolution(freq_chan,bpc_xds.attrs['dish_diam'],_beam_parms)
    #_beam_parms['cell_size'] = np.array([-min_delta,min_delta]) #- sign?
    delta   = (bpc_xds.attrs['max_rad_1GHz']/np.min(freq_chan/10**9))/_beam_parms['image_size']
    _beam_parms['cell_size'] = np.array([-delta[0],delta[1]])*_beam_parms['fov_scaling']
    
    #print('cell_size ',_beam_parms['cell_size'])
    
    image_size = _beam_parms['image_size']
    image_center = image_size//2
    cell_size = _beam_parms['cell_size']
    
    image_center = np.array(image_size)//2
    l = np.arange(-image_center[0], image_size[0]-image_center[0])*cell_size[0]
    m = np.arange(-image_center[1], image_size[1]-image_center[1])*cell_size[1]
    
    ipower = 1
    J = _3d_poly_beam(l,m,freq_chan,bpc_xds.BPC.values,ipower,bpc_xds.attrs['max_rad_1GHz'])

    coords = {'chan':freq_chan, 'pa': [0], 'pol': [0],'l':l,'m':m}
    
    J_xds = xr.Dataset()
    J_xds = J_xds.assign_coords(coords)
    
    J_xds['J'] = xr.DataArray(J[None,:,None,:,:], dims=['pa','chan','pol','l','m'])
    
    return J_xds


def evaluate_beam_models(beam_models,time_str,freq_chan,phase_center_ra_dec,site_location,beam_parms,check_parms=True):
    """
    Loops over beam_models and converts each Zernike polynomial coefficient xr.Datasets to an antenna beam image. The beam image dimensionality is [pa (paralactic angle), chan (channel), pol (polarization), l (orthographic/synthesis projection of directional cosine), m (orthographic/synthesis projection of directional cosine)]. The parallactic angles are also calculated for each date-time in time_str at the site_location and with a right ascension declination in phase_center_ra_dec. A subset of parallactic angles are used in the pa coordinate of the beam image, where all pa values are within beam_parms['pa_radius'] radians.

    Parameters
    ----------
    beam_models: list
        List of beam models to use. Beam models can be any combination of function parameter dictionaries, image xr.Datasets or Zernike polynomial coefficient xr.Datasets (models can be found in sirius_data/zernike_dish_models/data).
    time_str: str np.array, [n_time], 'YYYY-MM-DDTHH:MM:SS.SSS'
        Time series. Example '2019-10-03T19:00:00.000'.
    freq_chan: float np.array, [n_chan], Hz
        Channel frequencies.
    phase_center_ra_dec: float np.array, [n_time, 2], (singleton: n_time), radians
        Phase center of array.
    site_location: dict
        A dictionary with the location of telescope. For example [{'m0': {'unit': 'm', 'value': -1601185}, 'm1': {'unit': 'm', 'value': -5041977}, 'm2': {'unit': 'm', 'value': 3554875}, 'refer': 'ITRF', 'type': 'position'}]. The site location of telescopes can be found in site_pos attribute of the xarray dataset of the radio telescope array layout (see zarr files in sirius_data/telescope_layout/data/).
    parallactic_angles: float np.array, radians
        An array of the parallactic angles for which to calculate the antenna beams.
    freq_chan: float np.array, [n_chan], Hz
        Channel frequencies.
    beam_parms: dict
    beam_parms['mueller_selection']: int np.array, default=np.array([ 0, 5, 10, 15])
        The elements in the 4x4 beam Mueller matrix to use. The elements are numbered row wise.
        For example [ 0, 5, 10, 15] are the diagonal elements.
    beam_parms['pa_radius']: float, default=0.2, radians
        The change in parallactic angle that will trigger the calculation of a new beam when using Zernike polynomial aperture models.
    beam_parms['image_size']: int np.array, default=np.array([1000,1000])
        Size of the beam image generated from the Zernike polynomial coefficients.
    beam_parms['fov_scaling']: int, default=1.2
        This parameter should rarely be modified. Used to determine the cell size of the beam image so that it lies within the image that is generated.
    beam_parms['zernike_freq_interp']: str, default='nearest', options=['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
        What interpolation method to use for Zernike polynomial coefficients.
        
    Returns
    -------
    J_xds: xr.Dataset
        An xds that contains the image of the per antenna beam as a function of [pa (paralactic angle), chan (channel), pol (polarization), l (orthographic/synthesis projection of directional cosine), m (orthographic/synthesis projection of directional cosine)]. Should not be confused with the primary beam, which is the beam for a baseline and is equal to the product of two antenna beams.
    """
    
    #Calculate parallactic angles.
    pa = _calc_parallactic_angles_astropy(time_str,np.array([site_location['m0']['value'],site_location['m1']['value'],site_location['m2']['value']]),phase_center_ra_dec)
    #print('pa',pa*180/np.pi)
    pa_subset,_,_ = _find_optimal_set_angle(pa,beam_parms['pa_radius'] )
    #print('pa_subset',pa_subset*180/np.pi)
    
    _beam_parms = copy.deepcopy(beam_parms)

    # If beam model is a Zernike polynomial coefficient xr.Datasets convert it to an image.
    eval_beam_models = []
    for bm in beam_models:
        if 'ZPC' in bm: #check for zpc files
            J_xds = calc_zpc_beam(bm,pa_subset,freq_chan,_beam_parms,check_parms)
            J_xds.attrs = bm.attrs
            eval_beam_models.append(J_xds)
        else:
            eval_beam_models.append(bm)
    
    return eval_beam_models, pa


def make_mueler_mat(J_xds1, J_xds2, mueller_selection):
    #print(J_xds1.pol,J_xds1.pol.values)
    pol_indx = _pol_code_to_index(J_xds1.pol.values)
    pol1 = J_xds1.pol.values
    pol2 = J_xds2.pol.values
    #pa,chan,l,m must match
    J1_shape = J_xds1.J.shape
    J2_shape = J_xds2.J.shape
    
    #print(J1_shape,J2_shape)
    
    M_shape = J1_shape[0:2] + (len(mueller_selection),) + J1_shape[3:5]
    M = np.zeros(M_shape,np.complex)
    m_sel = mueller_selection
    pol1=np.zeros(len(mueller_selection))
    pol2=np.zeros(len(mueller_selection))

    for i,m_flat_indx in enumerate(mueller_selection):
        #print(m_flat_indx,m_flat_indx//4,m_flat_indx - 4*(m_flat_indx//4))
        #print(map_mueler_to_pol[m_flat_indx,0],map_mueler_to_pol[m_flat_indx,1])
        #print(np.where(pol_indx == map_mueler_to_pol[m_flat_indx,0])[0][0],np.where(pol_indx == map_mueler_to_pol[m_flat_indx,1])[0][0])
        M[:,:,i,:,:] = J_xds1.J[:,:,np.where(pol_indx == map_mueler_to_pol[m_flat_indx,0])[0][0],:,:]*np.conj(J_xds2.J[:,:,np.where(pol_indx == map_mueler_to_pol[m_flat_indx,1])[0][0],:,:])
        pol1[i]=_index_to_pol_code(map_mueler_to_pol[m_flat_indx,0],J_xds1.pol.values)
        pol2[i]=_index_to_pol_code(map_mueler_to_pol[m_flat_indx,1],J_xds2.pol.values)
    '''
    M_shape = J1_shape[0:2] + (4,4) + J1_shape[3:5]
    M = np.zeros(M_shape,np.complex)

    for m_flat_indx in mueller_selection:
        #print(m_flat_indx,m_flat_indx//4,m_flat_indx - 4*(m_flat_indx//4))
        #print(map_mueler_to_pol[m_flat_indx,0],map_mueler_to_pol[m_flat_indx,1])
        #print(np.where(pol_indx == map_mueler_to_pol[m_flat_indx,0])[0][0],np.where(pol_indx == map_mueler_to_pol[m_flat_indx,1])[0][0])
        M[:,:,m_flat_indx//4,m_flat_indx - 4*(m_flat_indx//4),:,:] = J_xds1.J[:,:,np.where(pol_indx == map_mueler_to_pol[m_flat_indx,0])[0][0],:,:]*np.conj(J_xds2.J[:,:,np.where(pol_indx == map_mueler_to_pol[m_flat_indx,1])[0][0],:,:])
    '''
        
    l = J_xds1.l
    m = J_xds1.m
    freq_chan = J_xds1.chan
    parallactic_angles = J_xds1.pa
    
    coords = {'chan':freq_chan, 'pa': parallactic_angles,'m_sel': m_sel,'pol1':('m_sel',pol1),'pol2':('m_sel',pol2),'l':l,'m':m}
    
    M_xds = xr.Dataset()
    M_xds = M_xds.assign_coords(coords)
    
    M_xds['M'] = xr.DataArray(M, dims=['pa','chan','m_sel','l','m'])
    
        
            
    return M_xds
    
