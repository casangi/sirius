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
from ._sirius_utils._beam_utils import _calc_ant_jones, _calc_resolution
from ._sirius_utils._calc_parallactic_angles import _calc_parallactic_angles, _find_optimal_set_angle
from ._parm_utils._check_beam_parms import _check_beam_parms

def calc_zpc_beam(zpc_xds,parallactic_angles,freq_chan,beam_parms,check_parms=True):
    """
    Calculates an antenna apertures from Zernike polynomial coefficients, and then Fourier transform it to obtain the antenna beam image.
    The beam image dimensionality is [pa (paralactic angle), chan (channel), pol (polarization), l (orthographic/synthesis projection of directional cosine), m (orthographic/synthesis projection of directional cosine)].

    Parameters
    ----------
    zpc_xds: xr.Dataset
        A Zernike polynomial coefficient xr.Datasets. Available models can be found in sirius_data/zernike_dish_models/data.
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
    beam_parms['fov_scaling']: int, default=15
        Used to scale the size of the beam image, which is given by fov_scaling*(1.22 *c/(dish_diam*frequency)).
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
    
    
    pb_freq = freq_chan
    pb_pa = parallactic_angles
    
    min_delta = _calc_resolution(pb_freq,zpc_xds.dish_diam,_beam_parms)
    #print('min_delta',min_delta)
    _beam_parms['cell_size'] = np.array([-min_delta,min_delta]) #- sign?
  
    map_mueler_to_pol = np.array([[0,0],[0,1],[1,0],[1,1],[0,2],[0,3],[1,2],[1,3],[2,0],[2,1],[3,0],[3,1],[2,2],[2,3],[3,2],[3,3]])
    _beam_parms['needed_pol'] = np.unique(np.ravel(map_mueler_to_pol[_beam_parms['mueller_selection']]))
    
    assert (0 in _beam_parms['mueller_selection']) or (15 in _beam_parms['mueller_selection']), "Mueller element 0 or 15 must be selected."
    
    pb_planes = _calc_ant_jones(zpc_xds,pb_freq,pb_pa,_beam_parms)
    
    image_size = _beam_parms['image_size']
    image_center = image_size//2
    cell_size = _beam_parms['cell_size']
    
    image_center = np.array(image_size)//2
    l = np.arange(-image_center[0], image_size[0]-image_center[0])*cell_size[0]
    m = np.arange(-image_center[1], image_size[1]-image_center[1])*cell_size[1]
    
    coords = {'chan':pb_freq, 'pa': pb_pa, 'pol': _beam_parms['needed_pol'],'l':l,'m':m}

    J_xds = xr.Dataset()
    J_xds = J_xds.assign_coords(coords)
    
    J_xds['J'] = xr.DataArray(pb_planes, dims=['pa','chan','pol','l','m'])
    
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
    beam_parms['fov_scaling']: int, default=15
        Used to scale the size of the beam image, which is given by fov_scaling*(1.22 *c/(dish_diam*frequency)).
    beam_parms['zernike_freq_interp']: str, default='nearest', options=['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
        What interpolation method to use for Zernike polynomial coefficients.
        
    Returns
    -------
    J_xds: xr.Dataset
        An xds that contains the image of the per antenna beam as a function of [pa (paralactic angle), chan (channel), pol (polarization), l (orthographic/synthesis projection of directional cosine), m (orthographic/synthesis projection of directional cosine)]. Should not be confused with the primary beam, which is the beam for a baseline and is equal to the product of two antenna beams.
    """
    
    #Calculate parallactic angles.
    pa = _calc_parallactic_angles(time_str,site_location,phase_center_ra_dec)
    pa_subset,vals_dif = _find_optimal_set_angle(pa[:,None],beam_parms['pa_radius'] )
    
    _beam_parms = copy.deepcopy(beam_parms)

    # If beam model is a Zernike polynomial coefficient xr.Datasets convert it to an image.
    eval_beam_models = []
    for bm in beam_models:
        if 'ZC' in bm: #check for zpc files
            J_xds = calc_zpc_beam(bm,pa_subset,freq_chan,_beam_parms,check_parms)
            J_xds.attrs = bm.attrs
            eval_beam_models.append(J_xds)
        else:
            eval_beam_models.append(bm)
    
    return eval_beam_models, pa
