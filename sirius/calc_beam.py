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

def calc_zpc_beam(zpc_xds,beam_parms):
    """
    .
    Parameters
    ----------
    point_source_flux : np.array
    Returns
    -------
    vis : np.array
    uvw : np.array
    """
    _beam_parms = copy.deepcopy(beam_parms)
    
    pb_freq = _beam_parms['freq']
    pb_pa = _beam_parms['pa']
    
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
    

def evaluate_beam_models(beam_models,beam_parms,freq_chan,phase_center_ra_dec,time_str,site_location):
    pa = _calc_parallactic_angles(time_str,site_location,phase_center_ra_dec)
    pa_subset,vals_dif = _find_optimal_set_angle(pa[:,None],beam_parms['pa_radius'] )
    
    _beam_parms = copy.deepcopy(beam_parms)
    _beam_parms['pa'] =  pa_subset
    _beam_parms['freq'] = freq_chan
    
    eval_beam_models = []
    for bm in beam_models:
        if 'ZC' in bm: #check for zpc files
            J_xds = calc_zpc_beam(bm,_beam_parms) #[None,None,:,:,:,:]
            J_xds.attrs = bm.attrs
            eval_beam_models.append(J_xds)
        else:
            eval_beam_models.append(bm)
    
    return eval_beam_models, pa
