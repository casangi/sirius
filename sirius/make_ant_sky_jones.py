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
c = 299792458
from ._sirius_utils._direction_rotate import _calc_rotation_mats, _cs_calc_rotation_mats
from ._sirius_utils._apply_primary_beam import _apply_casa_airy_pb
from ._sirius_utils._ant_jones_term import _calc_ant_jones
from sirius._sirius_utils._calc_parallactic_angles import _calc_parallactic_angles, _find_optimal_set_angle
import itertools
import matplotlib.pyplot as plt
from PIL import Image
import xarray as xr
import copy

def evaluate_beam_models(beam_models,beam_parms,freq_chan,phase_center_ra_dec,time_str,site):
    pa = _calc_parallactic_angles(time_str,site,phase_center_ra_dec)
    pa_subset,vals_dif = _find_optimal_set_angle(pa[:,None],beam_parms['pa_radius'] )
    
    _beam_parms = copy.deepcopy(beam_parms)
    _beam_parms['pa'] =  pa_subset
    _beam_parms['freq'] = freq_chan
    
    eval_beam_models = []
    for bm in beam_models:
        if 'ZC' in bm: #check for zpc files
            J_xds = make_ant_sky_jones([bm],_beam_parms) #[None,None,:,:,:,:]
            eval_beam_models.append(J_xds)
        else:
            eval_beam_models.append(bm)
    
    return eval_beam_models, pa
    


#def create_pb(a_parm_indx,list_zpc_dataset,gcf_a_freq,gcf_a_pa,gcf_parms,grid_parms):
def make_ant_sky_jones(list_zpc_dataset,pb_parms):
    pb_freq = pb_parms['freq']
    pb_pa = pb_parms['pa']
    
    min_delta = _calc_resolution(pb_freq,list_zpc_dataset,pb_parms)
    print(min_delta)
    pb_parms['cell_size'] = np.array([-min_delta,min_delta]) #- sign?
  
    map_mueler_to_pol = np.array([[0,0],[0,1],[1,0],[1,1],[0,2],[0,3],[1,2],[1,3],[2,0],[2,1],[3,0],[3,1],[2,2],[2,3],[3,2],[3,3]])
    pb_parms['needed_pol'] = np.unique(np.ravel(map_mueler_to_pol[pb_parms['mueller_selection']]))
    
    assert (0 in pb_parms['mueller_selection']) or (15 in pb_parms['mueller_selection']), "Mueller element 0 or 15 must be selected."
    
    pb_planes = _calc_ant_jones(list_zpc_dataset,pb_freq,pb_pa,pb_parms,pb_parms)
    
    image_size = pb_parms['image_size']
    image_center = image_size//2
    cell_size = 1/(pb_parms['cell_size']*pb_parms['image_size'])
    
    image_center = np.array(image_size)//2
    l = np.arange(-image_center[0], image_size[0]-image_center[0])*cell_size[0]
    m = np.arange(-image_center[1], image_size[1]-image_center[1])*cell_size[1]
    
    
    coords = {'chan':pb_freq, 'pa': pb_pa, 'pol': pb_parms['needed_pol'],'l':l,'m':m}

    J_xds = xr.Dataset()
    J_xds = J_xds.assign_coords(coords)
    
    J_xds['J'] = xr.DataArray(pb_planes, dims=['model','pa','chan','pol','l','m'])

    return J_xds
    
    
def _calc_resolution(pb_freq,list_zpc_xds,pb_parms):
    c = 299792458
    

    min_delta = 42
    for zpc_xds in list_zpc_xds:
        fov = pb_parms['fov_scaling']*(1.22 * c / (zpc_xds.dish_diam*pb_freq))
        delta = min(min(fov/pb_parms['image_size'][0]),min(fov/pb_parms['image_size'][1]))
        if delta < min_delta:
            min_delta =  delta
    return min_delta
