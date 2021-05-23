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
from direction_rotate import _calc_rotation_mats
import matplotlib.pyplot as plt

def calc_vis(uvw,vis_data_shape,point_source_flux,point_source_ra_dec,pointing_ra_dec,freq_chan):
    '''
    point_source_flux: [n_time, n_chan, n_pol, n_point_sources] (singleton: n_time, n_chan, n_pol)
    point_source_ra_dec:  [n_time, n_point_sources, 2]          (singleton: n_time)
    pointing_ra_dec:  [n_time, n_baseline, 2]                   (singleton: n_time, n_baseline)
    '''
    n_time, n_baseline, n_chan, n_pol = vis_data_shape
    vis_data = np.zeros(vis_data_shape,dtype=np.complex)
    # xxxx
    n_pol = 1
    n_point_source = point_source_ra_dec.shape[1]
    
    rotation_parms = {}
    rotation_parms['common_tangent_reprojection'] = False
    rotation_parms['reproject'] = True
    

    if pointing_ra_dec.shape[0] == 1: f_pt_time =  n_time
    
    #Check all dims are either 1 or n
    f_pt_time = n_time if pointing_ra_dec.shape[0] == 1 else 1
    f_pt_baseline = n_baseline if pointing_ra_dec.shape[1] == 1 else 1
    f_ps_time = n_time if point_source_ra_dec.shape[0] == 1 else 1
    
    
    f_sf_time = n_time if point_source_flux.shape[0] == 1 else 1
    f_sf_chan = n_chan if point_source_flux.shape[1] == 1 else 1
    f_sf_pol = n_pol if point_source_flux.shape[2] == 1 else 1
    
    prev_ra_dec_in = np.array([0.0,0.0])
    prev_ra_dec_out = np.array([0.0,0.0])

    for i_time in range(n_time):
        for i_baseline in range(n_baseline):
            ra_dec_in = pointing_ra_dec[i_time//f_pt_time,i_baseline//f_pt_baseline,:]

            for i_point_source in range(n_point_source):
                ra_dec_out = point_source_ra_dec[i_time//f_ps_time,i_point_source,:]
                
                if not(np.array_equal(prev_ra_dec_in, ra_dec_in) and np.array_equal(prev_ra_dec_out, ra_dec_out)):
                    uvw_rotmat, uvw_proj_rotmat, phase_rotation = _calc_rotation_mats(ra_dec_in, ra_dec_out, rotation_parms)
                    #print('ra_dec_in,ra_dec_in',ra_dec_in,ra_dec_in)
                    #print('uvw_rotmat', uvw_rotmat)
                    print('phase_rotation', phase_rotation)
                
                # Right Handed -> Left Handed and (ant2-ant1) -> (ant1-ant2)
                uvw[i_time,i_baseline,0] = -uvw[i_time,i_baseline,0]
                uvw[i_time,i_baseline,1] = -uvw[i_time,i_baseline,1]
                
                #print('uvw*uvrot_p ',uvw[i_time,i_baseline,:]@uvw_rotmat)
                phase = 2*1j*np.pi*phase_rotation@(uvw[i_time,i_baseline,:]@uvw_rotmat)
                #print('phase', phase)
                #uvw_for_func = uvw[i_time,i_baseline,:]@uvw_proj_rotmat #used for non point source functions

                prev_ra_dec_in = ra_dec_in
                prev_ra_dec_out = ra_dec_out
                
                for i_chan in range(n_chan):
                    phase_scaled = phase*freq_chan[i_chan]/c
                    #print('phase',phase_scaled,phase,freq_chan[i_chan]/c)
                    for i_pol in range(n_pol):
                        flux = point_source_flux[i_time//f_sf_time, i_chan//f_sf_chan, i_pol//f_sf_pol, i_point_source]
                        
                        vis_data[i_time,i_baseline,i_chan,i_pol] = vis_data[i_time,i_baseline,i_chan,i_pol] + flux*np.exp(phase_scaled)
                        #print('vis',vis_data[i_time,i_baseline,i_chan,i_pol])

    return vis_data


#@jit(nopython=True,cache=True,nogil=True)
def _directional_cosine(ra_dec):
   '''
   # In https://arxiv.org/pdf/astro-ph/0207413.pdf see equation 160
   ra_dec (RA,DEC)
   '''
   
   #lmn = np.zeros((3,),dtype=numba.f8)
   lmn = np.zeros((3,))
   lmn[0] = np.cos(ra_dec[0])*np.cos(ra_dec[1])
   lmn[1] = np.sin(ra_dec[0])*np.cos(ra_dec[1])
   lmn[2] = np.sin(ra_dec[1])
   return lmn
