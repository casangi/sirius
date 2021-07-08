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
import matplotlib.pyplot as plt

def calc_vis(uvw,vis_data_shape,point_source_flux,point_source_ra_dec,pointing_ra_dec,array_phase_center,ANTENNA1,ANTENNA2,n_ant,freq_chan,pb_parms):
    '''
    point_source_flux: [n_time, n_chan, n_pol, n_point_sources] (singleton: n_time, n_chan, n_pol)
    point_source_ra_dec:  [n_time, n_point_sources, 2]          (singleton: n_time)
    pointing_ra_dec:  [n_time, n_ant, 2]                   (singleton: n_time, n_ant)
    Singleton: Can have dimension = 1
    Warning: If mosaic, do not use n_time singleton with n_ant
    '''
    pb_parms['ipower'] = 1
    n_time, n_baseline, n_chan, n_pol = vis_data_shape
    vis_data = np.zeros(vis_data_shape,dtype=np.complex)
    n_point_source = point_source_ra_dec.shape[1]
    
    rotation_parms = {}
    rotation_parms['reproject'] = True
    rotation_parms['common_tangent_reprojection'] = False

    if pointing_ra_dec.shape[0] == 1: f_pt_time =  n_time
    
    #Check all dims are either 1 or n
    f_pt_time = n_time if pointing_ra_dec.shape[0] == 1 else 1
    f_pt_ant = n_ant if pointing_ra_dec.shape[1] == 1 else 1
    f_ps_time = n_time if point_source_ra_dec.shape[0] == 1 else 1
    n_ant_bool = bool(pointing_ra_dec.shape[1] != 1)
    
    f_sf_time = n_time if point_source_flux.shape[0] == 1 else 1
    f_sf_chan = n_chan if point_source_flux.shape[1] == 1 else 1
    f_sf_pol = n_pol if point_source_flux.shape[2] == 1 else 1
    
    prev_ra_dec_in = np.array([0.0,0.0])
    prev_ra_dec_out = np.array([0.0,0.0])
    
    #if autocorr:
    #    n_baseline = (n_ant**2 + n_ant)/2
    #else:
    #    n_baseline = (n_ant**2 - n_ant)/2
    
    antenna_baselines = np.concatenate((np.arange(0, n_baseline, 1).reshape((1, n_baseline)), ANTENNA1.reshape((1, n_baseline)), ANTENNA2.reshape((1, n_baseline))), axis = 0)
    

    for i_time in range(n_time):
        
        ra_dec_in = array_phase_center[i_time//f_pt_time]
        
        for i_baseline in range(n_baseline):
            #ra_dec_in = pointing_ra_dec[i_time//f_pt_time,i_baseline//f_pt_ant,:]
            if n_ant_bool:
                i_ant_1 = antenna_baselines[1, i_baseline//f_pt_baseline]
                ra_dec_in_1 = pointing_ra_dec[i_time//f_pt_time,i_ant_1//f_pt_ant,:]
                i_ant_2 = antenna_baselines[2, i_baseline//f_pt_baseline]
                ra_dec_in_2 = pointing_ra_dec[i_time//f_pt_time,i_ant_2//f_pt_ant,:]
            
            for i_point_source in range(n_point_source):
                ra_dec_out = point_source_ra_dec[i_time//f_ps_time,i_point_source,:]
                
                if not(np.array_equal(prev_ra_dec_in, ra_dec_in) and np.array_equal(prev_ra_dec_out, ra_dec_out)):
                    uvw_rotmat, lmn_rot = _calc_rotation_mats(ra_dec_in, ra_dec_out, rotation_parms)
                if n_ant_bool:
                    if not(np.array_equal(prev_ra_dec_in, ra_dec_in_1) and np.array_equal(prev_ra_dec_out, ra_dec_out)):
                        lmn_rot_1 = _directional_cosine(ra_dec_in_1, ra_dec_out, rotation_parms)
                    if not(np.array_equal(prev_ra_dec_in, ra_dec_in_2) and np.array_equal(prev_ra_dec_out, ra_dec_out)) and n_ant_bool:
                        lmn_rot_2 = _directional_cosine(ra_dec_in_2, ra_dec_out, rotation_parms)
                        #pb_scale = apply_airy_pb(pb_parms)
                    #uvw_rotmat, uvw_proj_rotmat, lmn_rot = _cs_calc_rotation_mats(ra_dec_in,ra_dec_out,rotation_parms)
                    
                # If using CASA functions (_cs): Right Handed -> Left Handed and (ant2-ant1) -> (ant1-ant2)
#                uvw[i_time,i_baseline,0] = -uvw[i_time,i_baseline,0]
#                uvw[i_time,i_baseline,1] = -uvw[i_time,i_baseline,1]
                
                
                phase = 2*1j*np.pi*lmn_rot@(uvw[i_time,i_baseline,:]@uvw_rotmat)
                
                prev_ra_dec_in = ra_dec_in
                prev_ra_dec_out = ra_dec_out
                
                for i_chan in range(n_chan):
                    #Add trigger for % change in frequncy (use mosaic gridder logic) and check for change in direction
                    #Add pb_scales array that temp stores pb scales
                    if np.logical_and(pb_parms['pb_func'] == 'casa_airy', n_ant_bool):
                        #lm_temp = np.array([-0.00156774,0.00203728])
                        pb_scale_1 = _apply_casa_airy_pb(lmn_rot_1,freq_chan[i_chan],pb_parms)
                        pb_scale_2 = _apply_casa_airy_pb(lmn_rot_2,freq_chan[i_chan],pb_parms)
                    elif np.logical_and(pb_parms['pb_func'] == 'airy', n_ant_bool):
                        pb_scale_1 = _apply_airy_pb(lmn_rot_1,freq_chan[i_chan],pb_parms)
                        pb_scale_2 = _apply_airy_pb(lmn_rot_2,freq_chan[i_chan],pb_parms)
                    elif np.logical_and(pb_parms['pb_func'] == 'casa_airy', not(n_ant_bool)):
                        pb_scale_1 = _apply_casa_airy_pb(lmn_rot,freq_chan[i_chan],pb_parms)
                        pb_scale_2 = 1
                    elif np.logical_and(pb_parms['pb_func'] == 'airy', not(n_ant_bool)):
                        pb_scale_1 = _apply_airy_pb(lmn_rot,freq_chan[i_chan],pb_parms)
                        pb_scale_2 = 1
                    else:
                        pb_scale_1 = 1
                        pb_scale_2 = 1
           
                    phase_scaled = phase*freq_chan[i_chan]/c
                    for i_pol in range(n_pol):
                        flux = point_source_flux[i_time//f_sf_time, i_chan//f_sf_chan, i_pol//f_sf_pol, i_point_source]
                        
                        vis_data[i_time,i_baseline,i_chan,i_pol] = vis_data[i_time,i_baseline,i_chan,i_pol] + pb_scale_1*pb_scale_2*flux*np.exp(phase_scaled)
                        #print(pb_scale*flux,np.abs(np.exp(phase_scaled)))

    return vis_data


