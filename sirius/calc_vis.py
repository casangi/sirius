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
from ._sirius_utils._apply_primary_beam import _apply_casa_airy_pb, _apply_airy_pb
from ._sirius_utils._ant_jones_term import _rot_coord
from ._sirius_utils._math import _find_angle_indx
import matplotlib.pyplot as plt
import time

def calc_vis(uvw,vis_data_shape,point_source_flux,point_source_ra_dec,pointing_ra_dec,phase_center_ra_dec,antenna1,antenna2,freq_chan,beam_model_map,eval_beam_models, parallactic_angle, pol, mueller_selection, pb_limit):
    '''
    point_source_flux: [n_time, n_chan, n_pol, n_point_sources] (singleton: n_time, n_chan, n_pol)
    point_source_ra_dec:  [n_time, n_point_sources, 2]          (singleton: n_time)
    pointing_ra_dec:  [n_time, n_ant, 2]                   (singleton: n_time, n_ant)
    phase_center_ra_dec: [n_time, 2]                        (singleton: n_time)
    Singleton: Can have dimension = 1
    Warning: If mosaic, do not use n_time singleton with n_ant
    '''
    
    n_time, n_baseline, n_chan, n_pol = vis_data_shape
    n_ant = len(beam_model_map)
    vis_data = np.zeros(vis_data_shape,dtype=np.complex)
    n_point_source = point_source_ra_dec.shape[1]
    
    rotation_parms = {}
    rotation_parms['reproject'] = True
    rotation_parms['common_tangent_reprojection'] = False

    
    
    #Check all dims are either 1 or n
    f_pc_time = n_time if phase_center_ra_dec.shape[0] == 1 else 1
    f_ps_time = n_time if point_source_ra_dec.shape[0] == 1 else 1
    f_sf_time = n_time if point_source_flux.shape[0] == 1 else 1
    f_sf_chan = n_chan if point_source_flux.shape[1] == 1 else 1
    #f_sf_pol = n_pol if point_source_flux.shape[2] == 1 else 1
    
    prev_ra_dec_in = np.array([0.0,0.0])
    prev_ra_dec_out = np.array([0.0,0.0])
    
    #if autocorr:
    #    n_baseline = (n_ant**2 + n_ant)/2
    #else:
    #    n_baseline = (n_ant**2 - n_ant)/2
    
    #antenna_baselines = np.concatenate((np.arange(0, n_baseline, 1).reshape((1, n_baseline)), ANTENNA1.reshape((1, n_baseline)), ANTENNA2.reshape((1, n_baseline))), axis = 0)
    pol = pol_code_to_index(pol)
    
    pb_limit = np.sqrt(pb_limit)
    
    do_pointing = False
    if pointing_ra_dec is not None:
        do_pointing = True
        if pointing_ra_dec.shape[0] == 1: f_pt_time =  n_time
        if pointing_ra_dec.shape[1] == 1: f_pt_time =  n_ant
    
    for i_time in range(n_time):
        print("Completed time step ", i_time,"of",n_time)
        pa = parallactic_angle[i_time]
        ra_dec_in = phase_center_ra_dec[i_time//f_pc_time, :]
        #print('phase_center_ra_dec',phase_center_ra_dec)
        
        for i_baseline in range(n_baseline):
            #print(i_baseline,n_baseline)
            i_ant_1 = antenna1[i_baseline]
            i_ant_2 = antenna2[i_baseline]
            if do_pointing:
                ra_dec_in_1 = pointing_ra_dec[i_time//f_pt_time,i_ant_1//f_pt_ant,:]
                ra_dec_in_2 = pointing_ra_dec[i_time//f_pt_time,i_ant_2//f_pt_ant,:]
            
            for i_point_source in range(n_point_source):
                #s0 = time.time()
                ra_dec_out = point_source_ra_dec[i_time//f_ps_time,i_point_source,:]
                #print('ra_dec_out',ra_dec_out)
                if not(np.array_equal(prev_ra_dec_in, ra_dec_in) and np.array_equal(prev_ra_dec_out, ra_dec_out)):
                    uvw_rotmat, lmn_rot = _calc_rotation_mats(ra_dec_in, ra_dec_out, rotation_parms)
                    
                if do_pointing:
                    if not(np.array_equal(prev_ra_dec_in, ra_dec_in_1) and np.array_equal(prev_ra_dec_out, ra_dec_out)):
                        lmn_rot_1 = _directional_cosine(ra_dec_in_1, ra_dec_out, rotation_parms)
                    if not(np.array_equal(prev_ra_dec_in, ra_dec_in_2) and np.array_equal(prev_ra_dec_out, ra_dec_out)):
                        lmn_rot_2 = _directional_cosine(ra_dec_in_2, ra_dec_out, rotation_parms)
                 
                phase = 2*1j*np.pi*lmn_rot@(uvw[i_time,i_baseline,:]@uvw_rotmat)
                
                prev_ra_dec_in = ra_dec_in
                prev_ra_dec_out = ra_dec_out
                #print("s0",time.time()-s0)
                
                #print('lmn_rot',lmn_rot)
                
                for i_chan in range(n_chan):
                    #s1 = time.time()
                    flux = point_source_flux[i_time//f_sf_time, i_chan//f_sf_chan, :, i_point_source]
                    bm1_indx = beam_model_map[i_ant_1]
                    bm2_indx = beam_model_map[i_ant_2]
                    #print("s1",time.time()-s1)
                    
                    #s2 = time.time()
                    if do_pointing:
                        flux_scaled = calc_pb_scale(flux,eval_beam_models[bm1_indx],eval_beam_models[bm2_indx],bm1_indx,bm2_indx,lmn_rot_1,lmn_rot_2,pa,freq_chan[i_chan],mueller_selection,pb_limit,do_pointing)
                    else:
                        flux_scaled = calc_pb_scale(flux,eval_beam_models[bm1_indx],eval_beam_models[bm2_indx],bm1_indx,bm2_indx,lmn_rot,lmn_rot,pa,freq_chan[i_chan],mueller_selection,pb_limit,do_pointing)
                    #print("s2",time.time()-s2)
                        
                    #s3 = time.time()
                    phase_scaled = phase*freq_chan[i_chan]/c
                    #print(flux_scaled[pol])
                    vis_data[i_time,i_baseline,i_chan,:] = vis_data[i_time,i_baseline,i_chan,:] + flux_scaled[pol]*np.exp(phase_scaled)/(1-lmn_rot[2])
                    #print("s3",time.time()-s3)
                    #for i_pol in range(n_pol):
                    #    vis_data[i_time,i_baseline,i_chan,i_pol] = vis_data[i_time,i_baseline,i_chan,i_pol] + pb_scale_1*pb_scale_2*flux*np.exp(phase_scaled)/(1-lmn_rot[2])
                        #print(pb_scale*flux,np.abs(np.exp(phase_scaled)))
                
    return vis_data
    
def calc_pb_scale(flux,bm1,bm2,bm1_indx,bm2_indx,lmn1,lmn2,pa,freq,mueller_selection,pb_limit,do_pointing):
    #print(mueller_selection)
    
    if (bm1_indx == bm2_indx) and ~do_pointing:
        #J = sample_ant_Jones(flux,bm1,bm1_indx,pa)
        
        if "J" in bm1:
            J_sampled = sample_J(bm1,lmn1,freq,pa)
            M = make_mueler_mat(J_sampled, J_sampled, np.array([0,1,2,3]), mueller_selection, inv=False)
        else: #analytic function
            J_sampled = sample_J_analytic(bm1,lmn1,freq)
            #print(J_sampled,J_sampled)
            if (J_sampled[0] > pb_limit):
                flux_scaled = flux*J_sampled**2
            else:
                flux_scaled = flux-flux
            
            return flux_scaled
            #M = make_mueler_mat(J_sampled, J_sampled, np.array([0,1,2,3]), mueller_selection, inv=False)
    else:
        if "J" in bm1:
            J_sampled1 = sample_J(bm1,lmn1,freq,pa)
        else:
            J_sampled1 = sample_J_analytic(bm1,lmn1,freq)

        if "J" in bm2:
            J_sampled2 = sample_J(bm2,lmn2,freq,pa)
        else:
            J_sampled2 = sample_J_analytic(bm2,lmn2,freq)

        #Add a check that bm1.pol.values is the same bm2.pol.values
        M = make_mueler_mat(J_sampled1, J_sampled2, np.array([0,1,2,3]), mueller_selection, inv=False)
        
    #Add check if J sampled is < 0 and then skip this
    if (M[0,0] > pb_limit) and (M[3,3] > pb_limit):
        flux_scaled = np.dot(M,flux)
    else:
        flux_scaled = np.array([0,0,0,0])
        
    return flux_scaled
    
def pol_code_to_index(pol):
    if pol[0] in [5,6,7,8]:
        return pol-5
    if pol[0] in [9,10,11,12]:
        return pol-9
    assert False, "Unsupported pol " + str(pol)
    
def sample_J_analytic(bm,lmn,freq):
    pb_parms = bm
    pb_parms['ipower'] = 1
    
    if pb_parms['pb_func'] == 'casa_airy':
        J_sampled = _apply_casa_airy_pb(lmn,freq,pb_parms)
    elif pb_parms['pb_func'] == 'airy':
        J_sampled = _apply_airy_pb(lmn,freq,pb_parms)
    else:
        J_sampled = 1
    J_sampled = np.array([J_sampled,0,0,J_sampled])
    return J_sampled

        
def sample_J(bm,lmn,freq,pa):
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
    

def make_mueler_mat(J1, J2, pol, mueller_selection, inv=False):
    
    #print('pol',pol)
    if inv:
        map_mueler_to_pol = np.array([[3, 3],[3, 2],[2, 3],[2, 2],[3, 1],[3, 0],[2, 1],[2, 0],[1, 3],[1, 2],[0, 3],[0, 2],[1, 1],[1, 0],[0, 1],[0, 0]]) # np.flip(map_mueler_to_pol,axis=0)
        #map_mueler_to_pol = np.array([ [[3, 3],[3, 2],[2, 3],[2, 2]],[[3, 1],[3, 0],[2, 1],[2, 0]],[[1, 3],[1, 2],[0, 3],[0, 2]],[[1, 1],[1, 0],[0, 1],[0, 0]]])
    else:
        map_mueler_to_pol = np.array([[0,0],[0,1],[1,0],[1,1],[0,2],[0,3],[1,2],[1,3],[2,0],[2,1],[3,0],[3,1],[2,2],[2,3],[3,2],[3,3]])
        #map_mueler_to_pol = np.array([[[0,0],[0,1],[1,0],[1,1]],[[0,2],[0,3],[1,2],[1,3]],[[2,0],[2,1],[3,0],[3,1]],[[2,2],[2,3],[3,2],[3,3]]])
        
    M = np.zeros((4,4),dtype=np.complex)
    
    for m_flat_indx in mueller_selection:
        #print(m_flat_indx//4,m_flat_indx - 4*(m_flat_indx//4))
        #print(np.where(map_mueler_to_pol[m_flat_indx][0] == pol)[0][0])
        #print(pol, map_mueler_to_pol[m_flat_indx][0])
        M[m_flat_indx//4,m_flat_indx - 4*(m_flat_indx//4)] = J1[np.where(map_mueler_to_pol[m_flat_indx][0] == pol)[0][0]]*np.conj(J2[np.where(map_mueler_to_pol[m_flat_indx][1] == pol)[0][0]])
            
    return M

    
    
    
#def calc_
#    if "J" in bm:
#
#    else:
        
       
    
    
    

    


    '''
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
        pb_parms['ipower'] = 2
        pb_scale_1 = _apply_casa_airy_pb(lmn_rot,freq_chan[i_chan],pb_parms)
        pb_scale_2 = 1
    elif np.logical_and(pb_parms['pb_func'] == 'airy', not(n_ant_bool)):
        pb_parms['ipower'] = 2
        pb_scale_1 = _apply_airy_pb(lmn_rot,freq_chan[i_chan],pb_parms)
        pb_scale_2 = 1
    else:
        pb_scale_1 = 1
        pb_scale_2 = 1
    '''
    
    
    #    pb_parms['ipower'] = 1
    
    '''
    antenna_baselines = np.concatenate((np.arange(0, n_baseline, 1).reshape((1, n_baseline)), ANTENNA1.reshape((1, n_baseline)), ANTENNA2.reshape((1, n_baseline))), axis = 0)
    
    for i_time in range(n_time):
        
        ra_dec_in = phase_center_ra_dec[i_time//f_pt_time, :]
        
        for i_baseline in range(n_baseline):
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
                
                #print(lmn_rot)
                
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
                        pb_parms['ipower'] = 2
                        pb_scale_1 = _apply_casa_airy_pb(lmn_rot,freq_chan[i_chan],pb_parms)
                        pb_scale_2 = 1
                    elif np.logical_and(pb_parms['pb_func'] == 'airy', not(n_ant_bool)):
                        pb_parms['ipower'] = 2
                        pb_scale_1 = _apply_airy_pb(lmn_rot,freq_chan[i_chan],pb_parms)
                        pb_scale_2 = 1
                    else:
                        pb_scale_1 = 1
                        pb_scale_2 = 1
                        
                    pb_scale = pb_scale_1*pb_scale_2
                    if(pb_scale <= pb_parms['pb_limit']):
                        pb_scale = 0

                    phase_scaled = phase*freq_chan[i_chan]/c
                    for i_pol in range(n_pol):
                        flux = point_source_flux[i_time//f_sf_time, i_chan//f_sf_chan, i_pol//f_sf_pol, i_point_source]
                        
                        vis_data[i_time,i_baseline,i_chan,i_pol] = vis_data[i_time,i_baseline,i_chan,i_pol] + pb_scale_1*pb_scale_2*flux*np.exp(phase_scaled)/(1-lmn_rot[2])
                        #print(pb_scale*flux,np.abs(np.exp(phase_scaled)))

    return vis_data
    '''

