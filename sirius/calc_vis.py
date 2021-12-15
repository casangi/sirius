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
import time
from numba import jit
import numba
from ._sirius_utils._direction_rotate import _calc_rotation_mats, _directional_cosine,  _sin_project
from ._sirius_utils._beam_utils import _calc_pb_scale, _beam_models_to_tuple, _pol_code_to_index
from sirius_data._constants import map_mueler_to_pol, c

def calc_vis(uvw,vis_data_shape,point_source_flux,point_source_ra_dec,pointing_ra_dec,phase_center_ra_dec,antenna1,antenna2,freq_chan,beam_model_map,beam_models, parallactic_angle, pol, mueller_selection):
    return 0

def calc_vis_chunk(uvw,vis_data_shape,point_source_flux,point_source_ra_dec,pointing_ra_dec,phase_center_ra_dec,antenna1,antenna2,freq_chan,beam_model_map,beam_models, parallactic_angle, pol, mueller_selection):
    """
    Simulate a interferometric visibilities.
    
    Parameters
    ----------
    point_source_flux: np.array [n_point_sources,n_time, n_chan, n_pol] (singleton: n_time, n_chan)
    point_source_ra_dec: np.array [n_time, n_point_sources, 2]          (singleton: n_time)
    pointing_ra_dec: np.array [n_time, n_ant, 2]                   (singleton: n_time, n_ant)
    phase_center_ra_dec: np.array [n_time, 2]                        (singleton: n_time)
    Returns
    -------
    vis : np.array
    """
    
    
    '''
    Singleton: Can have dimension = 1
    Warning: If mosaic, do not use n_time singleton with n_ant
    '''
    
#    rotation_parms = {}
#    rotation_parms['reproject'] = True
#    rotation_parms['common_tangent_reprojection'] = False
    
    #Function to appease Numba the terrible.
    beam_models_type0, beam_models_type1, beam_types, new_beam_model_map = _beam_models_to_tuple(beam_models,beam_model_map) #Needs to be done for numba
    pol = _pol_code_to_index(pol)
    

    n_time, n_baseline, n_chan, n_pol = vis_data_shape
    n_ant = len(beam_model_map)
    
    #Check all dims are either 1 or n
    f_pc_time = n_time if phase_center_ra_dec.shape[0] == 1 else 1
    f_ps_time = n_time if point_source_ra_dec.shape[0] == 1 else 1
    f_sf_time = n_time if point_source_flux.shape[1] == 1 else 1
    f_sf_chan = n_chan if point_source_flux.shape[2] == 1 else 1
    
    do_pointing = False
    if pointing_ra_dec is not None:
        do_pointing = True
        f_pt_time = n_time if phase_center_ra_dec.shape[0] == 1 else 1
        f_pt_ant =  n_ant if point_source_ra_dec.shape[1] == 1 else 1
    else:
        pointing_ra_dec = np.zeros((2,2,2))
        f_pt_time = n_time
        f_pt_ant = n_ant
        
    #print(beam_models_type0, beam_models_type1)
        
    #print("do_pointing ", do_pointing )
    vis_data = np.zeros(vis_data_shape,dtype=np.complex128)
    start = time.time()
    calc_vis_jit(vis_data, uvw,tuple(vis_data_shape),point_source_flux.astype(np.complex128),point_source_ra_dec,pointing_ra_dec,phase_center_ra_dec,antenna1,antenna2,freq_chan,beam_models_type0, beam_models_type1, beam_types, new_beam_model_map, parallactic_angle, pol, mueller_selection, f_pc_time, f_ps_time, f_sf_time, f_sf_chan, f_pt_time, f_pt_ant, do_pointing)
    
    return vis_data
    
    
#@jit(nopython=True,cache=True,nogil=True)
@jit(nopython=True,nogil=True)
def calc_vis_jit(vis_data,uvw,vis_data_shape,point_source_flux,point_source_ra_dec,pointing_ra_dec,phase_center_ra_dec,antenna1,antenna2,freq_chan,beam_models_type0, beam_models_type1, beam_types, beam_model_map, parallactic_angle, pol, mueller_selection, f_pc_time, f_ps_time, f_sf_time, f_sf_chan, f_pt_time, f_pt_ant, do_pointing):

    n_time, n_baseline, n_chan, n_pol = vis_data_shape
    n_ant = len(beam_model_map)
    #vis_data = np.zeros(vis_data_shape,dtype=np.complex)
    
    n_point_source = point_source_ra_dec.shape[1]
    
    #prev_ra_dec_in =  np.zeros((4,))
    #prev_ra_dec_out = np.zeros((4,))
    prev_ra_dec_in =  np.zeros((4,),dtype=numba.float64)
    prev_ra_dec_out = np.zeros((4,),dtype=numba.float64)
    
    #print(f_pt_time, f_pt_ant, do_pointing)
    
    #print('in calc vis')
    #print('pointing_ra_dec',pointing_ra_dec.dtype,point_source_ra_dec.dtype)

    for i_time in range(n_time):
        #print("Completed time step ", i_time,"of",n_time)
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
                    #uvw_rotmat = np.ones((3,3))
                    #lmn_rot = np.array([0.91651,0.4,0])
                    
                    uvw_rotmat, lmn_rot = _calc_rotation_mats(ra_dec_in, ra_dec_out)
                    lm_sin = _sin_project(ra_dec_in,ra_dec_out)
                    #lm_sin = lmn_rot #use 37 in apply_primary_beam r = 2.0*np.arcsin(np.sqrt(np.sum(lmn**2))/2.0)
                    sep = 2.0*np.arcsin(np.sqrt(np.sum(lm_sin**2))/2.0)
                    #print('lm_sin',lm_sin,lmn_rot,np.sqrt(lm_sin[0]**2 + lm_sin[1]**2),sep)
                    
                if do_pointing:
                    if not(np.array_equal(prev_ra_dec_in, ra_dec_in_1) and np.array_equal(prev_ra_dec_out, ra_dec_out)):
                        lm_sin_1 = _sin_project(ra_dec_in_1, ra_dec_out) # NBNBNB might need to change to _calc_rotation_mats depending
                        sep_1 = 2.0*np.arcsin(np.sqrt(np.sum(lm_sin_1**2))/2.0)
                    if not(np.array_equal(prev_ra_dec_in, ra_dec_in_2) and np.array_equal(prev_ra_dec_out, ra_dec_out)):
                        lm_sin_2 = _sin_project(ra_dec_in_2, ra_dec_out)
                        sep_2 = 2.0*np.arcsin(np.sqrt(np.sum(lm_sin_2**2))/2.0)
                
                #rad_to_arcmin = (60*180)/np.pi
                #print('ra_dec_in_1', ra_dec_in, ' ra_dec_out', ra_dec_out)
                #print(' lmn_rot ', lmn_rot, ' lm_sin ', lm_sin)
                #print(' lmn_rot ', np.sqrt(np.sum(lmn_rot**2))*rad_to_arcmin, ' lm_sin ', np.sqrt(np.sum(lm_sin**2))*rad_to_arcmin)
                #print(' lmn_rot ', 2.0*np.arcsin(np.sqrt(np.sum(lmn_rot**2))/2.0)*rad_to_arcmin, ' lm_sin ', 2.0*np.arcsin(np.sqrt(np.sum(lm_sin**2))/2.0)*rad_to_arcmin)
                #print('*'*30)
                #phase = 2*1j*np.pi*lmn_rot@(uvw[i_time,i_baseline,:]@uvw_rotmat)
                phase = 2*np.pi*lmn_rot@(uvw[i_time,i_baseline,:]@uvw_rotmat)
                
                prev_ra_dec_in = ra_dec_in
                prev_ra_dec_out = ra_dec_out
                #print("s0",time.time()-s0)
                
                #print('lmn_rot',lmn_rot)
                #print('uvw_rot',uvw[i_time,i_baseline,:]@uvw_rotmat)
                ##################### Apply primary beam to flux #####################
                for i_chan in range(n_chan):
                
                    
                    #s1 = time.time()
                    flux = point_source_flux[i_point_source,i_time//f_sf_time, i_chan//f_sf_chan, :]
                    
                    #print("s1",time.time()-s1)
                    
                    ################### Apply primary beams #####################
                    bm1_indx = beam_model_map[i_ant_1]
                    bm2_indx = beam_model_map[i_ant_2]
                    bm1_type = beam_types[i_ant_1]
                    bm2_type = beam_types[i_ant_2]

            

                    #print('flux_scaled',flux_scaled)
                    #s2 = time.time()
                    

                    if do_pointing:
                        flux_scaled, outside_beam = _calc_pb_scale(flux,sep_1,sep_2,bm1_indx,bm2_indx,bm1_type,bm2_type,lm_sin_1,lm_sin_2,beam_models_type0,beam_models_type1,pa,freq_chan[i_chan],mueller_selection,do_pointing)
                    else:
                        flux_scaled, outside_beam = _calc_pb_scale(flux,sep,sep,bm1_indx,bm2_indx,bm1_type,bm2_type,lm_sin,lm_sin,beam_models_type0,beam_models_type1,pa,freq_chan[i_chan],mueller_selection,do_pointing)

                    #flux_scaled = np.array([1,0,0,1])
                    #outside_beam = False
                    
                    #print('flux_scaled', flux_scaled, outside_beam, beam_models_type1[bm1_indx], beam_models_type1[bm2_indx])
                    #print("s2",time.time()-s2)
                    if not outside_beam:
                        #s3 = time.time()
                        phase_scaled = 1j*phase*freq_chan[i_chan]/c
                        #print(flux_scaled[pol])
                        #vis_data[i_time,i_baseline,i_chan,:] = vis_data[i_time,i_baseline,i_chan,:] + flux_scaled[pol]*np.exp(phase_scaled)/(1-lmn_rot[2])
                        #print("s3",time.time()-s3)
                        
                        #print(phase_scaled,type(phase),type(phase_scaled))
                        
                        #s4 = time.time()
                        for i_pol in range(n_pol):
                            #vis_data[i_time,i_baseline,i_chan,i_pol] = vis_data[i_time,i_baseline,i_chan,i_pol] + flux_scaled[pol[i_pol]]*np.exp(phase_scaled)/(1-lmn_rot[2])
                            
                            vis_data[i_time,i_baseline,i_chan,i_pol] = vis_data[i_time,i_baseline,i_chan,i_pol] + flux_scaled[pol[i_pol]]*np.exp(phase_scaled)/(1-lmn_rot[2])
                            #print(i_time,i_baseline,i_chan,i_pol,vis_data[i_time,i_baseline,i_chan,i_pol],flux_scaled[i_pol])
                        #print("s4",time.time()-s4)
                            #print(pb_scale*flux,np.abs(np.exp(phase_scaled)))
                    #exstract_arrays_from_bm_xds(x=7)
            #return vis_data
    
#sample_J_analytic(lmn, freq, dish_diameter, blockage_diameter, ipower, pb_func)
#sample_J(bm_J, bm_pa, bm_chan, lmn, freq, pa, delta_l, delta_m)
    

    

    

