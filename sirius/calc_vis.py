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
from ._sirius_utils._direction_rotate import _calc_rotation_mats, _cs_calc_rotation_mats, _directional_cosine,  _sin_project
from ._sirius_utils._apply_primary_beam import _apply_casa_airy_pb, _apply_airy_pb, apply_casa_airy_pb, apply_airy_pb
from ._sirius_utils._math_utils import _interp_array, _rot_coord
from ._sirius_utils._array_utils import _find_angle_indx, _find_val_indx
import matplotlib.pyplot as plt
import time
from numba import jit
import numba
from sirius_data._constants import map_mueler_to_pol

def calc_vis(uvw,vis_data_shape,point_source_flux,point_source_ra_dec,pointing_ra_dec,phase_center_ra_dec,antenna1,antenna2,freq_chan,beam_model_map,beam_models, parallactic_angle, pol, mueller_selection):
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
    beam_models_type0, beam_models_type1, beam_types, new_beam_model_map = beam_models_to_tuple(beam_models,beam_model_map) #Needs to be done for numba
    pol = pol_code_to_index(pol)
    

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
                        flux_scaled, outside_beam = calc_pb_scale(flux,sep_1,sep_2,bm1_indx,bm2_indx,bm1_type,bm2_type,lm_sin_1,lm_sin_2,beam_models_type0,beam_models_type1,pa,freq_chan[i_chan],mueller_selection,do_pointing)
                    else:
                        flux_scaled, outside_beam = calc_pb_scale(flux,sep,sep,bm1_indx,bm2_indx,bm1_type,bm2_type,lm_sin,lm_sin,beam_models_type0,beam_models_type1,pa,freq_chan[i_chan],mueller_selection,do_pointing)
                        
                    
                    #print('flux_scaled', flux_scaled, outside_beam, beam_models_type1[bm1_indx], beam_models_type1[bm2_indx])
                    #print("s2",time.time()-s2)
                    if not outside_beam:
                        #s3 = time.time()
                        phase_scaled = 1j*phase*freq_chan[i_chan]/c
                        #print(flux_scaled[pol])
                        #vis_data[i_time,i_baseline,i_chan,:] = vis_data[i_time,i_baseline,i_chan,:] + flux_scaled[pol]*np.exp(phase_scaled)/(1-lmn_rot[2])
                        #print("s3",time.time()-s3)
                        
                        #print(phase_scaled,type(phase),type(phase_scaled))
                        
                        for i_pol in range(n_pol):
                            #vis_data[i_time,i_baseline,i_chan,i_pol] = vis_data[i_time,i_baseline,i_chan,i_pol] + flux_scaled[pol[i_pol]]*np.exp(phase_scaled)/(1-lmn_rot[2])
                            
                            vis_data[i_time,i_baseline,i_chan,i_pol] = vis_data[i_time,i_baseline,i_chan,i_pol] + flux_scaled[pol[i_pol]]*np.exp(phase_scaled)/(1-lmn_rot[2])
                            #print(i_time,i_baseline,i_chan,i_pol,vis_data[i_time,i_baseline,i_chan,i_pol],flux_scaled[i_pol])
                            
                            #print(pb_scale*flux,np.abs(np.exp(phase_scaled)))
                    #exstract_arrays_from_bm_xds(x=7)
            #return vis_data
    
#sample_J_analytic(lmn, freq, dish_diameter, blockage_diameter, ipower, pb_func)
#sample_J(bm_J, bm_pa, bm_chan, lmn, freq, pa, delta_l, delta_m)
    

def beam_models_to_tuple(beam_models,beam_model_map):
    new_beam_model_map = np.zeros(beam_model_map.shape,dtype=np.int)
    beam_types = np.zeros(beam_model_map.shape,dtype=np.int)
    beam_models_list_type0 = []
    beam_models_list_type1 = []
    
    i_t0 = 0
    i_t1 = 0
    for i,bm in enumerate(beam_models):
        if "J" in bm:
            beam_models_list_type0.append(exstract_arrays_from_bm_xds(bm))
            beam_types[beam_model_map==i] = 0
            new_beam_model_map[beam_model_map==i] = i_t0
            i_t0 = i_t0 + 1
        else:
            beam_models_list_type1.append(exstract_vals_from_analytic_dict(bm))
            beam_types[beam_model_map==i] = 1
            new_beam_model_map[beam_model_map==i] = i_t1
            i_t1 = i_t1 + 1
            
    #Dummy for sample_j
    if not bool(beam_models_list_type0):
        d_arr_float = np.array([0.])
        d_arr_int = np.array([0])
        d_J = np.zeros((1,1,1,1,1),np.complex128)
        beam_models_list_type0.append((0,d_J,d_arr_float,d_arr_float,
                                        d_arr_int,d_arr_float,d_arr_float,0.0))
    
    #Dummy for sample_j_analytic
    if not bool(beam_models_list_type1):
        beam_models_list_type1.append((1,"none",0.,0.,0.0))
    
    return tuple(beam_models_list_type0), tuple(beam_models_list_type1), tuple(beam_types), tuple(new_beam_model_map)


def exstract_arrays_from_bm_xds(bm):
    bm_J = bm.J.isel(model=0).values #make sure that there is only one model
    print("bm_J", bm_J.shape)
    pa = bm.pa.values
    chan = bm.chan.values
    pol = bm.pol.values
    delta_l = (bm.l[1].values - bm.l[0].values).astype(float)
    delta_m = (bm.m[1].values - bm.m[0].values).astype(float)
    print(bm)
    max_rad_1GHz = bm.attrs['max_rad_1GHz']
    return (0,bm_J,pa,chan,pol,delta_l,delta_m,max_rad_1GHz)
    
def exstract_vals_from_analytic_dict(bm):
    pb_func = bm['pb_func']
    #dish_diameter = bm['dish_diameter']
    #blockage_diameter = bm['blockage_diameter']
    dish_diameter = bm['dish_diam']
    blockage_diameter = bm['blockage_diam']
    max_rad_1GHz = bm['max_rad_1GHz']
    return (1,pb_func,dish_diameter,blockage_diameter,max_rad_1GHz)


#@jit(nopython=True,cache=True,nogil=True)
@jit(nopython=True,nogil=True)
def calc_pb_scale(flux, sep1, sep2, bm1_indx,bm2_indx,bm1_type,bm2_type,lmn1,lmn2,beam_models_type0,beam_models_type1,pa,freq,mueller_selection,do_pointing):

    #print(sep1,sep2,bm1_indx,bm2_indx,bm1_type,bm2_type)
    outside_beam = False
    if (bm1_indx == bm2_indx) and (bm1_type == bm2_type) and ~do_pointing: #Antennas are the same
        if bm1_type == 0: #Checks if it is a zernike model
            bm1 = beam_models_type0[bm1_indx]
            max_rad = bm1[7]*freq/10**9 # scale max_rad_1GHz to freq
            
            if sep1 < max_rad:
                #bm1 = beam_models_type0[bm1_indx]
                J_sampled = sample_J(bm1[1],bm1[2],bm1[3],bm1[4],bm1[5],bm1[6],pa,freq,lmn1)[:,0]
                
                #J_sampled = np.zeros((4,),dtype=numba.complex128)
                M = make_mueler_mat(J_sampled, J_sampled, mueller_selection)
                #Add check if J sampled is < 0 and then skip this
                flux_scaled = np.dot(M,flux)
            else:
                outside_beam = True
        else:
            #bm1 = beam_models_type1[bm1_indx]
            bm1 = beam_models_type1[bm1_indx]
            max_rad = bm1[4]*freq/10**9 # scale max_rad_1GHz to freq
            if sep1 < max_rad:
                J_sampled = sample_J_analytic(bm1[1],bm1[2],bm1[3],lmn1,freq,1)
                #print('J_sampled',J_sampled,bm1[1],bm1[2],bm1[3],lmn1,freq,1)
                flux_scaled = flux*J_sampled**2
            else:
                outside_beam = True
    else:
        if bm1_type == 0:
            bm1 = beam_models_type0[bm1_indx]
            max_rad = bm1[7]*freq/10**9 # scale max_rad_1GHz to freq
            if sep1 < max_rad:
                J_sampled1 = sample_J(bm1[1],bm1[2],bm1[3],bm1[4],bm1[5],bm1[6],pa,freq,lmn1)[:,0]
            else:
                outside_beam = True
        else:
            bm1 = beam_models_type1[bm1_indx]
            max_rad = bm1[4]*freq/10**9 # scale max_rad_1GHz to freq
            if sep1 < max_rad:
                J_sampled1 = sample_J_analytic(bm1[1],bm1[2],bm1[3],lmn1,freq,1)
            else:
                outside_beam = True
        if bm2_type == 0:
            bm2 = beam_models_type0[bm2_indx]
            max_rad = bm2[7]*freq/10**9 # scale max_rad_1GHz to freq
            if sep2 < max_rad:
                J_sampled2 = sample_J(bm2[1],bm2[2],bm2[3],bm2[4],bm2[5],bm2[6],pa,freq,lmn2)[:,0]
            else:
                outside_beam =  True
        else:
            bm2 = beam_models_type1[bm2_indx]
            max_rad = bm2[4]*freq/10**9 # scale max_rad_1GHz to freq
            if sep2 < max_rad:
                J_sampled2 = sample_J_analytic(bm2[1],bm2[2],bm2[3],lmn2,freq,1)
            else:
                outside_beam = True
                
        if not outside_beam:
            M = make_mueler_mat(J_sampled1, J_sampled2, mueller_selection)
            flux_scaled = np.dot(M,flux)
    if outside_beam:
        flux_scaled = np.zeros(4, dtype = numba.complex128)

    return flux_scaled, outside_beam

def pol_code_to_index(pol):
    if pol[0] in [5,6,7,8]:
        return pol-5
    if pol[0] in [9,10,11,12]:
        return pol-9
    assert False, "Unsupported pol " + str(pol)
    
#@jit(nopython=True,cache=True,nogil=True)
@jit(nopython=True,nogil=True)
def sample_J_analytic(pb_func, dish_diameter,blockage_diameter, lmn, freq, ipower):
    #pb_parms = bm
    #pb_parms['ipower'] = 1
    
    if pb_func == 'casa_airy':
        J_sampled = _apply_casa_airy_pb(lmn,freq,dish_diameter, blockage_diameter, ipower)
    elif pb_func == 'airy':
        J_sampled = _apply_airy_pb(lmn,freq,dish_diameter, blockage_diameter, ipower)
    else:
        J_sampled = 1
    
    J_sampled_array = np.zeros(4, dtype = numba.float64)
    J_sampled_array[0] = J_sampled
    J_sampled_array[3] = J_sampled
    #J_sampled = np.array([J_sampled,0,0,J_sampled])
    return J_sampled_array.astype(numba.complex128)
    
@jit(nopython=True,cache=True,nogil=True)
def sample_J(bm_J,bm_pa,bm_chan, bm_pol, bm_delta_l,bm_delta_m,pa,freq,lmn):
    pa_indx = _find_angle_indx(bm_pa,pa)
    chan_indx = _find_val_indx(bm_chan, freq)
    bm_J_sub = bm_J[pa_indx, chan_indx]

    x_rot, y_rot  = _rot_coord(lmn[0],lmn[1],pa-bm_pa[pa_indx])
    xrot = np.ones(1, dtype = numba.float64)
    xrot[0] = x_rot
    yrot = np.ones(1, dtype = numba.float64)
    yrot[0] = y_rot
    #print((xrot/bm[4]) + len(bm_J_sub[0, :, 0])//2)
    #print((yrot/bm[5]) + len(bm_J_sub[0, 0, :])//2)
    
    J_temp = _interp_array(bm_J_sub, xrot, yrot, bm_delta_l, bm_delta_m)
    
    J = np.zeros((4,J_temp.shape[1]),dtype=numba.complex128) + 1
    for i,p in enumerate(bm_pol):
        J[p] = J_temp[i,:]
        
    return J

#Non-numba versions:
def sample_J_analytic_og(bm,lmn,freq):
    pb_parms = bm
    pb_parms['ipower'] = 1
    
    if pb_parms['pb_func'] == 'casa_airy':
        J_sampled = apply_casa_airy_pb(lmn,freq,pb_parms)
    elif pb_parms['pb_func'] == 'airy':
        J_sampled = apply_airy_pb(lmn,freq,pb_parms)
    else:
        J_sampled = 1
    J_sampled = np.array([J_sampled,0,0,J_sampled])
    return J_sampled


def sample_J_og(bm,lmn,freq,pa):
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
def make_mueler_mat(J1, J2, mueller_selection):

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
    
    
    

    

