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
from ._sirius_utils._coord_transforms import _calc_rotation_mats, _directional_cosine,  _sin_project
from ._sirius_utils._beam_utils import _calc_pb_scale, _beam_models_to_tuple, _pol_code_to_index
from ._sirius_utils._array_utils import _is_subset
from sirius_data._constants import map_mueler_to_pol, c
from ._parm_utils._check_parms import _check_parms
#def calc_vis(uvw,vis_data_shape,point_source_flux,point_source_ra_dec,pointing_ra_dec,phase_center_ra_dec,antenna1,antenna2,freq_chan,beam_model_map,beam_models, parallactic_angle, pol, mueller_selection):
#    return 0

def calc_vis_chunk(uvw,vis_data_shape, point_source_flux,point_source_ra_dec,pointing_ra_dec,phase_center_ra_dec,antenna1,antenna2,chan_chunk,beam_model_map,beam_models, parallactic_angle, pol, mueller_selection, check_parms=True):
    """
    Simulate interferometric visibilities.
    
    Parameters
    ----------
    uvw : float np.array, [n_time,n_baseline,3]   
        Spatial frequency coordinates.
    vis_data_shape : float np.array, [4]
        Dimensions of visibility data [n_time, n_baseline, n_chan, n_pol].
    point_source_flux: float np.array, [n_point_sources,n_time, n_chan, n_pol], (singleton: n_time, n_chan), Janskys
        The flux of the point sources.
    point_source_ra_dec: float np.array, [n_time, n_point_sources, 2], (singleton: n_time), radians
        The position of the point sources.
    pointing_ra_dec: float np.array, [n_time, n_ant, 2], (singleton: n_time, n_ant), radians
        Pointings of antennas, if they are different from the phase center. Set to None if no pointing offsets are required.
    phase_center_ra_dec: float np.array, [n_time, 2], (singleton: n_time), radians
        Phase center of array.
    antenna1: np.array of int, [n_baseline]
        The indices of the first antenna in the baseline pairs. The _calc_baseline_indx_pair function in sirius._sirius_utils._array_utils can be used to calculate these values.
    antenna2: np.array of int, [n_baseline]
        The indices of the second antenna in the baseline pairs. The _calc_baseline_indx_pair function in sirius._sirius_utils._array_utils can be used to calculate these values.
    chan_chunk: float np.array, [n_chan], Hz
        Channel frequencies.
    beam_model_map: int np.array, [n_ant]
        Each element in beam_model_map is an index into beam_models.
    beam_models: list
        List of beam models to use. Beam models can be any combination of function parameter dictionaries or image xr.Datasets. Any Zernike polynomial coefficient xr.Datasets models must be converted to images using sirius.calc_beam.evaluate_beam_models.
    parallactic_angle: float np.array, [n_time], radians
        Parallactic angle over time at the array center.
    pol: int np.array 
        Must be a subset of ['RR','RL','LR','LL'] => [5,6,7,8] or ['XX','XY','YX','YY'] => [9,10,11,12].
    mueller_selection: int np.array
        The elements in the 4x4 beam Mueller matrix to use. The elements are numbered row wise.
        For example [ 0, 5, 10, 15] are the diagonal elements.
    check_parms: bool
        Check input parameters and asign defaults.
        
    Returns
    -------
    vis : complex np.array, [n_time,n_baseline,n_chan,n_pol]   
        Visibility data.
    """
    
    if check_parms:
        n_time, n_baseline, n_chan, n_pol = vis_data_shape
        n_ant = tel_xds.dims['ant_name']
        
        pol = np.array(pol)
        assert(_is_subset(pol_codes_RL,pol) or _is_subset(pol_codes_XY,pol)), 'Pol selection invalid, must either be subset of [5,6,7,8] or [9,10,11,12] but is '
        if not(_check_parms(beam_parms, 'mueller_selection', [list,np.array], list_acceptable_data_types=[np.int64], default = np.array([ 0, 5, 10, 15]),list_len=-1)): parms_passed = False
        
        #Check dimensions.
        assert(point_source_flux.shape[0] == point_source_ra_dec.shape[1]), 'n_point_sources dimension of point_source_flux[' + str(point_source_flux.shape[0]) +'] and point_source_ra_dec['+str(point_source_ra_dec.shape[1])+'] do not match.'
        assert(point_source_flux.shape[1] == 1) or (point_source_flux.shape[1] == n_time), 'n_time dimension in point_source_flux[' + str(point_source_flux.shape[1]) + '] must be either 1 or ' + str(n_time) + ' (see time_xda parameter).'
        assert(point_source_flux.shape[2] == 1) or (point_source_flux.shape[2] == n_chan), 'n_chan dimension in point_source_flux[' + str(point_source_flux.shape[2]) + '] must be either 1 or ' + str(n_chan) + ' (see chan_xda parameter).'
        assert(point_source_flux.shape[3] == 4), 'n_pol dimension in point_source_flux[' + str(point_source_flux.shape[3]) + '] must be 4.'
        
        assert(point_source_ra_dec.shape[0] == 1) or (point_source_ra_dec.shape[0] == n_time), 'n_time dimension in point_source_ra_dec[' + str(point_source_ra_dec.shape[0]) + '] must be either 1 or ' + str(n_time) + ' (see time_xda parameter).'
        assert(point_source_ra_dec.shape[2] == 2), 'ra,dec dimension in point_source_ra_dec[' + str(point_source_ra_dec.shape[2]) + '] must be 2.'
        
        if pointing_ra_dec is not None:
            assert(pointing_ra_dec.shape[0] == 1) or (pointing_ra_dec.shape[0] == n_time), 'n_time dimension in pointing_ra_dec[' + str(pointing_ra_dec.shape[0]) + '] must be either 1 or ' + str(n_time) + ' (see time_xda parameter).'
            assert(pointing_ra_dec.shape[1] == 1) or (pointing_ra_dec.shape[1] == n_ant), 'n_ant dimension in pointing_ra_dec[' + str(pointing_ra_dec.shape[1]) + '] must be either 1 or ' + str(n_ant) + ' (see tel_xds.dims[\'ant_name\']).'
            assert(pointing_ra_dec.shape[2] == 2), 'ra,dec dimension in pointing_ra_dec[' + str(pointing_ra_dec.shape[2]) + '] must be 2.'
            
            
        assert(phase_center_ra_dec.shape[0] == 1) or (phase_center_ra_dec.shape[0] == n_time), 'n_time dimension in phase_center_ra_dec[' + str(phase_center_ra_dec.shape[0]) + '] must be either 1 or ' + str(n_time) + ' (see time_xda parameter).'
        assert(phase_center_ra_dec.shape[1] == 2), 'ra,dec dimension in phase_center_ra_dec[' + str(phase_center_ra_dec.shape[1]) + '] must be 2.'
            
        assert(phase_center_names.shape[0] == 1) or (phase_center_names.shape[0] == n_time), 'n_time dimension in phase_center_ra_dec[' + str(phase_center_names.shape[0]) + '] must be either 1 or ' + str(n_time) + ' (see time_xda parameter).'
    
    # The _beam_models_to_tuple function is here to appease Numba the terrible. It unpacks the beam models from dictionaries and xr.Datasets to fixed tuples.
    beam_models_type0, beam_models_type1, beam_types, new_beam_model_map = _beam_models_to_tuple(beam_models,beam_model_map) 
    
    #print(beam_models_type0, beam_models_type1, beam_types, new_beam_model_map)
    
    do_pointing = False
    if pointing_ra_dec is not None:
        do_pointing = True
    else:
        pointing_ra_dec = np.zeros((2,2,2))
        
    vis_data = np.zeros(vis_data_shape,dtype=np.complex128)
    calc_vis_jit(vis_data, uvw,tuple(vis_data_shape),point_source_flux.astype(np.complex128),point_source_ra_dec,pointing_ra_dec,phase_center_ra_dec,antenna1,antenna2,chan_chunk,beam_models_type0, beam_models_type1, beam_types, new_beam_model_map, parallactic_angle, pol, mueller_selection, do_pointing)
    
    return vis_data
    
    
#@jit(nopython=True,cache=True,nogil=True)
@jit(nopython=True,nogil=True) #Jit compile function because it has large nested for loop (can't be easily vectorized).
def calc_vis_jit(vis_data,uvw,vis_data_shape,point_source_flux,point_source_ra_dec,pointing_ra_dec,phase_center_ra_dec,antenna1,antenna2,freq_chan,beam_models_type0, beam_models_type1, beam_types, beam_model_map, parallactic_angle, pol, mueller_selection, do_pointing):

    n_time, n_baseline, n_chan, n_pol = vis_data_shape
    n_ant = len(beam_model_map)

    n_point_source = point_source_ra_dec.shape[1]
    
    #Converts pol codes to indices.
    pol = _pol_code_to_index(pol)
    
    #Find all singleton dimensions, so that indexing is done correctly.
    n_time, n_baseline, n_chan, n_pol = vis_data_shape
    n_ant = len(beam_model_map)
    f_pc_time = n_time if phase_center_ra_dec.shape[0] == 1 else 1
    f_ps_time = n_time if point_source_ra_dec.shape[0] == 1 else 1
    f_sf_time = n_time if point_source_flux.shape[1] == 1 else 1
    f_sf_chan = n_chan if point_source_flux.shape[2] == 1 else 1
    
  
    if do_pointing:
        f_pt_time = n_time if phase_center_ra_dec.shape[0] == 1 else 1
        f_pt_ant =  n_ant if point_source_ra_dec.shape[1] == 1 else 1
    else:
        f_pt_time = n_time
        f_pt_ant = n_ant
    
    #prev_ra_dec_in =  np.zeros((4,))
    #prev_ra_dec_out = np.zeros((4,))
    prev_ra_dec_in =  np.zeros((4,),dtype=numba.float64)
    prev_ra_dec_out = np.zeros((4,),dtype=numba.float64)
    
    f_pc_time = n_time if phase_center_ra_dec.shape[0] == 1 else 1
    
    # Loop over dimensions n_time, n_baseline, n_point_source, n_chan, n_pol
    for i_time in range(n_time):
        #print("Completed time step ", i_time,"of",n_time)
        pa = parallactic_angle[i_time]
        ra_dec_in = phase_center_ra_dec[i_time//f_pc_time, :]
        
        for i_baseline in range(n_baseline):
            i_ant_1 = antenna1[i_baseline]
            i_ant_2 = antenna2[i_baseline]
            if do_pointing:
                ra_dec_in_1 = pointing_ra_dec[i_time//f_pt_time,i_ant_1//f_pt_ant,:]
                ra_dec_in_2 = pointing_ra_dec[i_time//f_pt_time,i_ant_2//f_pt_ant,:]
                 
            for i_point_source in range(n_point_source):
                ra_dec_out = point_source_ra_dec[i_time//f_ps_time,i_point_source,:]
                if not(np.array_equal(prev_ra_dec_in, ra_dec_in) and np.array_equal(prev_ra_dec_out, ra_dec_out)):
                    uvw_rotmat, lmn_rot = _calc_rotation_mats(ra_dec_in, ra_dec_out)
                    lm_sin = _sin_project(ra_dec_in,ra_dec_out)
                    sep = np.sqrt(np.sum(lm_sin**2))
                    
                if do_pointing:
                    if not(np.array_equal(prev_ra_dec_in, ra_dec_in_1) and np.array_equal(prev_ra_dec_out, ra_dec_out)):
                        lm_sin_1 = _sin_project(ra_dec_in_1, ra_dec_out) 
                        sep_1 = np.sqrt(np.sum(lm_sin_1**2))
                    if not(np.array_equal(prev_ra_dec_in, ra_dec_in_2) and np.array_equal(prev_ra_dec_out, ra_dec_out)):
                        lm_sin_2 = _sin_project(ra_dec_in_2, ra_dec_out)
                        sep_2 = np.sqrt(np.sum(lm_sin_2**2))
                

                phase = 2*np.pi*lmn_rot@(uvw[i_time,i_baseline,:]@uvw_rotmat)
                prev_ra_dec_in = ra_dec_in
                prev_ra_dec_out = ra_dec_out

                for i_chan in range(n_chan):
                    #s1 = time.time()
                    flux = point_source_flux[i_point_source,i_time//f_sf_time, i_chan//f_sf_chan, :]
                    #print("s1",time.time()-s1)
                    
                    ################### Apply primary beams #####################
                    bm1_indx = beam_model_map[i_ant_1]
                    bm2_indx = beam_model_map[i_ant_2]
                    bm1_type = beam_types[i_ant_1]
                    bm2_type = beam_types[i_ant_2]

                    #s2 = time.time()
                    if do_pointing:
                        flux_scaled, outside_beam = _calc_pb_scale(flux,sep_1,sep_2,bm1_indx,bm2_indx,bm1_type,bm2_type,lm_sin_1,lm_sin_2,beam_models_type0,beam_models_type1,pa,freq_chan[i_chan],mueller_selection,do_pointing)
                    else:
                        flux_scaled, outside_beam = _calc_pb_scale(flux,sep,sep,bm1_indx,bm2_indx,bm1_type,bm2_type,lm_sin,lm_sin,beam_models_type0,beam_models_type1,pa,freq_chan[i_chan],mueller_selection,do_pointing)
                    #print("s2",time.time()-s2)
                    
                    if not outside_beam:
                        #s3 = time.time()
                        phase_scaled = 1j*phase*freq_chan[i_chan]/c
                        #print("s3",time.time()-s3)
                        
                        #s4 = time.time()
                        for i_pol in range(n_pol):
                            vis_data[i_time,i_baseline,i_chan,i_pol] = vis_data[i_time,i_baseline,i_chan,i_pol] + flux_scaled[pol[i_pol]]*np.exp(phase_scaled)/(1-lmn_rot[2])
                        #print("s4",time.time()-s4)


