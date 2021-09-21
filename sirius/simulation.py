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

def simulation(point_source_flux, point_source_ra_dec, pointing_ra_dec, phase_center_ra_dec, beam_parms,beam_models,beam_model_map,uvw_parms, ant_pos, time_str, freq_chan, pol, antenna1, antenna2, pb_limit, uvw_precompute, a_noise_parms=None):
    """
    Simulate a interferometric visibilities and uvw coordinates.
    
    Parameters
    ----------
    point_source_flux : np.array
    Returns
    -------
    vis : np.array
    uvw : np.array
    """
    
    from sirius import calc_vis, calc_uvw, evaluate_beam_models, calc_a_noise
    import numpy as np
    from ._parm_utils._check_beam_parms import _check_beam_parms
    
    #Calculate uvw coordinates
    if uvw_precompute is None:
        if uvw_parms['calc_method'] == 'astropy':
            uvw = calc_uvw.calc_uvw_astropy(ant_pos, time_str, uvw_parms['site'], phase_center_ra_dec, antenna1, antenna2)
    else:
        uvw = uvw_precompute
          
    #Evaluate zpc files
    assert(_check_beam_parms(beam_parms)), "######### ERROR: make_ant_sky_jones beam_parms checking failed."
    eval_beam_models, pa = evaluate_beam_models(beam_models,beam_parms,freq_chan,phase_center_ra_dec,time_str,uvw_parms['site'])
    
    #print(eval_beam_models)
#
    #Calculate visibilities
    #shape, point_source_flux, point_source_ra_dec, pointing_ra_dec, phase_center_ra_dec, antenna1, antenna2, n_ant, freq_chan, pb_parms = calc_vis_tuple
    
    vis_data_shape =  np.concatenate((uvw.shape[0:2],[len(freq_chan)],[len(pol)]))
    
    #print('pol',pol)
    vis =calc_vis(uvw,vis_data_shape,point_source_flux,point_source_ra_dec,pointing_ra_dec,phase_center_ra_dec,antenna1,antenna2,freq_chan,beam_model_map,eval_beam_models, pa, pol, beam_parms['mueller_selection'],pb_limit)

    if a_noise_parms is not None:
        #calc_a_noise(vis,eval_beam_models,a_noise_parms)
        from sirius import  calc_a_noise
        print(calc_a_noise)
        calc_a_noise(vis,eval_beam_models,a_noise_parms)
    return vis, uvw

    
