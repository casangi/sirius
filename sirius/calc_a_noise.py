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

#https://colab.research.google.com/drive/13kdPUQW8AD1amPzKnCqwLsA03kdI3MrP?ts=61001ba6
#https://safe.nrao.edu/wiki/pub/ALMA/SimulatorCookbook/corruptguide.pdf
#https://casadocs.readthedocs.io/en/latest/api/tt/casatools.simulator.html#casatools.simulator.simulator.setnoise
# SimACohCalc
# https://casaguides.nrao.edu/index.php/Simulating_ngVLA_Data-CASA5.4.1
# https://casaguides.nrao.edu/index.php/Corrupting_Simulated_Data_(Simulator_Tool)
# https://library.nrao.edu/public/memos/alma/main/memo128.pdf

def calc_a_noise(vis,uvw,beam_model_map,eval_beam_models, antenna1, antenna2, noise_parms):
    """
    Add noise to visibilities.
    
    Parameters
    ----------
    vis : np.array
    Returns
    -------
    vis : np.array
    """
    from ._sirius_utils._constants import k_B
    noise = np.zeros(vis.shape,dtype=np.complex)
    
    n_time, n_baseline, n_chan, n_pol = vis.shape
    dish_sizes = get_dish_sizes(eval_beam_models)
    
    #For now tau (Zenith Atmospheric Opacity) will be set to 0 (don't have to do elevation calculation)
    twiddle = 100
    #print('Twiddle 2 is active.', twiddle)
    factor = (4*np.sqrt(2)*k_B*(10**23))/(noise_parms['ant_efficiency']*noise_parms['corr_efficiency']*np.pi)*twiddle
    
    
    t_sys = noise_parms['t_receiver'] + noise_parms['t_atmos']*(1-noise_parms['spill_efficiency']) + noise_parms['t_cmb']
    
    del_nu = noise_parms['freq_resolution'] #should it be the total bandwidth of the spectral window?
    del_t = noise_parms['time_delta']
    
    dish_size_per_ant = dish_sizes[beam_model_map] # n_ant array with the size of each dish in meter.
    baseline_dish_diam_product = dish_size_per_ant[antenna1]*dish_size_per_ant[antenna2] # n_baseline array of dish_i*dish_j.
    
    #ms v2 weight and sigma do not have a spectral component (if we move to ms v3 this will change)
    sigma = np.tile(factor*t_sys/(baseline_dish_diam_product*np.sqrt(del_nu*del_t))[None,:], (n_time,1))
    #print('sigma[0,:]',sigma[0,:])
    #sigma[sigma < 10**-9] = 10**-9 in SimACohCalc
    weight = 1.0/(sigma**2)
    
    if not noise_parms['auto_corr']:
        sigma_full_dim = np.tile(sigma[:,:,None,None],(1,1,n_chan,n_pol))
        noise_re = np.random.normal(loc=0.0,scale=sigma_full_dim)
        noise_im = np.random.normal(loc=0.0,scale=sigma_full_dim)
        
        vis = vis + noise_re + 1j*noise_im
    else:
        #Most probaly will have to include the autocorrelation weight.
        auto_corr_mask = ((uvw[:,:,0]!=0) & (uvw[:,:,1]!=0)).astype(int)
        auto_corr_scale = np.copy(auto_corr_mask)
        auto_corr_scale[auto_corr_scale==0] = np.sqrt(2)

        sigma_full_dim = np.tile(sigma[:,:,None,None],(1,1,n_chan,n_pol))
        noise_re = np.random.normal(loc=0.0,scale=sigma_full_dim*auto_corr_scale[:,:,None,None])
        noise_im = np.random.normal(loc=0.0,scale=sigma_full_dim)
        
        vis = vis + noise_re + 1j*noise_im*auto_corr_mask[:,:,None,None]
    
    return vis, np.tile(weight[:,:,None],(1,1,n_pol)), np.tile(sigma[:,:,None],(1,1,n_pol))
    
    
def get_dish_sizes(eval_beam_models):
    dish_sizes = []
    for bm in eval_beam_models:
        if "J" in bm:
            dish_sizes.append(bm.attrs['dish_diam'])
        else:
            dish_sizes.append(bm['dish_diam'])
   
        
    return np.array(dish_sizes)





