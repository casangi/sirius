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
from sirius._sirius_utils._sirius_logger import _get_sirius_logger

#https://colab.research.google.com/drive/13kdPUQW8AD1amPzKnCqwLsA03kdI3MrP?ts=61001ba6
#https://safe.nrao.edu/wiki/pub/ALMA/SimulatorCookbook/corruptguide.pdf
#https://casadocs.readthedocs.io/en/latest/api/tt/casatools.simulator.html#casatools.simulator.simulator.setnoise
# SimACohCalc
# https://casaguides.nrao.edu/index.php/Simulating_ngVLA_Data-CASA5.4.1
# https://casaguides.nrao.edu/index.php/Corrupting_Simulated_Data_(Simulator_Tool)
# https://library.nrao.edu/public/memos/alma/main/memo128.pdf

#def calc_a_noise(vis,uvw,beam_model_map,beam_models, antenna1, antenna2, noise_parms):
#    return 0




def calc_noise_chunk(vis_shape,uvw,beam_model_map,beam_models, antenna1, antenna2, noise_parms, check_parms=True):
    """
    Add noise to visibilities.
    
    Parameters
    ----------
    vis_data_shape : float np.array, [4]
        Dimensions of visibility data [n_time, n_baseline, n_chan, n_pol].
    uvw : float np.array, [n_time,n_baseline,3]
        Spatial frequency coordinates. Can be None if no autocorrelations are present.
    beam_model_map: int np.array, [n_ant]
        Each element in beam_model_map is an index into beam_models.
    beam_models: list
        List of beam models to use. Beam models can be any combination of function parameter dictionaries, image xr.Datasets or Zernike polynomial coefficient xr.Datasets.
    antenna1: np.array of int, [n_baseline]
        The indices of the first antenna in the baseline pairs. The _calc_baseline_indx_pair function in sirius._sirius_utils._array_utils can be used to calculate these values.
    antenna2: np.array of int, [n_baseline]
        The indices of the second antenna in the baseline pairs. The _calc_baseline_indx_pair function in sirius._sirius_utils._array_utils can be used to calculate these values.
    noise_parms: dict
        Set various system parameters from which the thermal (ie, random additive) noise level will be calculated.
        See https://casadocs.readthedocs.io/en/stable/api/tt/casatools.simulator.html#casatools.simulator.simulator.setnoise.
    noise_parms['mode']: str, default='tsys-manual', options=['simplenoise','tsys-manual','tsys-atm']
        Currently only 'tsys-manual' is implemented.
    noise_parms['t_atmos']: , float, default = 250.0, Kelvin
        Atmosphere Temperature (mode='tsys-manual')
    noise_parms['tau']: float, default = 0.1
        Zenith Atmospheric Opacity (if tsys-manual). Currently the effect of Zenith Atmospheric Opacity (Tau) is not included in the noise modeling.
    noise_parms['ant_efficiency']: float, default=0.8
        Antenna efficiency.
    noise_parms['spill_efficiency']: float, default=0.85
        Forward spillover efficiency.
    noise_parms['corr_efficiency']: float, default=0.88
        Correlator efficiency.
    noise_parms['quantization_efficiency']: float, default=0.96
        Digitizer quantization efficiency.
    noise_parms['t_receiver']: float, default=50.0, Kelvin
        Receiver temperature (ie, all non-atmospheric Tsys contributions).
    noise_parms['t_cmb']: float, default=2.725, Kelvin
        Cosmic microwave background temperature.
    noise_parms['auto_corr']: bool, default=False
        If True autocorrelations are also calculated.
    noise_parms['freq_resolution']: float, Hz
        Width of a single channel.
    noise_parms['time_delta']: float, s
        Integration time.
    check_parms: bool
        Check input parameters and asign defaults.
        
    Returns
    -------
    noise : complex np.array,  [n_time, n_baseline, n_chan, n_pol]
    
    weight :  float np.array,  [n_time, n_baseline, n_pol]
    
    sigma :  float np.array,  [n_time, n_baseline, n_pol]
    """
    logger = _get_sirius_logger()
    
    from sirius_data._constants import k_B
    n_time, n_baseline, n_chan, n_pol = vis_shape
    dish_sizes = get_dish_sizes(beam_models)
    
    #Calculate Effective Dish Area
    dish_size_per_ant = dish_sizes[beam_model_map] # n_ant array with the size of each dish in meter.
    baseline_dish_diam_product = dish_size_per_ant[antenna1]*dish_size_per_ant[antenna2] # [n_baseline] array of dish_i*dish_j.
    A_eff = noise_parms['ant_efficiency']*(np.pi*baseline_dish_diam_product)/4 # [n_baseline] array
    
    logger.info('A_eff shape: ' +  str(A_eff.shape) + ', value [0,0]: ' + str(A_eff[0]))
    
    #Calculate system temperature t_sys. For now tau (Zenith Atmospheric Opacity) will be set to 0 (elevation calculation not yet implemented)
    t_sys = noise_parms['t_receiver'] + noise_parms['t_atmos']*(1-noise_parms['spill_efficiency']) + noise_parms['t_cmb']
    # When Zenith Atmospheric Opacity is included t_sys will be a function of time and baseline:
    # t_sys [n_time,n_baseline]
    # azel = ... # [n_time,n_ant,1] Azimuth Elevation, calculate using astropy
    # elevation_1 = azel[:,antenna1,0] # [n_time,n_baseline], remember antenna1 is an [n_basline] array with the antenna indx for the first antenna in the baseline pair.
    # airmass_1 = 1.0/sin(elevation_1) # [n_time,n_baseline]
    # AO = sqrt(e^{tau*airmass_1})*sqrt(e^{tau*airmass_2}) # AO atmospheric opacity factor [n_time,n_baseline]
    # t_sys = t_receiver * AO + t_atmos (AO - eta_spill) + t_cmb # [n_time,n_baseline]
    logger.info('System temperature: ' +  str(t_sys))
    
    # RMS noise level at the correlator ourput (for only a single component real or imagenary)
    eta_corr = noise_parms['corr_efficiency']
    eta_q = noise_parms['quantization_efficiency']
    del_nu = noise_parms['freq_resolution']
    del_t = noise_parms['time_delta']
    sigma = np.sqrt(2)*k_B*t_sys*(10**26)/(eta_corr*eta_q*A_eff*np.sqrt(del_nu*del_t)) # [n_baseline] array
    sigma = np.tile(sigma[None,:], (n_time,1)) # [n_time,n_baseline] array
    
    logger.info('eta_corr ' +  str(eta_corr) + ', eta_q: ' + str(eta_q) + ', del_nu: ' + str(del_nu) + ', del_t: ' + str(del_t))
    
    if not noise_parms['auto_corr']:
        sigma_full_dim = np.tile(sigma[:,:,None,None],(1,1,n_chan,n_pol))
        noise_re = np.random.normal(loc=0.0,scale=sigma_full_dim)
        noise_im = np.random.normal(loc=0.0,scale=sigma_full_dim)
        
        noise = noise_re + 1j*noise_im
    else:
        #Most probaly will have to include the autocorrelation weight.
        #This is incorrect
        auto_corr_mask = ((uvw[:,:,0]!=0) & (uvw[:,:,1]!=0)).astype(int)
        auto_corr_scale = np.copy(auto_corr_mask)
        auto_corr_scale[auto_corr_scale==0] = np.sqrt(2)

        sigma_full_dim = np.tile(sigma[:,:,None,None],(1,1,n_chan,n_pol))
        noise_re = np.random.normal(loc=0.0,scale=sigma_full_dim*auto_corr_scale[:,:,None,None])
        noise_im = np.random.normal(loc=0.0,scale=sigma_full_dim)
        
        noise = noise_re + 1j*noise_im*auto_corr_mask[:,:,None,None]
        
    sigma = sigma/np.sqrt(2) #The sqrt(2) is required since the sigma is not stored seperately for the real and imagenary component. (see equation 6.51 in Thompson 2nd edition). For a naturally weighted image the expected rms noise in the residual (after complete deconvolution) should be 1/np.sqrt(np.sum(weight)*n_channels).
    weight = 1.0/(sigma**2)
    logger.info('Sigma shape: ' +  str(sigma.shape) + ', value [0,0]: ' + str(sigma[0,0]))
    
    return noise, np.tile(weight[:,:,None],(1,1,n_pol)), np.tile(sigma[:,:,None],(1,1,n_pol))
    
    
def get_dish_sizes(beam_models):
    dish_sizes = []
    for bm in beam_models:
        if "J" in bm:
            dish_sizes.append(bm.attrs['dish_diam'])
        else:
            dish_sizes.append(bm['dish_diam'])
   
        
    return np.array(dish_sizes)





