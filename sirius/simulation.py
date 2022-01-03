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

import os
import time
import numpy as np
import dask.array as da
import dask
import xarray as xr
from itertools import cycle
import itertools
import copy
from astropy import units as u
from ._sirius_utils._array_utils import _ndim_list, _calc_n_baseline, _is_subset
from ._parm_utils._check_beam_parms import _check_beam_parms
from ._parm_utils._check_uvw_parms import _check_uvw_parms
from ._parm_utils._check_save_parms import _check_save_parms
from ._parm_utils._check_noise_parms import _check_noise_parms
from sirius_data._constants import pol_codes_RL, pol_codes_XY
from sirius.calc_a_noise import calc_a_noise_chunk
from sirius.calc_uvw import calc_uvw_chunk 
from sirius.calc_vis import calc_vis_chunk
from sirius.calc_beam import evaluate_beam_models
from sirius.dio import write_to_ms


def simulation(point_source_flux, point_source_ra_dec, pointing_ra_dec, phase_center_ra_dec, phase_center_names, beam_parms,beam_models,beam_model_map,uvw_parms, tel_xds, time_xda, chan_xda, pol, noise_parms, save_parms):
    """
    Creates a dask graph that computes a simulated measurement set and triggers a compute and saves the ms to disk.
    
    Parameters
    ----------
    point_source_flux: float np.array, [n_point_sources,n_time, n_chan, n_pol], (singleton: n_time, n_chan), Janskys
        The flux of the point sources.
    point_source_ra_dec: float np.array, [n_time, n_point_sources, 2], (singleton: n_time), radians
        The position of the point sources.
    pointing_ra_dec: float np.array, [n_time, n_ant, 2], (singleton: n_time, n_ant), radians
        Pointings of antennas, if they are different from the phase center. Set to None if no pointing offsets are required.
    phase_center_ra_dec: float np.array, [n_time, 2], (singleton: n_time), radians
        Phase center of array.
    phase_center_names: str np.array, [n_time], (singleton: n_time)
        Strings that are used to label phase centers.
    beam_parms: dict
    beam_parms['mueller_selection']: int np.array, default=np.array([ 0, 5, 10, 15])
        The elements in the 4x4 beam Mueller matrix to use. The elements are numbered row wise.
        For example [ 0, 5, 10, 15] are the diagonal elements.
    beam_parms['pa_radius']: float, default=0.2, radians
        The change in parallactic angle that will trigger the calculation of a new beam when using Zernike polynomial aperture models.
    beam_parms['image_size']: int np.array, default=np.array([1000,1000])
        Size of the beam image generated from the Zernike polynomial coefficients.
    beam_parms['fov_scaling']: int, default=15
        Used to scale the size of the beam image which is given fov_scaling*(1.22 *c/(dish_diam*frequency)).
    beam_parms['zernike_freq_interp']: str, default='nearest', options=['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
        What interpolation method to use for Zernike polynomial coefficients.
    beam_models: list
        List of beam models to use. Beam models can be any combination of function parameter dictionaries, image xr.Datasets or Zernike polynomial coefficient xr.Datasets.
    beam_model_map: int np.array, [n_ant]
        Each element in beam_model_map is an index into beam_models.
    uvw_parms: dict
    uvw_parms['calc_method']: str, default='astropy', options=['astropy','casa']
        Astropy coordinates or CASA tool measures can be used to calculate uvw coordinates.
    uvw_parms['auto_corr']: bool, default=False
        If True autocorrelations are also calculated.
    tel_xds: xr.Dataset
        An xarray dataset of the radio telescope array layout (see zarr files in sirius_data/telescope_layout/data/ for examples). 
    time_xda: xr.DataArray
        Time series xarray array.
    chan_xda: xr.DataArray
        Channel frequencies xarray array.
    pol: int np.array 
        Must be a subset of ['RR','RL','LR','LL'] => [5,6,7,8] or ['XX','XY','YX','YY'] => [9,10,11,12].
    noise_parms: dict
        Set various system parameters from which the thermal (ie, random additive) noise level will be calculated.
        See https://casadocs.readthedocs.io/en/stable/api/tt/casatools.simulator.html#casatools.simulator.simulator.setnoise.
    noise_parms['mode']: str, default='tsys-manuel', options=['simplenoise','tsys-manuel','tsys-atm']
        Currently only 'tsys-manuel' is implemented.
    noise_parms['t_atmos']: , float, default = 250.0, Kelvin
        Temperature of atmosphere (mode='tsys-manual')
    noise_parms['tau']: float, default = 0.1
        Zenith Atmospheric Opacity (if tsys-manual). Currently the effect of Zenith Atmospheric Opacity (Tau) is not included in the noise modeling.
    noise_parms['ant_efficiency']: float, default=0.8
        Antenna efficiency
    noise_parms['spill_efficiency']: float, default=0.85
        Forward spillover efficiency.
    noise_parms['corr_efficiency']: float, default=0.88
        Correlation efficiency.
    noise_parms['t_receiver']: float, default=50.0, Kelvin
        Receiver temp (ie, all non-atmospheric Tsys contributions).
    noise_parms['t_ground']: float, default=270.0, Kelvin
        Temperature of ground/spill.
    noise_parms['t_cmb']: float, default=2.725, Kelvin
        Cosmic microwave background temperature.
    save_parms: dict
    save_parms['write_to_ms']: bool, default=True
        Name of the ms file to create. If None no compute is triggered and an xds that contains the graph (lazy) that calculates the visibilities, uvw coordinates, weights and timing information is returned.
    save_parms['DAG_name_vis_uvw_gen']: str, default=False
        Creates a DAG diagram png, named save_parms['DAG_name_write'], of how the visibilities and uvw coordinates are calculated.
    save_parms['DAG_name_write']: str, default=False
        Creates a DAG diagram png, named save_parms['DAG_name_write'], of how the ms is created with name.
    save_parms['ms_name']:str, default='sirius_sim.ms'
    
    Returns
    -------
    ms_xds: xr.Dataset
        If save_parms['ms_name'] is not None a dask ms xds is retruned (a compute is triggered and the ms is saved to disk). 
        If save_parms['ms_name'] is None an xds that contains the graph (lazy) that calculates the visibilities, uvw coordinates, weights and timing information is returned.
    """
    
    ########################
    ### Check Parameters ###
    ########################
    
    _beam_parms = copy.deepcopy(beam_parms)
    _uvw_parms = copy.deepcopy(uvw_parms)
    _save_parms = copy.deepcopy(save_parms)
    _noise_parms = copy.deepcopy(noise_parms)
    assert(_check_uvw_parms(_uvw_parms)), "######### ERROR: calc_uvw uvw_parms checking failed."
    assert(_check_beam_parms(_beam_parms)), "######### ERROR: beam_parms checking failed."
    if noise_parms is not None:
        _noise_parms['freq_resolution'] = chan_xda.freq_resolution
        _noise_parms['time_delta'] = time_xda.time_delta
        _noise_parms['auto_corr'] = _uvw_parms['auto_corr']
        assert(_check_noise_parms(_noise_parms)), "######### ERROR: beam_parms checking failed."
    assert(_check_save_parms(_save_parms)), "######### ERROR: save_parms checking failed."
    
    pol = np.array(pol)
    assert(_is_subset(pol_codes_RL,pol) or _is_subset(pol_codes_XY,pol)), 'Pol selection invalid, must either be subset of [5,6,7,8] or [9,10,11,12] but is '
    
    #Get dimensions of data.
    n_time = len(time_xda)
    n_ant = tel_xds.dims['ant_name']
    n_baselines = _calc_n_baseline(n_ant,_uvw_parms['auto_corr'])
    n_chan = len(chan_xda)
    n_pol = len(pol)
    
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
    
    #Find all singleton dimensions, so that indexing is done correctly.
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
    
    ###################
    ### Build graph ###
    ###################
    
    # Number of parallel processes will be equal to n_time_chunks x n_chan_chunks.
    n_time_chunks = time_xda.data.numblocks[0]
    n_chan_chunks = chan_xda.data.numblocks[0]
    
    #Iter over time,chan
    iter_chunks_indx = itertools.product(np.arange(n_time_chunks), np.arange(n_chan_chunks))
    n_pol = len(pol)
    
    #Create empty n-dimensional lists where delayed arrays will be stored.
    vis_list = _ndim_list((n_time_chunks,1,n_chan_chunks,1))
    uvw_list = _ndim_list((n_time_chunks,1,1))
    weight_list = _ndim_list((n_time_chunks,1,1))
    sigma_list = _ndim_list((n_time_chunks,1,1))
    t_list = _ndim_list((n_time_chunks,n_chan_chunks,1))
    
    for c_time, c_chan in iter_chunks_indx:
        #print(c_time,c_chan)
        time_chunk = time_xda.data.partitions[c_time]
        chan_chunk = chan_xda.data.partitions[c_chan]
        
        s_time = c_time*time_xda.data.chunks[0][0]
        e_time = c_time*time_xda.data.chunks[0][0] + time_xda.data.chunks[0][c_time] - 1 #-1 needed for // to work.
        s_chan = c_chan*chan_xda.data.chunks[0][0]
        e_chan = c_chan*chan_xda.data.chunks[0][0] + chan_xda.data.chunks[0][c_chan] - 1 #-1 needed for // to work.
        
        point_source_flux_chunk = point_source_flux[:,s_time//f_sf_time:e_time//f_sf_time+1,s_chan//f_sf_chan:e_chan//f_sf_chan+1,:]
        point_source_ra_dec_chunk = point_source_ra_dec[s_time//f_ps_time:e_time//f_ps_time+1,:,:]
        phase_center_ra_dec_chunk = phase_center_ra_dec[s_time//f_pc_time:e_time//f_pc_time+1,:]
        
        if do_pointing:
            pointing_ra_dec_chunk = pointing_ra_dec[s_time//f_pt_time:e_time//f_pt_time+1,:,:]
        else:
            pointing_ra_dec_chunk = None

        ### TO DO ###
        # Subselect channels for each beam_model with channel axis
        
        sim_chunk = dask.delayed(simulation_chunk)(
            dask.delayed(point_source_flux_chunk),
            dask.delayed(point_source_ra_dec_chunk),
            dask.delayed(pointing_ra_dec_chunk),
            dask.delayed(phase_center_ra_dec_chunk),
            dask.delayed(_beam_parms),beam_models,
            dask.delayed(beam_model_map),
            dask.delayed(_uvw_parms),
            tel_xds,
            time_chunk,
            chan_chunk,
            dask.delayed(pol), dask.delayed(_noise_parms),
            dask.delayed(None))

        vis_list[c_time][0][c_chan][0] = da.from_delayed(sim_chunk[0],(len(time_chunk), n_baselines, len(chan_chunk),n_pol),dtype=np.complex)
        uvw_list[c_time][0][0] = da.from_delayed(sim_chunk[1],(len(time_chunk), n_baselines, 3),dtype=np.complex)
        weight_list[c_time][0][0] = da.from_delayed(sim_chunk[2],(len(time_chunk), n_baselines, n_pol),dtype=np.float)
        sigma_list[c_time][0][0] = da.from_delayed(sim_chunk[3],(len(time_chunk), n_baselines, n_pol),dtype=np.float)
        t_list[c_time][c_chan][0] = da.from_delayed(sim_chunk[4],((4,)),dtype=np.float)
        
    vis = da.block(vis_list)
    uvw = da.block(uvw_list)
    weight = da.block(weight_list)
    sigma = da.block(sigma_list)
    timing = da.block(t_list)
    
    if _save_parms['DAG_name_vis_uvw_gen']:
        dask.visualize([vis,uvw],filename=_save_parms['DAG_name_vis_uvw_gen'])
        
    #Create simple xds with simulated vis, uvw, weight and sigma
    vis_xds = xr.Dataset()
    coords = {'time':time_xda.data,'chan': chan_xda.data, 'pol': pol}
    vis_xds = vis_xds.assign_coords(coords)
        
    vis_xds['DATA'] = xr.DataArray(vis, dims=['time','baseline','chan','pol'])
    vis_xds['UVW'] = xr.DataArray(uvw, dims=['time','baseline','uvw'])
    vis_xds['WEIGHT'] = xr.DataArray(weight, dims=['time','baseline','pol'])
    vis_xds['SIGMA'] = xr.DataArray(sigma, dims=['time','baseline','pol'])
    vis_xds['TIMING'] = xr.DataArray(timing, dims=['time_chunk','chan_chunk','4'])
        
    if save_parms['write_to_ms']:
        ms_xds= write_to_ms(vis_xds, time_xda, chan_xda, pol, tel_xds, phase_center_names, phase_center_ra_dec, _uvw_parms['auto_corr'],_save_parms)
        return ms_xds
    else:
        return vis_xds
    

def simulation_chunk(point_source_flux, point_source_ra_dec, pointing_ra_dec, phase_center_ra_dec, beam_parms,beam_models,beam_model_map,uvw_parms, tel_xds, time_chunk, chan_chunk, pol, noise_parms, uvw_precompute=None):
    """
    Simulates uvw coordinates, interferometric visibilities and adds noise. This function does not produce a measurement set.  
    
    Parameters
    ----------
    point_source_flux: float np.array, [n_point_sources,n_time, n_chan, n_pol], (singleton: n_time, n_chan), Janskys
        The flux of the point sources.
    point_source_ra_dec: float np.array, [n_time, n_point_sources, 2], (singleton: n_time), radians
        The position of the point sources.
    pointing_ra_dec: float np.array, [n_time, n_ant, 2], (singleton: n_time, n_ant), radians
        Pointings of antennas, if they are different from the phase center. Set to None if no pointing offsets are required.
    phase_center_ra_dec: float np.array, [n_time, 2], (singleton: n_time), radians
        Phase center of array.
    beam_parms: dict
    beam_parms['mueller_selection']: int np.array, default=np.array([ 0, 5, 10, 15])
        The elements in the 4x4 beam Mueller matrix to use. The elements are numbered row wise.
        For example [ 0, 5, 10, 15] are the diagonal elements.
    beam_parms['pa_radius']: float, default=0.2, radians
        The change in parallactic angle that will trigger the calculation of a new beam when using Zernike polynomial aperture models.
    beam_parms['image_size']: int np.array, default=np.array([1000,1000])
        Size of the beam image generated from the Zernike polynomial coefficients.
    beam_parms['fov_scaling']: int, default=15
        Used to scale the size of the beam image, which is given by fov_scaling*(1.22 *c/(dish_diam*frequency)).
    beam_parms['zernike_freq_interp']: str, default='nearest', options=['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
        What interpolation method to use for Zernike polynomial coefficients.
    beam_models: list
        List of beam models to use. Beam models can be any combination of function parameter dictionaries, image xr.Datasets or Zernike polynomial coefficient xr.Datasets.
    beam_model_map: int np.array, [n_ant]
        Each element in beam_model_map is an index into beam_models.
    uvw_parms: dict
    uvw_parms['calc_method']: str, default='astropy', options=['astropy','casa']
        Astropy coordinates or CASA tool measures can be used to calculate uvw coordinates.
    uvw_parms['auto_corr']: bool, default=False
        If True autocorrelations are also calculated.
    tel_xds: xr.Dataset
        An xarray dataset of the radio telescope array layout (see zarr files in sirius_data/telescope_layout/data/ for examples). 
    time_chunk: str np.array, [n_time], 'YYYY-MM-DDTHH:MM:SS.SSS'
        Time series. Example '2019-10-03T19:00:00.000'.
    chan_chunk: float np.array, [n_chan], Hz
        Channel frequencies.
    pol: int np.array 
        Must be a subset of ['RR','RL','LR','LL'] => [5,6,7,8] or ['XX','XY','YX','YY'] => [9,10,11,12].
    noise_parms: dict
        Set various system parameters from which the thermal (ie, random additive) noise level will be calculated.
        See https://casadocs.readthedocs.io/en/stable/api/tt/casatools.simulator.html#casatools.simulator.simulator.setnoise.
    noise_parms['mode']: str, default='tsys-manuel', options=['simplenoise','tsys-manuel','tsys-atm']
        Currently only 'tsys-manuel' is implemented.
    noise_parms['t_atmos']: , float, default = 250.0, Kelvin
        Temperature of atmosphere (mode='tsys-manual')
    noise_parms['tau']: float, default = 0.1
        Zenith Atmospheric Opacity (if tsys-manual). Currently the effect of Zenith Atmospheric Opacity (Tau) is not included in the noise modeling.
    noise_parms['ant_efficiency']: float, default=0.8
        Antenna efficiency
    noise_parms['spill_efficiency']: float, default=0.85
        Forward spillover efficiency.
    noise_parms['corr_efficiency']: float, default=0.88
        Correlation efficiency.
    noise_parms['t_receiver']: float, default=50.0, Kelvin
        Receiver temp (ie, all non-atmospheric Tsys contributions).
    noise_parms['t_ground']: float, default=270.0, Kelvin
        Temperature of ground/spill.
    noise_parms['t_cmb']: float, default=2.725, Kelvin
        Cosmic microwave background temperature.
    save_parms: dict
    save_parms['write_to_ms']: bool, default=True
        Name of the ms file to create. If None no compute is triggered and an xds that contains the graph (lazy) that calculates the visibilities, uvw coordinates, weights and timing information is returned.
    save_parms['DAG_name_vis_uvw_gen']: str, default=False
        Creates a DAG diagram png, named save_parms['DAG_name_write'], of how the visibilities and uvw coordinates are calculated.
    save_parms['DAG_name_write']: str, default=False
        Creates a DAG diagram png, named save_parms['DAG_name_write'], of how the ms is created with name.
    save_parms['ms_name']:str, default='sirius_sim.ms'
    
    Returns
    -------
    vis : complex np.array, [n_time,n_baseline,n_chan,n_pol]   
        Visibility data.
    uvw : float np.array, [n_time,n_baseline,3]   
        Spatial frequency coordinates.
    weight: complex np.array, [n_time,n_baseline,n_pol]
        Data weights.
    sigma: complex np.array, [n_time,n_baseline,n_pol]
        RMS noise of data.
    t_arr: float np.array, [4]
        Timing infromation: calculate uvw, evaluate_beam_models, calculate visibilities, calculate additive noise.
    """

    #Calculate uvw coordinates
    t0 = time.time()
    if uvw_precompute is None:
        uvw, antenna1,antenna2 = calc_uvw_chunk(tel_xds, time_chunk, phase_center_ra_dec, uvw_parms,check_parms=False)
    else:
        from ._sirius_utils._array_utils import _calc_baseline_indx_pair
        n_ant = len(ant_pos)
        antenna1,antenna2=_calc_baseline_indx_pair(n_ant,uvw_parms['auto_corr'])
        uvw = uvw_precompute
    t0 = time.time()-t0
      
    t1 = time.time()
    #Evaluate zpc files
    eval_beam_models, pa = evaluate_beam_models(beam_models,time_chunk,chan_chunk,phase_center_ra_dec,tel_xds.site_pos,beam_parms,check_parms=False)
    t1 = time.time()-t1

    #Calculate visibilities
    t2 = time.time()
    vis_data_shape =  np.concatenate((uvw.shape[0:2],[len(chan_chunk)],[len(pol)]))
    vis =calc_vis_chunk(uvw,vis_data_shape,point_source_flux,point_source_ra_dec,pointing_ra_dec,phase_center_ra_dec,antenna1,antenna2,chan_chunk,beam_model_map,eval_beam_models, pa, pol, beam_parms['mueller_selection'],check_parms=False)
    t2 = time.time()-t2

    #Calculate and add noise
    t3 = time.time()
    if noise_parms is not None:
        noise, weight, sigma = calc_a_noise_chunk(vis.shape,uvw,beam_model_map,eval_beam_models, antenna1, antenna2,noise_parms,check_parms=False)
        vis = vis + noise
    else:
        n_time, n_baseline, n_chan, n_pol = vis.shape
        weight = np.ones((n_time,n_baseline,n_pol))
        sigma = np.ones((n_time,n_baseline,n_pol))
    t3 = time.time()-t3
    
    t_arr = np.array([t0,t1,t2,t3])
    return vis, uvw, weight, sigma, t_arr
