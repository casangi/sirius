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

#Parallel writes https://github.com/ornladios/ADIOS2
#https://github.com/casacore/casacore/issues/432
#https://github.com/casacore/casacore/issues/729

import os
import dask.array as da
import dask
import xarray as xr
import numpy as np
from astropy import units as u
import time

def simulation(point_source_flux, point_source_ra_dec, pointing_ra_dec, phase_center_ra_dec, phase_center_names, beam_parms,beam_models,beam_model_map,uvw_parms, tel_xds, time_xda, chan_xda, pol, noise_parms, save_parms):
    """
    Simulate a interferometric visibilities and uvw coordinates. Dask enabled function
    
    Parameters
    ----------
    point_source_flux : np.array
    Returns
    -------
    vis : np.array
    uvw : np.array
    """

    from sirius import calc_vis, calc_uvw, calc_a_noise
    from sirius.make_ant_sky_jones import evaluate_beam_models
    from ._sirius_utils._array_utils import _ndim_list
    from ._parm_utils._check_beam_parms import _check_beam_parms
    from ._parm_utils._check_uvw_parms import _check_uvw_parms
    from ._parm_utils._check_save_parms import _check_save_parms
    from ._parm_utils._check_noise_parms import _check_noise_parms
    from ._sirius_utils._array_utils import _is_subset
    from sirius_data._constants import pol_codes_RL, pol_codes_XY
    import numpy as np
    from itertools import cycle
    import itertools
    import copy
    
    
    _beam_parms = copy.deepcopy(beam_parms)
    _uvw_parms = copy.deepcopy(uvw_parms)
    _save_parms = copy.deepcopy(save_parms)
    _noise_parms = copy.deepcopy(noise_parms)
    assert(_check_uvw_parms(_uvw_parms)), "######### ERROR: calc_uvw uvw_parms checking failed."
    assert(_check_beam_parms(_beam_parms)), "######### ERROR: make_ant_sky_jones beam_parms checking failed."
    if noise_parms is not None:
        _noise_parms['freq_resolution'] = chan_xda.freq_resolution
        _noise_parms['time_delta'] = time_xda.time_delta
        _noise_parms['auto_corr'] = _uvw_parms['auto_corr']
        assert(_check_noise_parms(_noise_parms)), "######### ERROR: make_ant_sky_jones beam_parms checking failed."
    assert(_check_save_parms(_save_parms)), "######### ERROR: save_parms checking failed."
    
    pol = np.array(pol)
    assert(_is_subset(pol_codes_RL,pol) or _is_subset(pol_codes_XY,pol)), print('Pol selection invalid, must either be subset of [5,6,7,8] or [9,10,11,12] but is ', pol)
    
    ### TO DO ###
    #Add checks dims n_time, n_chan, n_ant, n_point_sources are consistant or singleton when allowed.
    
    n_time = len(time_xda)
    n_chan = len(chan_xda)
    n_ant = tel_xds.dims['ant_name']
    #print(n_time,n_chan,n_ant)
    
    
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
    
    
    n_time_chunks = time_xda.data.numblocks[0]
    n_chan_chunks = chan_xda.data.numblocks[0]
    
    #Iter over time,chan
    iter_chunks_indx = itertools.product(np.arange(n_time_chunks), np.arange(n_chan_chunks))
    n_pol = len(pol)
    
    vis_list = _ndim_list((n_time_chunks,1,n_chan_chunks,1))
    uvw_list = _ndim_list((n_time_chunks,1,1))
    weight_list = _ndim_list((n_time_chunks,1,1))
    sigma_list = _ndim_list((n_time_chunks,1,1))
    
    from ._sirius_utils._array_utils import _calc_n_baseline
    n_baselines = _calc_n_baseline(n_ant,_uvw_parms['auto_corr'])
    
    # Build graph
    for c_time, c_chan in iter_chunks_indx:
        
        time_chunk = time_xda.data.partitions[c_time]
        chan_chunk = chan_xda.data.partitions[c_chan]
        
        #print(time_da.chunks[0][0])
        s_time = c_time*time_xda.data.chunks[0][0]
        e_time = c_time*time_xda.data.chunks[0][0] + time_xda.data.chunks[0][c_time] - 1 #-1 needed for // to work.
        s_chan = c_chan*chan_xda.data.chunks[0][0]
        e_chan = c_chan*chan_xda.data.chunks[0][0] + chan_xda.data.chunks[0][c_chan] - 1 #-1 needed for // to work.
        
        #print(s_time_indx,e_time_indx + 1)
        #print(s_time_indx//f_sf_time,e_time_indx//f_sf_time + 1)
        
        #point_source_flux: np.array [n_point_sources,n_time, n_chan, n_pol] (singleton: n_time, n_chan, n_pol)
        point_source_flux_chunk = point_source_flux[:,s_time//f_sf_time:e_time//f_sf_time+1,s_chan//f_sf_chan:e_chan//f_sf_chan+1,:]
        point_source_ra_dec_chunk = point_source_ra_dec[s_time//f_ps_time:e_time//f_ps_time+1,:,:]
        phase_center_ra_dec_chunk = phase_center_ra_dec[s_time//f_pc_time:e_time//f_pc_time+1,:]
        
        if do_pointing:
            pointing_ra_dec_chunk = pointing_ra_dec[s_time//f_pt_time:e_time//f_pt_time+1,:,:]
        else:
            pointing_ra_dec_chunk = None

        ### TO DO ###
        # Subselect channels for each beam_model with channel axis
        
        #print('time_chunk',time_chunk, chan_chunk)
    
        
        #print(c_time, c_chan)
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
        #sim_chunk.compute()
        
        vis_list[c_time][0][c_chan][0] = da.from_delayed(sim_chunk[0],(len(time_chunk), n_baselines, len(chan_chunk),n_pol),dtype=np.complex)
        uvw_list[c_time][0][0] = da.from_delayed(sim_chunk[1],(len(time_chunk), n_baselines, 3),dtype=np.complex)
        weight_list[c_time][0][0] = da.from_delayed(sim_chunk[2],(len(time_chunk), n_baselines, n_pol),dtype=np.float)
        sigma_list[c_time][0][0] = da.from_delayed(sim_chunk[3],(len(time_chunk), n_baselines, n_pol),dtype=np.float)
        
    vis = da.block(vis_list)
    uvw = da.block(uvw_list)
    weight = da.block(weight_list)
    sigma = da.block(sigma_list)
    
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
    ###################
    
    write_to_ms(vis_xds, time_xda, chan_xda, pol, tel_xds, phase_center_names, phase_center_ra_dec, _uvw_parms['auto_corr'],_save_parms)
    
    return vis_xds
    

def simulation_chunk(point_source_flux, point_source_ra_dec, pointing_ra_dec, phase_center_ra_dec, beam_parms,beam_models,beam_model_map,uvw_parms, tel_xds, time_chunk, chan_chunk, pol, _noise_parms, uvw_precompute=None):
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

    from sirius import calc_vis, calc_uvw, calc_a_noise
    from sirius.make_ant_sky_jones import evaluate_beam_models
    import numpy as np
    
    #Calculate uvw coordinates
    if uvw_precompute is None:
        uvw, antenna1,antenna2 = calc_uvw(tel_xds, time_chunk, phase_center_ra_dec, uvw_parms,check_parms=False)
    else:
        from ._sirius_utils._array_utils import _calc_baseline_indx_pair
        n_ant = len(ant_pos)
        antenna1,antenna2=_calc_baseline_indx_pair(n_ant,uvw_parms['auto_corr'])
        uvw = uvw_precompute
          
    #Evaluate zpc files
    eval_beam_models, pa = evaluate_beam_models(beam_models,beam_parms,chan_chunk,phase_center_ra_dec,time_chunk,tel_xds.attrs['telescope_name'])
    
    #print(eval_beam_models)
#
    #Calculate visibilities
    #shape, point_source_flux, point_source_ra_dec, pointing_ra_dec, phase_center_ra_dec, antenna1, antenna2, n_ant, chan_chunk, pb_parms = calc_vis_tuple
    
    vis_data_shape =  np.concatenate((uvw.shape[0:2],[len(chan_chunk)],[len(pol)]))
    
    #print('pol',pol)
    vis =calc_vis(uvw,vis_data_shape,point_source_flux,point_source_ra_dec,pointing_ra_dec,phase_center_ra_dec,antenna1,antenna2,chan_chunk,beam_model_map,eval_beam_models, pa, pol, beam_parms['mueller_selection'])

    if _noise_parms is not None:
        vis, weight, sigma = calc_a_noise(vis,uvw,beam_model_map,eval_beam_models, antenna1, antenna2,_noise_parms)
    else:
        n_time, n_baseline, n_chan, n_pol = vis.shape
        weight = np.ones((n_time,n_baseline,n_pol))
        sigma = np.ones((n_time,n_baseline,n_pol))
    
    return vis, uvw, weight, sigma

    
def make_time_xda(time_start='2019-10-03T19:00:00.000',time_delta=3600,n_samples=10,n_chunks=4):
    """
    Create a time series xarray array.
    Parameters
    ----------
    -------
    time_da : dask.array
    """
    from astropy.timeseries import TimeSeries

    ts = np.array(TimeSeries(time_start=time_start,time_delta=time_delta*u.s,n_samples=n_samples).time.value)
    chunksize = int(np.ceil(n_samples/n_chunks))
    time_da = da.from_array(ts, chunks=chunksize)
    print('Number of chunks ', len(time_da.chunks[0]))
    
    time_xda = xr.DataArray(data=time_da,dims=["time"],attrs={'time_delta':float(time_delta)})
    
    return time_xda

def make_chan_xda(spw_name='sband',freq_start = 3*10**9, freq_delta = 0.4*10**9, freq_resolution=0.01*10**9, n_channels=3, n_chunks=3):
    """
    Create a time series xarray array.
    Parameters
    ----------
    -------
    time_da : dask.array
    """
    freq_chan = (np.arange(0,n_channels)*freq_delta + freq_start).astype(float) #astype(float) needed for interfacing with CASA simulator.
    chunksize = int(np.ceil(n_channels/n_chunks))
    chan_da = da.from_array(freq_chan, chunks=chunksize)
    print('Number of chunks ', len(chan_da.chunks[0]))
    
    chan_xda = xr.DataArray(data=chan_da,dims=["chan"],attrs={'freq_resolution':float(freq_resolution),'spw_name':spw_name, 'freq_delta':float(freq_delta)})
    return chan_xda
      
def write_to_ms(vis_xds, time_xda, chan_xda, pol, tel_xds, phase_center_names, phase_center_ra_dec, auto_corr,save_parms):
    from casatools import simulator
    from casatasks import mstransform
    sm = simulator()
    n_time, n_baseline, n_chan, n_pol = vis_xds.DATA.shape
    
    ant_pos = tel_xds.ANT_POS.values
    os.system('rm -rf ' + save_parms['ms_name'])
    sm.open(ms=save_parms['ms_name']);
    
    ###########################################################################################################################
    ## Set the antenna configuration
    sm.setconfig(telescopename=tel_xds.telescope_name,
                    x=ant_pos[:,0],
                    y=ant_pos[:,1],
                    z=ant_pos[:,2],
                    dishdiameter=tel_xds.DISH_DIAMETER.values,
                    mount=['alt-az'],
                    antname=list(tel_xds.ant_name.values),  #CASA can't handle an array of antenna names.
                    coordsystem='global',
                    referencelocation=tel_xds.site_pos[0]);
                    
    ## Set the polarization mode (this goes to the FEED subtable)
    from sirius_data._constants import pol_codes_RL, pol_codes_XY, pol_str
    from sirius._sirius_utils._array_utils import _is_subset
    if _is_subset(pol_codes_RL,pol): #['RR','RL','LR','LL']
        sm.setfeed(mode='perfect R L', pol=['']);
    elif _is_subset(pol_codes_XY,pol): #['XX','XY','YX','YY']
        sm.setfeed(mode='perfect X Y', pol=['']);
    else:
        assert False, print('Pol selection invalid, must either be subset of [5,6,7,8] or [9,10,11,12] but is ', pol)
    
    sm.setspwindow(spwname=chan_xda.spw_name,
               freq=chan_xda.data[0].compute(),
               deltafreq=chan_xda.freq_delta,
               freqresolution=chan_xda.freq_resolution,
               nchannels=len(chan_xda),
               refcode='LSRK',
               stokes=' '.join(pol_str[pol]));


    if auto_corr:
        sm.setauto(autocorrwt=1.0)
    else:
        sm.setauto(autocorrwt=0.0)
        
    import astropy
    from astropy.time import Time
    from astropy import units as u
    mjd = Time(time_xda.data[0:2].compute(), scale='utc')
    integration_time = (mjd[1]-mjd[0]).to('second')
    
    start_time = (mjd[0] - (integration_time/2 + 37*u.second)).mjd
    start_time_dict = {'m0': {'unit': 'd', 'value': start_time}, 'refer': 'UTC', 'type': 'epoch'}

    sm.settimes(integrationtime=integration_time.value,
             usehourangle=False,
             referencetime=start_time_dict);
        
    fields_set = []
    from collections import Counter
    field_time_count = Counter(phase_center_names)
    
    #print(field_time_count,phase_center_names)
    if len(phase_center_names) == 1: #Single field case
        field_time_count[list(field_time_count.keys())[0]] = n_time
    
    start_time = 0
    for i,ra_dec in enumerate(phase_center_ra_dec): #In future make phase_center_ra_dec a unique list
        if phase_center_names[i] not in fields_set:
            dir_dict = {'m0': {'unit': 'rad', 'value': ra_dec[0]}, 'm1': {'unit': 'rad', 'value': ra_dec[1]}, 'refer': 'J2000', 'type': 'direction'}
            sm.setfield(sourcename=phase_center_names[i],sourcedirection=dir_dict)
            fields_set.append(phase_center_names[i])
            
            stop_time = start_time + integration_time.value*field_time_count[phase_center_names[i]]
            sm.observe(sourcename=phase_center_names[i],
                spwname=chan_xda.spw_name,
                starttime= str(start_time) + 's',
                stoptime= str(stop_time) + 's')
            start_time = stop_time
         
    
    n_row = n_time*n_baseline
    
    #print(vis_data.shape)
    #print(n_row,n_time, n_baseline, n_chan, n_pol)
    
    #This code will most probably be moved into simulation if we get rid of row time baseline split.
    vis_data_reshaped = vis_xds.DATA.data.reshape((n_row, n_chan, n_pol))
    uvw_reshaped = vis_xds.UVW.data.reshape((n_row, 3))
    weight_reshaped = vis_xds.WEIGHT.data.reshape((n_row,n_pol))
    sigma_reshaped = vis_xds.SIGMA.data.reshape((n_row,n_pol))
    
    
    #weight_spectrum_reshaped = np.tile(weight_reshaped[:,None,:],(1,n_chan,1))
    
    
    
#    print(weight_reshaped.compute().shape)
#    print(sigma_reshaped.compute().shape)
#    print(weight_reshaped)
#    print(sigma_reshaped)
    
    #dask_ddid = da.full(n_row, 0, chunks=chunks['row'], dtype=np.int32)
    
    #print('vis_data_reshaped',vis_data_reshaped)
    
    from daskms import xds_to_table, xds_from_ms, Dataset
    
    #print('vis_data_reshaped.chunks',vis_data_reshaped.chunks)
    row_id = da.arange(n_row,chunks=vis_data_reshaped.chunks[0],dtype='int32')
    
    
    dataset = Dataset({'DATA': (("row", "chan", "corr"), vis_data_reshaped), 'CORRECTED_DATA': (("row", "chan", "corr"), vis_data_reshaped),'UVW': (("row","uvw"), uvw_reshaped), 'SIGMA': (("row","pol"), sigma_reshaped), 'WEIGHT': (("row","pol"), weight_reshaped),  'ROWID': (("row",),row_id)})
    #,'WEIGHT_SPECTRUM': (("row","chan","pol"), weight_spectrum_reshaped)
    ms_writes = xds_to_table(dataset, save_parms['ms_name'], columns="ALL")
    
    if save_parms['DAG_name_write']:
        dask.visualize(ms_writes,filename=save_parms['DAG_name_write'])
        
    if save_parms['write_to_ms']:
        start = time.time()
        dask.compute(ms_writes)
        print('*** Dask compute time',time.time()-start)
    

    sm.close()
    
    from casatasks import flagdata
    flagdata(vis=save_parms['ms_name'],mode='unflag')

