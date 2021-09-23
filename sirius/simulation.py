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



def simulation(point_source_flux, point_source_ra_dec, pointing_ra_dec, phase_center_ra_dec, beam_parms,beam_models,beam_model_map,uvw_parms, tel_xds, time_str, freq_chan, pol, uvw_precompute=None, a_noise_parms=None):
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
    from ._parm_utils._check_beam_parms import _check_beam_parms
    import numpy as np
    import copy
    
    _beam_parms = copy.deepcopy(beam_parms)
    _uvw_parms = copy.deepcopy(uvw_parms)
    assert(_check_beam_parms(_uvw_parms)), "######### ERROR: calc_uvw uvw_parms checking failed."
    assert(_check_beam_parms(_beam_parms)), "######### ERROR: make_ant_sky_jones beam_parms checking failed."
    
    #Calculate uvw coordinates
    if uvw_precompute is None:
        uvw, antenna1,antenna2 = calc_uvw(tel_xds, time_str, phase_center_ra_dec, _uvw_parms,check_parms=False)
    else:
        from ._sirius_utils._array_utils import _calc_baseline_indx_pair
        n_ant = len(ant_pos)
        antenna1,antenna2=_calc_baseline_indx_pair(n_ant,_uvw_parms['auto_corr'])
        uvw = uvw_precompute
          
    #Evaluate zpc files
    eval_beam_models, pa = evaluate_beam_models(beam_models,_beam_parms,freq_chan,phase_center_ra_dec,time_str,uvw_parms['site'])
    
    print(eval_beam_models)
#
    #Calculate visibilities
    #shape, point_source_flux, point_source_ra_dec, pointing_ra_dec, phase_center_ra_dec, antenna1, antenna2, n_ant, freq_chan, pb_parms = calc_vis_tuple
    
    vis_data_shape =  np.concatenate((uvw.shape[0:2],[len(freq_chan)],[len(pol)]))
    
    #print('pol',pol)
    vis =calc_vis(uvw,vis_data_shape,point_source_flux,point_source_ra_dec,pointing_ra_dec,phase_center_ra_dec,antenna1,antenna2,freq_chan,beam_model_map,eval_beam_models, pa, pol, _beam_parms['mueller_selection'])

    if a_noise_parms is not None:
        #calc_a_noise(vis,eval_beam_models,a_noise_parms)
        from sirius import  calc_a_noise
        print(calc_a_noise)
        calc_a_noise(vis,eval_beam_models,a_noise_parms)
    return vis, uvw

    
def make_time_da(time_start='2019-10-03T19:00:00.000',time_delta=3600,n_samples=10,n_chunks=4):
    """
    Create a time series dask array.
    Parameters
    ----------
    -------
    time_da : dask.array
    """
    from astropy import units as u
    from astropy.timeseries import TimeSeries
    import dask.array as da
    import numpy as np
    ts = np.array(TimeSeries(time_start=time_start,time_delta=time_delta*u.s,n_samples=n_samples).time.value)
    chunksize = int(np.ceil(n_samples/n_chunks))
    time_da = da.from_array(ts, chunks=chunksize)
    print('Number of chunks ', len(time_da.chunks[0]))
    return time_da

def make_chan_da(freq_start = 3*10**9, freq_delta = 0.4*10**9, freq_resolution=0.01*10**9, n_channels=3, n_chunks=3):
    """
    Create a time series dask array.
    Parameters
    ----------
    -------
    time_da : dask.array
    """
    from astropy import units as u
    import dask.array as da
    import numpy as np
    freq_chan = np.arange(0,n_channels)*freq_delta + freq_start
    chunksize = int(np.ceil(n_channels/n_chunks))
    chan_da = da.from_array(freq_chan, chunks=chunksize)
    print('Number of chunks ', len(chan_da.chunks[0]))
    return chan_da
      
