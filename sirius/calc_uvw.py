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

from ._parm_utils._check_uvw_parms import _check_uvw_parms
from ._sirius_utils._array_utils import _calc_baseline_indx_pair
from ._sirius_utils._uvw_utils import _calc_uvw_astropy, _calc_uvw_casacore, _calc_uvw_casa
import copy
import numpy as np


def calc_uvw_chunk(tel_xds, time_str, phase_center_ra_dec, uvw_parms, check_parms=True):
    """
    Calculates a chunk of uvw values. This function forms part of a node in a Dask graph. 
    This function can be used on its own if no parallelism is required.
    
    Parameters
    ----------
    tel_xds: xr.Dataset
        An xarray dataset of the radio telescope array layout (see zarr files in sirius_data/telescope_layout/data/ for examples). 
    time_str: str np.array, [n_time], 'YYYY-MM-DDTHH:MM:SS.SSS'
        Time serie. Example '2019-10-03T19:00:00.000'.
    phase_center_ra_dec: numpy.array of floats, [n_time, 2], (singleton: n_time), radians
        Phase center of array.
    uvw_parms: dict
    uvw_parms['calc_method']: str, default='astropy', options=['astropy','casa']
        The uvw coordinates can be calculated using the astropy package or the casa measures tool.
    uvw_parms['auto_corr']: bool, default=False
        If True autocorrelations are also calculated.
    check_parms: bool
        Check input parameters and asign defaults.
    
    Returns
    -------
    uvw: float np.array, [n_time,n_baseline,3]
        The uvw coorsdinates in per wavelength units.
    antenna1: int np.array, [n_baseline]
        The indices of the first antenna in the baseline pairs.
    antenna2: int np.array, [n_baseline]
        The indices of the second antenna in the baseline pairs.
    """
    
    _uvw_parms = copy.deepcopy(uvw_parms)
    if check_parms: assert(_check_uvw_parms(_uvw_parms)), "######### ERROR: calc_uvw uvw_parms checking failed."
        
    n_ant = tel_xds.dims['ant_name']
    antenna1,antenna2=_calc_baseline_indx_pair(n_ant,_uvw_parms['auto_corr']) #The indices of the antenna pairs for each baseline.
                                                   
    if _uvw_parms['calc_method'] == 'astropy':
        uvw = _calc_uvw_astropy(tel_xds, time_str, phase_center_ra_dec, antenna1, antenna2)
    elif _uvw_parms['calc_method'] == 'casa': #CASA measures tool is not thread safe so python casacore is the prefered CASA method.
        uvw = _calc_uvw_casacore(tel_xds, time_str, phase_center_ra_dec, antenna1, antenna2)
    elif _uvw_parms['calc_method'] == 'casa_thread_unsafe': #This function is retained for future development. 
        uvw = _calc_uvw_casa(tel_xds, time_str, phase_center_ra_dec, antenna1, antenna2)
    #elif _uvw_parms['calc_method'] == 'calc11': #Under development.
        #uvw = _calc_uvw_casa(tel_xds, time_str, phase_center_ra_dec, antenna1, antenna2)
    return uvw, antenna1, antenna2

''' Under development 
def calc_uvw(tel_xds, time_str, phase_center_ra_dec, uvw_parms, check_parms=True):
    """
    Under development.
    
    Parameters
    ----------
    
    Returns
    -------
    """
    _uvw_parms = copy.deepcopy(uvw_parms)
    _save_parms = copy.deepcopy(save_parms)
    assert(_check_uvw_parms(_uvw_parms)), "######### ERROR: calc_uvw uvw_parms checking failed."
 
    n_time = len(time_xda)
    n_ant = tel_xds.dims['ant_name']
    #print(n_time,n_chan,n_ant)
   
    n_time_chunks = time_xda.data.numblocks[0]
                
    uvw_list = _ndim_list((n_time_chunks,1,1))
                
    from ._sirius_utils._array_utils import _calc_n_baseline
    n_baselines = _calc_n_baseline(n_ant,_uvw_parms['auto_corr'])
                
    # Build graph
    for c_time in range(n_time_chunks):
                    #print(c_time,c_chan)
        time_chunk = time_xda.data.partitions[c_time]
        
        sim_chunk = dask.delayed(calc_uvw_chunk)(
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
        uvw_list[c_time][0][0] = da.from_delayed(sim_chunk[1],(len(time_chunk), n_baselines, 3),dtype=np.complex)

        

    uvw = da.block(uvw_list)

        
    if _save_parms['DAG_name_uvw_gen']:
        dask.visualize([vis,uvw],filename=_save_parms['DAG_name_uvw_gen'])
    
    #Create simple xds with simulated vis, uvw, weight and sigma
    uvw_xds = xr.Dataset()
    coords = {'time':time_xda.data,'chan': chan_xda.data, 'pol': pol}
    uvw_xds = vis_xds.assign_coords(coords)
        

    uvw_xds['UVW'] = xr.DataArray(uvw, dims=['time','baseline','uvw'])
    ###################
        
        write_to_ms(vis_xds, time_xda, chan_xda, pol, tel_xds, phase_center_names, phase_center_ra_dec, _uvw_parms['auto_corr'],_save_parms)
        
        return uvw_xds
'''
