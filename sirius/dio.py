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
import xarray as xr
from astropy.timeseries import TimeSeries
from astropy.time import Time
from astropy import units as u
import dask.array as da
from collections import Counter
import time
import dask
import os
 
def make_time_xda(time_start='2019-10-03T19:00:00.000',time_delta=3600,n_samples=10,n_chunks=4):
    """
    Create a time series xarray array.
    Parameters
    ----------
    -------
    time_da : dask.array
    """
    ts = np.array(TimeSeries(time_start=time_start,time_delta=time_delta*u.s,n_samples=n_samples).time.value)
    chunksize = int(np.ceil(n_samples/n_chunks))
    time_da = da.from_array(ts, chunks=chunksize)
    print('Number of chunks ', len(time_da.chunks[0]))
    
    time_xda = xr.DataArray(data=time_da,dims=["time"],attrs={'time_delta':float(time_delta)})
    
    return time_xda

def make_chan_xda(spw_name='sband',freq_start = 3*10**9, freq_delta = 0.4*10**9, freq_resolution=0.01*10**9, n_channels=3, n_chunks=3):
    """
    Create a channel frequencies xarray array.
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
    
    if save_parms['write_to_ms']:
        start = time.time()
        from casatools import simulator
        from casatasks import mstransform
        n_time, n_baseline, n_chan, n_pol = vis_xds.DATA.shape

        sm = simulator()
        
        ant_pos = tel_xds.ANT_POS.values
        os.system('rm -rf ' + save_parms['ms_name'])
        sm.open(ms=save_parms['ms_name']);

        ###########################################################################################################################
        ## Set the antenna configuration
        sm.setconfig(telescopename= tel_xds.telescope_name,
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
            
        mjd = Time(time_xda.data[0:2].compute(), scale='utc')
        integration_time = (mjd[1]-mjd[0]).to('second')

        start_time = (mjd[0] - (integration_time/2 + 37*u.second)).mjd
        start_time_dict = {'m0': {'unit': 'd', 'value': start_time}, 'refer': 'UTC', 'type': 'epoch'}

        sm.settimes(integrationtime=integration_time.value,
                    usehourangle=False,
                    referencetime=start_time_dict);
            
        fields_set = []
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

        print('Meta data creation ',time.time()-start)

        #print(vis_data.shape)
        #print(n_row,n_time, n_baseline, n_chan, n_pol)

        start = time.time()
        #This code will most probably be moved into simulation if we get rid of row time baseline split.
        vis_data_reshaped = vis_xds.DATA.data.reshape((n_row, n_chan, n_pol))
        uvw_reshaped = vis_xds.UVW.data.reshape((n_row, 3))
        weight_reshaped = vis_xds.WEIGHT.data.reshape((n_row,n_pol))
        sigma_reshaped = vis_xds.SIGMA.data.reshape((n_row,n_pol))

        print('reshape time ', time.time()-start)
        #weight_spectrum_reshaped = np.tile(weight_reshaped[:,None,:],(1,n_chan,1))



        #    print(weight_reshaped.compute().shape)
        #    print(sigma_reshaped.compute().shape)
        #    print(weight_reshaped)
        #    print(sigma_reshaped)

        #dask_ddid = da.full(n_row, 0, chunks=chunks['row'], dtype=np.int32)

        #print('vis_data_reshaped',vis_data_reshaped)

        start = time.time()
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

        print('compute and save time ', time.time()-start)

        sm.close()

        from casatasks import flagdata
        flagdata(vis=save_parms['ms_name'],mode='unflag')
        
        
        return xds_from_ms(save_parms['ms_name'])

