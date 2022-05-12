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
import dask.array as da
from astropy.timeseries import TimeSeries
from astropy.time import Time
from astropy import units as u
from collections import Counter
import time
import dask
import os
import shutil
import logging
from sirius.display_tools import print_dict_as_html_table

###### Functions that assist with creation of ant_xds ######

def get_available_ant_models():
    """
    telescope_layout_name
    zernike coefficients
    function
    """
    
    
    ######### Supported Telescopes #######
    import re
    def num_sort(test_string):
        try:
            key = list(map(int, re.findall(r'\d+', test_string)))[0]
        except:
            key = -1
        return key
    import pkg_resources
    import glob
    tel_dir = pkg_resources.resource_filename('sirius_data', 'telescope_layout/data/')
    tel_paths = glob.glob(tel_dir+'/*.tel.zarr')
    telescope_layout = [os.path.basename(x) for x in tel_paths]
    
    supported_telescopes = {'alma':[],'aca':[],'evla':[],'vla':[],'ngvla':[]}

    for tel in telescope_layout:
        for tel_name in supported_telescopes.keys():
            if tel_name == 'vla':
                if (tel_name in tel) and ('ngvla' not in tel) and ('evla' not in tel):
                    supported_telescopes[tel_name].append(tel)
            else:
                if tel_name in tel:
                    supported_telescopes[tel_name].append(tel)
                
    for key in supported_telescopes: supported_telescopes[key].sort(key=num_sort)
    print_dict_as_html_table(supported_telescopes,"Available Telescope Array Layouts")
        
    ####### Available Aperture Polynomial Coefficient Models ########
    apc_dir = pkg_resources.resource_filename('sirius_data', 'aperture_polynomial_coefficient_models/data/')
    apc_paths = glob.glob(apc_dir+'/*.apc.zarr')
    apc_names = [os.path.basename(x) for x in apc_paths]
    
    print(apc_names)




def make_time_xda(
    time_start="2019-10-03T19:00:00.000", time_delta=3600, n_samples=10, n_chunks=4
):
    """
    Create a time series xarray array.
    Parameters
    ----------
    -------
    time_xda : xarray.DataArray
    """
    ts = np.array(
        TimeSeries(
            time_start=time_start, time_delta=time_delta * u.s, n_samples=n_samples
        ).time.value
    )
    chunksize = int(np.ceil(n_samples / n_chunks))
    time_da = da.from_array(ts, chunks=chunksize)
    #print("Number of chunks ", len(time_da.chunks[0]))

    time_xda = xr.DataArray(
        data=time_da, dims=["time"], attrs={"time_delta": float(time_delta)}
    )
    
    time_xda.attrs['time_delta'] = time_delta

    return time_xda


def make_chan_xda(
    spw_name="sband",
    freq_start=3 * 10**9,
    freq_delta=0.4 * 10**9,
    freq_resolution=0.01 * 10**9,
    n_channels=3,
    n_chunks=3,
):
    """
    Create a channel frequencies xarray array.
    Parameters
    ----------
    -------
    chan_xda : xarray.DataArray
    """
    freq_chan = (np.arange(0, n_channels) * freq_delta + freq_start).astype(
        float
    )  # astype(float) needed for interfacing with CASA simulator.
    chunksize = int(np.ceil(n_channels / n_chunks))
    chan_da = da.from_array(freq_chan, chunks=chunksize)
    #print("Number of chunks ", len(chan_da.chunks[0]))

    chan_xda = xr.DataArray(
        data=chan_da,
        dims=["chan"],
        attrs={
            "freq_resolution": float(freq_resolution),
            "spw_name": spw_name,
            "freq_delta": float(freq_delta),
        },
    )
    return chan_xda














#####################move down#####################

from dask.distributed import get_client


def create_mxds(
    xds0,
    time_xda,
    chan_xda,
    pol,
    tel_xds,
    phase_center_names,
    phase_center_ra_dec,
    auto_corr,
    save_parms,
):
    n_row, n_chan, n_pol = xds0['DATA'].shape
    n_time = len(time_xda)
    n_baseline = int(n_row/n_time)
    
    
    assert (n_row/n_time - int(n_row/n_time)) == 0, 'n_time*n_basline does not equal n_row'

    #Create mxds
    mxds = xr.Dataset()
    coords = {'polarization_ids':np.array([0]), 'spw_ids': np.array([0])}
    mxds = mxds.assign_coords(coords)


    ############## Create main table xds ##############
    coords = {'chan':chan_xda.values, 'pol': pol, 'pol_id':np.array([0]), 'spw_id': np.array([0])}
    
    chunks = {"row": (xds0['DATA'].chunks[0][0],), "chan": (xds0['DATA'].chunks[1][0],), "pol": (xds0['DATA'].chunks[2][0],), "uvw": (3,)}
    chunks =  (xds0['DATA'].chunks[0][0], xds0['DATA'].chunks[1][0],xds0['DATA'].chunks[2][0])


    empty_data_column = da.zeros(xds0['DATA'].shape, chunks=chunks, dtype="complex")
    flag_rows = da.zeros(n_row,chunks=chunks[0], dtype=np.bool)
    flags = da.zeros(xds0['DATA'].shape,chunks=chunks, dtype=np.bool) # don't flag any of the data yet
    zero_column = da.zeros(n_row, chunks=chunks[0], dtype="int32")
    exposures = da.full((n_row,), time_xda.time_delta, chunks=chunks[0], dtype="float64") # fill with input in units of the input array, which we

    xds0 = xds0.assign_coords(coords)
    xds0['CORRECTED_DATA'] = xds0['DATA']
    xds0['MODEL_DATA'] = xr.DataArray(empty_data_column, dims=['row','chan','pol'])
    xds0['FLAG'] = xr.DataArray(flags, dims=['row','chan','pol'])
    xds0['FLAG_ROW'] = xr.DataArray(flag_rows, dims=['row'])
    xds0['DATA_DESC_ID'] = xr.DataArray(zero_column, dims=['row'])
    xds0['ARRAY_ID'] = xr.DataArray(zero_column, dims=['row'])
    xds0['FEED1'] = xr.DataArray(zero_column, dims=['row'])
    xds0['FEED2'] = xr.DataArray(zero_column, dims=['row'])
    xds0['OBSERVATION_ID'] = xr.DataArray(zero_column, dims=['row'])
    xds0['PROCESSOR_ID'] = xr.DataArray(zero_column, dims=['row'])
    xds0['SCAN_NUMBER'] = xr.DataArray(zero_column, dims=['row'])
    xds0['STATE_ID'] = xr.DataArray(zero_column, dims=['row'])
    xds0['EXPOSURE'] = xr.DataArray(exposures, dims=['row'])
    xds0['INTERVAL'] = xr.DataArray(exposures, dims=['row'])
    xds0['TIME_CENTROID'] = xds0['TIME']
    
    xds0.attrs['MS_VERSION'] = 2.0
    from sirius_data._ms_column_descriptions_dicts import main_column_description
    xds0.attrs['column_descriptions'] = main_column_description
    xds0.attrs['info'] = {'type': 'Measurement Set', 'subType': 'simulator', 'readme': 'This is a MeasurementSet Table holding measurements from a Telescope\nThis is a MeasurementSet Table holding simulated astronomical observations\n'}
    xds0.attrs['bad_cols'] = ['FLAG_CATEGORY']
    mxds.attrs['xds0'] = xds0
    
    from sirius._sirius_utils._ms_utils import _sub_table_xds
    mxds = _sub_table_xds(mxds,time_xda,chan_xda,pol,tel_xds,phase_center_names,phase_center_ra_dec,auto_corr)
    return mxds
    
def write_to_ms_daskms_and_sim_tool(
    vis_xds,
    time_xda,
    chan_xda,
    pol,
    tel_xds,
    phase_center_names,
    phase_center_ra_dec,
    auto_corr,
    save_parms,
):
    """
    Write out a MeasurementSet to disk using dask-ms

    This first implementation is kept only temporarily, until performance comparisons are completed.
    """

    start = time.time()
    from casatools import simulator
    from casatasks import mstransform

    n_row, n_chan, n_pol = vis_xds.DATA.shape
    n_time = len(time_xda.data)
    
    print(n_row, n_chan, n_pol)
    print(vis_xds)

    sm = simulator()

    ant_pos = tel_xds.ANT_POS.values
    os.system("rm -rf " + save_parms["ms_name"])
    sm.open(ms=save_parms["ms_name"])

    ###########################################################################################################################
    ## Set the antenna configuration
    sm.setconfig(
        telescopename=tel_xds.telescope_name,
        x=ant_pos[:, 0],
        y=ant_pos[:, 1],
        z=ant_pos[:, 2],
        dishdiameter=tel_xds.DISH_DIAMETER.values,
        mount=["alt-az"],
        antname=list(
            tel_xds.ant_name.values
        ),  # CASA can't handle an array of antenna names.
        coordsystem="global",
        referencelocation=tel_xds.site_pos[0],
    )

    ## Set the polarization mode (this goes to the FEED subtable)
    from sirius_data._constants import pol_codes_RL, pol_codes_XY, pol_str
    from sirius._sirius_utils._array_utils import _is_subset

    if _is_subset(pol_codes_RL, pol):  # ['RR','RL','LR','LL']
        sm.setfeed(mode="perfect R L", pol=[""])
    elif _is_subset(pol_codes_XY, pol):  # ['XX','XY','YX','YY']
        sm.setfeed(mode="perfect X Y", pol=[""])
    else:
        assert False, print(
            "Pol selection invalid, must either be subset of [5,6,7,8] or [9,10,11,12] but is ",
            pol,
        )

    sm.setspwindow(
        spwname=chan_xda.spw_name,
        freq=chan_xda.data[0].compute(),
        deltafreq=chan_xda.freq_delta,
        freqresolution=chan_xda.freq_resolution,
        nchannels=len(chan_xda),
        refcode="LSRK",
        stokes=" ".join(pol_str[pol]),
    )

    if auto_corr:
        sm.setauto(autocorrwt=1.0)
    else:
        sm.setauto(autocorrwt=0.0)

    mjd = Time(time_xda.data[0].compute(), scale="utc")
     
    integration_time = time_xda.attrs['time_delta']* u.second
    
    print('mjd',mjd)

    start_time = (mjd - (integration_time / 2 + 37 * u.second)).mjd
    start_time_dict = {
        "m0": {"unit": "d", "value": start_time},
        "refer": "UTC",
        "type": "epoch",
    }

    sm.settimes(
        integrationtime=integration_time.value,
        usehourangle=False,
        referencetime=start_time_dict,
    )

    fields_set = []
    field_time_count = Counter(phase_center_names)

    # print(field_time_count,phase_center_names)
    if len(phase_center_names) == 1:  # Single field case
        field_time_count[list(field_time_count.keys())[0]] = n_time

    start_time = 0
    for i, ra_dec in enumerate(
        phase_center_ra_dec
    ):  # In future make phase_center_ra_dec a unique list
        if phase_center_names[i] not in fields_set:
            dir_dict = {
                "m0": {"unit": "rad", "value": ra_dec[0]},
                "m1": {"unit": "rad", "value": ra_dec[1]},
                "refer": "J2000",
                "type": "direction",
            }
            sm.setfield(sourcename=phase_center_names[i], sourcedirection=dir_dict)
            fields_set.append(phase_center_names[i])

            stop_time = (
                start_time
                + integration_time.value * field_time_count[phase_center_names[i]]
            )
            sm.observe(
                sourcename=phase_center_names[i],
                spwname=chan_xda.spw_name,
                starttime=str(start_time) + "s",
                stoptime=str(stop_time) + "s",
            )
            start_time = stop_time

    print("Meta data creation ", time.time() - start)

    # print(vis_data.shape)
    # print(n_row,n_time, n_baseline, n_chan, n_pol)

    start = time.time()
    # This code will most probably be moved into simulation if we get rid of row time baseline split.
    #vis_data_reshaped = vis_xds.DATA.data.reshape((n_row, n_chan, n_pol))
    #uvw_reshaped = vis_xds.UVW.data.reshape((n_row, 3))
    #weight_reshaped = vis_xds.WEIGHT.data.reshape((n_row, n_pol))
    #sigma_reshaped = vis_xds.SIGMA.data.reshape((n_row, n_pol))

    #print("reshape time ", time.time() - start)
    # weight_spectrum_reshaped = np.tile(weight_reshaped[:,None,:],(1,n_chan,1))

    #    print(weight_reshaped.compute().shape)
    #    print(sigma_reshaped.compute().shape)
    #    print(weight_reshaped)
    #    print(sigma_reshaped)

    # dask_ddid = da.full(n_row, 0, chunks=chunks['row'], dtype=np.int32)

    # print('vis_data_reshaped',vis_data_reshaped)

    start = time.time()
    from daskms import xds_to_table, xds_from_ms, Dataset

    # print('vis_data_reshaped.chunks',vis_data_reshaped.chunks)
    row_id = da.arange(n_row, chunks=vis_xds.DATA.chunks[0], dtype="int32")

    dataset = Dataset(
        {
            "DATA": (("row", "chan", "corr"), vis_xds.DATA.data),
            "CORRECTED_DATA": (("row", "chan", "corr"), vis_xds.DATA.data),
            "UVW": (("row", "uvw"), vis_xds.UVW.data),
            "SIGMA": (("row", "pol"), vis_xds.SIGMA.data),
            "WEIGHT": (("row", "pol"), vis_xds.WEIGHT.data),
            "ROWID": (("row",), row_id),
        }
    )
    # ,'WEIGHT_SPECTRUM': (("row","chan","pol"), weight_spectrum_reshaped)
    ms_writes = xds_to_table(dataset, save_parms["ms_name"], columns="ALL")


    if save_parms["DAG_name_write"]:
        dask.visualize(ms_writes, filename=save_parms["DAG_name_write"])

    start = time.time()
    dask.compute(ms_writes)
    print("*** Dask compute time", time.time() - start)

    sm.close()

    from casatasks import flagdata

    flagdata(vis=save_parms["ms_name"], mode="unflag")

    
        
def read_zarr(
    infile,
    sel_xds=None,
    chunks=None,
    consolidated=True,
    overwrite_encoded_chunks=True,
    **kwargs,
):
    """
    Read zarr format Visibility data from disk to xarray Dataset

    Parameters
    ----------
    infile : str
        input Visibility filename
    sel_xds : string or list
        Select the ddi to open, for example ['xds0','xds1'] will open the first two ddi. Default None returns everything
    chunks : dict
        sets specified chunk size per dimension. Dict is in the form of 'dim':chunk_size, for example {'time':100, 'baseline':400, 'chan':32, 'pol':1}.
        Default None uses the original zarr chunking.
    consolidated : bool
        use zarr consolidated metadata capability. Only works for stores that have already been consolidated. Default True works with datasets
        produced by convert_ms which automatically consolidates metadata.
    overwrite_encoded_chunks : bool
        drop the zarr chunks encoded for each variable when a dataset is loaded with specified chunk sizes.  Default True, only applies when chunks
        is not None.
        
    Returns
    -------
    xarray.core.dataset.Dataset
        New xarray Dataset of Visibility data contents
    """
    import os
    import numpy as np
    import cngi._utils._io as xdsio
    from xarray import open_zarr

    if chunks is None:
        chunks = "auto"
        #overwrite_encoded_chunks = False
    #print('overwrite_encoded_chunks',overwrite_encoded_chunks)

    infile = os.path.expanduser(infile)
    if sel_xds is None:
        sel_xds = os.listdir(infile)
    sel_xds = list(np.atleast_1d(sel_xds))
    
    
    #print(os.path.join(infile, 'DDI_INDEX'))
    mxds = open_zarr(os.path.join(infile, 'DDI_INDEX'), chunks=chunks,consolidated=consolidated,overwrite_encoded_chunks=overwrite_encoded_chunks)

    for part in os.listdir(os.path.join(infile, "global")):
        xds_temp = open_zarr(os.path.join(infile, 'global/'+part), chunks=chunks,
                                                                     consolidated=consolidated,
                                                                     overwrite_encoded_chunks=overwrite_encoded_chunks)
        xds_temp = _fix_dict_for_ms(part,xds_temp)
        mxds.attrs[part] = xds_temp.compute()

    for part in os.listdir(infile):
        if ('xds' in part) and (part in sel_xds):
            xds_temp = open_zarr(os.path.join(infile, part), chunks=chunks,
                                                                     consolidated=consolidated,
                                                                     overwrite_encoded_chunks=overwrite_encoded_chunks)
            xds_temp = _fix_dict_for_ms(part,xds_temp)
            mxds.attrs[part] = xds_temp
            
    return mxds
 
    
def write_zarr(mxds, outfile, chunks_on_disk=None, partition=None, consolidated=True, compressor=None, overwrite=True, graph_name='write_zarr'):
    """
    Write xarray dataset to zarr format on disk. When chunks_on_disk is not specified the chunking in the input dataset is used.
    When chunks_on_disk is specified that dataset is saved using that chunking.

    Parameters
    ----------
    mxds : xarray.core.dataset.Dataset
        Dataset of dataset to write to disk
    outfile : str
        outfile filename, generally ends in .zarr
    chunks_on_disk : dict of int
        A dictionary with the chunk size that will be used when writing to disk. For example {'time': 20, 'chan': 6}.
        If chunks_on_disk is not specified the chunking of dataset will be used.
    partition : str or list
        Name of partition xds to write into outfile (from the mxds attributes section). Overwrites existing partition of same name.
        Default None writes entire mxds
    compressor : numcodecs.blosc.Blosc
        The blosc compressor to use when saving the converted data to disk using zarr.
        If None the zstd compression algorithm used with compression level 2.
    graph_name : string
        The time taken to execute the graph and save the dataset is measured and saved as an attribute in the zarr file.
        The graph_name is the label for this timing information.

    Returns
    -------
    """
    import xarray as xr
    import zarr
    import time
    from numcodecs import Blosc
    from itertools import cycle
    import os
    import numpy as np

    if compressor is None:
        compressor = Blosc(cname='zstd', clevel=2, shuffle=0)

    if partition is None:
        partition = list(mxds.attrs.keys())
    partition = list(np.atleast_1d(partition))
        
    if overwrite:
        try:
            os.remove(outfile)
        except IsADirectoryError:
            shutil.rmtree(outfile)
        except FileNotFoundError:
            pass
        os.system('mkdir ' + outfile)
    else:
        assert not os.path.isfile(outfile), 'vis.zarr folder already exists. Set overwrite to True.'
        
    ddi_indx_xds = xr.Dataset()
    ddi_indx_xds['polarization_ids'] = mxds['polarization_ids']
    ddi_indx_xds['spw_ids'] = mxds['spw_ids']
    encoding = dict(zip(list(ddi_indx_xds.data_vars), cycle([{'compressor': compressor}])))
    xr.Dataset.to_zarr(ddi_indx_xds, store=outfile+'/DDI_INDEX', mode='w', encoding=encoding,consolidated=consolidated)
        
    for xds_name in partition:
        if "xds" in xds_name:
            xds_outfile = outfile + '/' + xds_name
            xds_for_disk = mxds.attrs[xds_name]
            if chunks_on_disk is not None:
                xds_for_disk = xds_for_disk.chunk(chunks=chunks_on_disk)
        else:
            xds_outfile = outfile + '/global/' + xds_name
            xds_for_disk = mxds.attrs[xds_name]
            
        xds_for_disk = _fix_dict_for_zarr(xds_name, xds_for_disk)
            
        # Create compression encoding for each datavariable
        encoding = dict(zip(list(xds_for_disk.data_vars), cycle([{'compressor': compressor}])))
        start = time.time()

        # Consolidated is set to False so that the timing information is included in the consolidate metadata.
        xr.Dataset.to_zarr(xds_for_disk, store=xds_outfile, mode='w', encoding=encoding,consolidated=False)
        time_to_calc_and_store = time.time() - start
        print('Time to store and execute graph for ', xds_name, graph_name, time_to_calc_and_store)
      
        #Add timing information
        #dataset_group = zarr.open_group(xds_outfile, mode='a')
        #dataset_group.attrs[graph_name+'_time'] = time_to_calc_and_store
            
        if consolidated == True:
            zarr.consolidate_metadata(xds_outfile)


def _fix_dict_for_ms(name, xds):
    xds.attrs['column_descriptions'] = xds.attrs['column_descriptions'][0]
    xds.attrs['info'] = xds.attrs['info'][0]

    if "xds" in name:
        xds.column_descriptions['UVW']['shape'] = np.array(xds.column_descriptions['UVW']['shape'].split(',')).astype(int)

    if "SPECTRAL_WINDOW" == name:
        #print('2.',xds.column_descriptions)
        xds.column_descriptions['CHAN_FREQ']['keywords']['MEASINFO']['TabRefCodes'] = np.array(xds.column_descriptions['CHAN_FREQ']['keywords']['MEASINFO']['TabRefCodes'].split(',')).astype(int)
        xds.column_descriptions['REF_FREQUENCY']['keywords']['MEASINFO']['TabRefCodes'] =  np.array(xds.column_descriptions['REF_FREQUENCY']['keywords']['MEASINFO']['TabRefCodes'].split(',')).astype(int)
        
    if "ANTENNA" == name:
        xds.column_descriptions['OFFSET']['shape'] = np.array(xds.column_descriptions['OFFSET']['shape'].split(',')).astype(int)
        xds.column_descriptions['POSITION']['shape'] = np.array(xds.column_descriptions['POSITION']['shape'].split(',')).astype(int)
    
    if "FEED" == name:
        xds.column_descriptions['POSITION']['shape'] = np.array(xds.column_descriptions['POSITION']['shape'].split(',')).astype(int)

    if "OBSERVATION" == name:
        xds.column_descriptions['TIME_RANGE']['shape'] = np.array(xds.column_descriptions['TIME_RANGE']['shape'].split(',')).astype(int)

    return xds
    
def _fix_dict_for_zarr(name, xds):
    xds.attrs['column_descriptions'] = [xds.attrs['column_descriptions']]
    xds.attrs['info'] = [xds.attrs['info']]
    
    if "xds" in name:
        xds.column_descriptions[0]['UVW']['shape'] = ','.join(map(str, xds.column_descriptions[0]['UVW']['shape']))

    if "SPECTRAL_WINDOW" == name:
        #print('1.',xds.column_descriptions)
        xds.column_descriptions[0]['CHAN_FREQ']['keywords']['MEASINFO']['TabRefCodes'] = ','.join(map(str, xds.column_descriptions[0]['CHAN_FREQ']['keywords']['MEASINFO']['TabRefCodes']))
        xds.column_descriptions[0]['REF_FREQUENCY']['keywords']['MEASINFO']['TabRefCodes'] = ','.join(map(str, xds.column_descriptions[0]['REF_FREQUENCY']['keywords']['MEASINFO']['TabRefCodes']))
    
    if "ANTENNA" == name:
        xds.column_descriptions[0]['OFFSET']['shape'] = ','.join(map(str, xds.column_descriptions[0]['OFFSET']['shape']))
        xds.column_descriptions[0]['POSITION']['shape'] = ','.join(map(str, xds.column_descriptions[0]['POSITION']['shape']))
    
    if "FEED" == name:
        xds.column_descriptions[0]['POSITION']['shape'] = ','.join(map(str, xds.column_descriptions[0]['POSITION']['shape']))

    if "OBSERVATION" == name:
        xds.column_descriptions[0]['TIME_RANGE']['shape'] = ','.join(map(str, xds.column_descriptions[0]['TIME_RANGE']['shape']))
        
    return xds












################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################

    '''
    min_memory_for_a_thread, _, _ = _get_schedular_info()
    #chunksize_row = _calc_optimal_ms_chunk_shape(min_memory_for_a_thread,(n_row,),64,'TIME')
    chunksize_row = _calc_optimal_ms_chunk_shape(min_memory_for_a_thread,(n_row,),128,'MODEL_DATA')
    empty_data_column = da.zeros(xds0['DATA'].shape, chunks=(chunksize_row,-1,-1), dtype="complex")
    
    chunksize_row = _calc_optimal_ms_chunk_shape(min_memory_for_a_thread,(n_row,),8,'FLAG_ROW')
    flag_rows = da.zeros(n_row,chunks=chunksize_row, dtype=np.bool)

    chunksize_row = _calc_optimal_ms_chunk_shape(min_memory_for_a_thread,xds0['DATA'].shape,8,'FLAG')
    flags = da.zeros(xds0['DATA'].shape,chunks=(chunksize_row,-1,-1), dtype=np.bool) # don't flag any of the data yet

   
#    ddid = da.zeros(n_row, chunks=chunksize_row, dtype="int32") # we run this function on only a single DDI at a time
#    array_ids = da.zeros_like(ddid) # currently don't support subarrays, so only one array ID assigned
#    feeds = da.zeros_like(ddid)  # not supporting different feed types
#    observation_ids = da.zeros_like(ddid) # this function is also only run for a single observation at once
#    processor_ids = da.zeros_like(ddid) # currently don't support group processing
#    scan_numbers = da.ones_like(ddid)
#    state_ids = da.zeros_like(ddid)
    
    chunksize_row = _calc_optimal_ms_chunk_shape(min_memory_for_a_thread,(n_row,),32,'ZERO')
    zero_column = da.zeros(n_row, chunks=chunksize_row, dtype="int32")

    chunksize_row = _calc_optimal_ms_chunk_shape(min_memory_for_a_thread,(n_row,),64,'EXPOSURE/INTERVAL')
    exposures = da.full((n_row,), time_xda.time_delta, chunks=chunksize_row, dtype="float64") # fill with input in units of the input array, which we
    '''


