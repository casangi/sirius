#  CASA Next Generation Infrastructure
#  Copyright (C) 2021 AUI, Inc. Washington DC, USA
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

#################################
# Helper File
#
# Not exposed in API
#
#################################
import warnings, time, os, psutil, multiprocessing, logging, re
import numpy as np
# from casatools import table as tb
from casatools import ms
from casatools import image as ia
from casatools import quanta as qa
from scipy.optimize import minimize

try:
    import pandas as pd
    import xarray, dask, dask.array, dask.delayed, dask.distributed
except:
    print('#### ERROR - dask and/or xarray dependencies are missing ####')

try:
    from casacore import tables
except:
    print('#### ERROR - python-casacore not found, must be manually installed by user ####')

warnings.filterwarnings('ignore', category=FutureWarning)


# TODO: python-casacore dependency is needed here
# Problems with the table tool:
#     - inflates data sizes by reading everything as 64-bit float / 128-bit complex,
#     - segfaults when used in dask delayed objects with non-locking reads
#     - row access not available, segfaults on column access for some test data


########################################################
# local helper - initialize the processing environment
def initialize_processing(cores=None, memory_limit=None):
    # setup dask.distributed based multiprocessing environment
    if cores is None: cores = multiprocessing.cpu_count()
    if memory_limit is None: memory_limit = str(round(((psutil.virtual_memory().available / (1024 ** 2)) * 0.75) / cores)) + 'MB'
    dask.config.set({"distributed.scheduler.allowed-failures": 10})
    dask.config.set({"distributed.scheduler.work-stealing": False})
    dask.config.set({"distributed.scheduler.unknown-task-duration": '99m'})
    dask.config.set({"distributed.worker.memory.pause": False})
    dask.config.set({"distributed.worker.memory.terminate": False})
    dask.config.set({"distributed.worker.memory.recent-to-old-time": '999s'})
    dask.config.set({"distributed.comm.timeouts.connect": '360s'})
    dask.config.set({"distributed.comm.timeouts.tcp": '360s'})
    dask.config.set({"distributed.nanny.environ.OMP_NUM_THREADS": 1})
    dask.config.set({"distributed.nanny.environ.MKL_NUM_THREADS": 1})
    cluster = dask.distributed.LocalCluster(n_workers=cores, threads_per_worker=1, processes=True, memory_limit=memory_limit, silence_logs=logging.ERROR)
    client = dask.distributed.Client(cluster)
    return client


########################################################
# local helper - read time columns to datetime format
# pandas datetimes are referenced against a 0 of 1970-01-01
# CASA's modified julian day reference time is (of course) 1858-11-17
# this requires a correction of 3506716800 seconds which is hardcoded to save time
def convert_time(rawtimes):
    correction = 3506716800.0
    return pd.to_datetime(np.array(rawtimes) - correction, unit='s').values
    # dt = pd.to_datetime(np.atleast_1d(rawtimes) - correction, unit='s').values
    # if len(np.array(rawtimes).shape) == 0: dt = dt[0]
    # return dt

def revert_time(datetimes):
    return (datetimes.astype(float) / 10 ** 9) + 3506716800.0


####################################
# local helper - return a dictionary of table attributes created from keywords and column descriptions
def extract_table_attributes(infile):
    tb_tool = tables.table(infile, readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    kwd = tb_tool.getkeywords()
    attrs = dict([(kk, kwd[kk]) for kk in kwd if kk not in os.listdir(infile)])
    cols = tb_tool.colnames()
    column_descriptions = {}
    for col in cols:
        column_descriptions[col] = tb_tool.getcoldesc(col)
    attrs['column_descriptions'] = column_descriptions
    attrs['info'] = tb_tool.info()
    tb_tool.close()
    return attrs


#####################################
# local helper - translate numpy dtypes to casacore type strings
def type_converter(npdtype):
    cctype = 'bad'
    if (npdtype == 'int64') or (npdtype == 'int32'):
        cctype = 'int'
    elif npdtype == 'bool':
        cctype = 'bool'
    elif npdtype == 'float32':
        cctype = 'float'
    elif (npdtype == 'float64') or (npdtype == 'datetime64[ns]'):
        cctype = 'double'
    elif npdtype == 'complex64':
        cctype = 'complex'
    elif npdtype == 'complex128':
        cctype = 'dcomplex'
    elif str(npdtype).startswith('<U'):
        cctype = 'string'

    return cctype


####################################
# local helper - create and initialize new output table
def create_table(outfile, xds, max_rows, infile=None, cols=None, generic=False):
    if os.path.isdir(outfile):
        os.system('rm -fr %s' % outfile)

    # create column descriptions for table description
    if cols is None: cols = list(set(list(xds.data_vars) + list(xds.attrs['column_descriptions'].keys())) if 'column_descriptions' in xds.attrs else list(xds.data_vars))
    tabledesc = {}
    for col in cols:
        if ('column_descriptions' in xds.attrs) and (col in xds.attrs['column_descriptions']):
            coldesc = xds.attrs['column_descriptions'][col]
        else:
            coldesc = {'valueType': type_converter(xds[col].dtype)}
            if generic or (col == 'UVW'):  # will be statically shaped even if not originally
                coldesc = {'shape': tuple(np.clip(xds[col].shape[1:], 1, None))}
            elif xds[col].ndim > 1:  # make variably shaped
                coldesc = {'ndim': xds[col].ndim - 1}
            coldesc['name'] = col
            coldesc['desc'] = col
        tabledesc[col] = coldesc

        # fix the fun set of edge cases from casatestdata that cause errors
        if (tabledesc[col]['dataManagerType'] == 'TiledShapeStMan') and (tabledesc[col]['ndim'] == 1):
            tabledesc[col]['dataManagerType'] = ''

    if generic:
        tb_tool = tables.table(outfile, tabledesc=tabledesc, nrow=max_rows, readonly=False, lockoptions={'option': 'permanentwait'}, ack=False)
    else:
        tb_tool = tables.default_ms(outfile, tabledesc)
        tb_tool.addrows(max_rows)
        #if 'DATA_DESC_ID' in cols: tb_tool.putcol('DATA_DESC_ID', np.zeros((max_rows), dtype='int32') - 1, 0, max_rows)

    # write xds attributes to table keywords, skipping certain reserved attributes
    existing_keywords = tb_tool.getkeywords()
    for attr in xds.attrs:
        if attr in ['bad_cols', 'bad_types', 'column_descriptions', 'history', 'subtables', 'info'] + list(existing_keywords.keys()): continue
        tb_tool.putkeyword(attr, xds.attrs[attr])
    if 'info' in xds.attrs: tb_tool.putinfo(xds.attrs['info'])

    # copy subtables and add to main table
    if infile:
        subtables = [ss.path for ss in os.scandir(infile) if ss.is_dir() and ('SORTED_TABLE' not in ss.path)]
        os.system('cp -r %s %s' % (' '.join(subtables), outfile))
        for subtable in subtables:
            if not tables.tableexists(os.path.join(outfile, subtable[subtable.rindex('/') + 1:])): continue
            sub_tbl = tables.table(os.path.join(outfile, subtable[subtable.rindex('/') + 1:]), readonly=False, lockoptions={'option': 'permanentwait'}, ack=False)
            tb_tool.putkeyword(subtable[subtable.rindex('/') + 1:], sub_tbl, makesubrecord=True)
            sub_tbl.close()

    tb_tool.close()


####################################
# local helper - automatically compute best data chunking
def optimal_chunking(ndim=None, didxs=None, chunk_size='auto', data_shape=None):
    """
    determine the optimal chunk shape for reading an MS or Image based on machine resources and intended operations

    Parameters
    ----------
    ndim : int
        number of dimensions to chunk. An MS is 3, an expanded MS is 4. An image could be anywhere from 2 to 5. Not needed if data_shape is given.
    didxs : int or list of ints
        dimension indices over which subsequent operations will be performed. Values should be less than ndim. Tries to reduce inter-process communication
        of data contents. Needs to know the shape to do this well. Default None balances chunk size across all dimensions.
    chunk_size : str
        target chunk size ('large', 'small', 'auto'). Default 'auto' tries to guess by looking at CPU core count and available memory.
    data_shape : tuple
        shape of the total MS DDI or Image data. Helps to know. Default None does not optimize based on shape

    Returns
    -------
    tuple of ints
      optimal chunking for reading the ms (row, chan, pol)
    """
    assert (ndim is not None) or (data_shape is not None), "either ndim or data_shape must be given"
    assert chunk_size in ['large', 'small', 'auto'], "invalid chunk_size parameter"
    if ndim is None: ndim = len(data_shape)

    opt_dims = didxs if (didxs is not None) and (len(didxs) > 0) else np.arange(ndim)  # maximize these dim chunk sizes
    nonopt_dims = np.setdiff1d(np.arange(ndim), opt_dims)       # at the expense of these

    max_chunk_sizes = data_shape if data_shape is not None else [dd for ii, dd in enumerate([10000, 10000, 10000, 4, 10]) if ii < ndim]
    min_chunk_sizes = np.ceil(np.array(data_shape)/80).astype(int) if data_shape is not None else [1000,1,1] if ndim==3 else [dd for ii, dd in enumerate([10,10,1,1,1]) if ii<ndim]
    target_size = 175 * 1024 ** 2 / 8  # ~175 MB chunk worst case with 8-byte DATA column
    bytes_per_core = int(round(((psutil.virtual_memory().available * 0.10) / multiprocessing.cpu_count())))
    if data_shape is not None: bytes_per_core = min(bytes_per_core, np.prod(data_shape)*8/2)  # ensure at least two chunks
    if chunk_size == 'large':
        target_size = target_size * 6  # ~1 GB
    if chunk_size == 'auto':
        target_size = max(min(target_size * 6, bytes_per_core / 8), target_size)

    # start by setting the optimized dims to their max size and non-optimized dims to their min size
    chunks = np.zeros((ndim), dtype='int')
    chunks[opt_dims] = np.array(max_chunk_sizes)[opt_dims]
    chunks[nonopt_dims] = np.array(min_chunk_sizes)[nonopt_dims]

    # iteratively walk towards an optimal chunk size
    # iteration is needed because rounding to nearest integer index can make a big different (2x) in chunk size
    # for small dimensions like pol
    for ii in range(10):
        # if the resulting size is too big, reduce the sizes of the optimized dimensions
        if (np.prod(chunks) > target_size) and (len(opt_dims) > 0):
            chunks[opt_dims] = np.round(chunks[opt_dims] * (target_size/np.prod(chunks))**(1/len(opt_dims)))
        # else if the resulting size is too small, increase the sizes of the non-optimized dimensions
        elif (np.prod(chunks) < target_size) and (len(nonopt_dims) > 0):
            chunks[nonopt_dims] = np.round(chunks[nonopt_dims] * (target_size/np.prod(chunks))**(1/len(nonopt_dims)))
        chunks = np.min((chunks, max_chunk_sizes), axis=0)
        chunks = np.max((chunks, min_chunk_sizes), axis=0)

    return tuple(chunks)



##################################################################################################
##
## MeasurementSets
##
##################################################################################################


############################################
# local helper - takes a list of visibility xarray datasets and packages them as a dataset of datasets
def vis_xds_packager(xds_list):
    mxds = xarray.Dataset(attrs=dict(xds_list))

    coords = {}
    if 'ANTENNA' in mxds.attrs:
        coords['antenna_ids'] = mxds.ANTENNA.row.values
        coords['antennas'] = xarray.DataArray(mxds.ANTENNA.NAME.values, dims=['antenna_ids'])
    if 'FIELD' in mxds.attrs:
        coords['field_ids'] = mxds.FIELD.row.values
        coords['fields'] = xarray.DataArray(mxds.FIELD.NAME.values, dims=['field_ids'])
    if 'FEED' in mxds.attrs:
        coords['feed_ids'] = mxds.FEED.FEED_ID.values
    if 'OBSERVATION' in mxds.attrs:
        coords['observation_ids'] = mxds.OBSERVATION.row.values
        coords['observations'] = xarray.DataArray(mxds.OBSERVATION.PROJECT.values, dims=['observation_ids'])
    if 'POLARIZATION' in mxds.attrs:
        coords['polarization_ids'] = mxds.POLARIZATION.row.values
    if 'SOURCE' in mxds.attrs:
        coords['source_ids'] = mxds.SOURCE.SOURCE_ID.values
        coords['sources'] = xarray.DataArray(mxds.SOURCE.NAME.values, dims=['source_ids'])
    if 'SPECTRAL_WINDOW' in mxds.attrs:
        coords['spw_ids'] = mxds.SPECTRAL_WINDOW.row.values
    if 'STATE' in mxds.attrs:
        coords['state_ids'] = mxds.STATE.row.values

    mxds = mxds.assign_coords(coords)

    return mxds


#################################
## local helper - transform casa MS selection parameters into dictionary of ddi:(rows, chans)
def ms_selection(infile, verbose=False, **kwargs):
    """
    translates MS selection parameters into corresponding row indices and channel indices

    Parameters
    ----------
    infile : str
        Input MS filename
    verbose : bool
        What you would expect.
    **kwargs : str
        Selection parameters from the standard way of making CASA MS selections.
        Must be in ['spw', 'field', 'scan', 'baseline', 'time', 'scanintent', 'uvdist', 'polarization', 'array', 'observation']

    Returns
    -------
    dict
      returns a dictionary with keys of DDI numbers and values of (rowidxs, chanidxs)
    """
    assert np.all([kk in ['spw', 'field', 'scan', 'baseline', 'time', 'scanintent', 'uvdist', 'polarization', 'array', 'observation'] for kk in kwargs.keys()]), 'invalid selection parameter'
    infile = os.path.expanduser(infile)
    mstool = ms()
    mstool.open(infile)
    start = time.time()

    # build the selection structure
    selection = dict([(kk, vv) for kk, vv in kwargs.items() if vv is not None])

    # build structure of indices per DDI, intersected with selection criteria
    ddis, total_rows = [], None
    chanmap = {}  # dict of ddis to channels
    if len(selection) > 0:
        if verbose: print('selecting data %s...' % str(selection))
        mstool.msselect(selection)
        total_rows = mstool.range('rows')['rows']
        selectedindices = mstool.msselectedindices()
        ddis, chanranges = selectedindices['dd'], selectedindices['channel']
        for ci, cr in enumerate(chanranges):
            if ddis[ci] not in chanmap: chanmap[ddis[ci]] = []
            chanmap[ddis[ci]] = np.concatenate((chanmap[ddis[ci]], list(range(cr[1], cr[2] + 1, cr[3]))), axis=0).astype(int)

    mstool.reset()
    if len(ddis) == 0:  # selection didn't reduce ddi count, so get them all
        ddis = list(mstool.range('data_desc_id')['data_desc_id'])

    # figure out which selected rows are in which ddis
    if verbose: print('intersecting DDI row ids...')
    rowmap = {}  # dict of ddis to (rows, channels)
    for ddi in ddis:
        mstool.selectinit(datadescid=ddi)
        ddirowidxs = mstool.range('rows')['rows']
        if total_rows is None:
            rowmap[ddi] = (ddirowidxs, chanmap[ddi] if ddi in chanmap else None)
        else:
            rowmap[ddi] = (np.intersect1d(ddirowidxs, total_rows, assume_unique=True), chanmap[ddi] if ddi in chanmap else None)
        mstool.reset()

    mstool.close()
    if verbose: print('selection complete in %0.2f s' % (time.time() - start))
    return rowmap


################################
## local helper - expand row dimension of xds to (time, baseline)
def expand_xds(xds):
    txds = xds.copy()
    unique_baselines, baselines = np.unique([txds.ANTENNA1.values, txds.ANTENNA2.values], axis=1, return_inverse=True)
    txds['baseline'] = xarray.DataArray(baselines.astype('int32'), dims=['row'])
    txds['time'] = txds['TIME'].copy()
    try:
        txds = txds.set_index(row=['time', 'baseline']).unstack('row').transpose('time', 'baseline', ...)

        # unstack makes everything a float, so we need to reset to the proper type
        for dv in txds.data_vars:
            txds[dv] = txds[dv].astype(xds[dv].dtype)
    except:
        print("WARNING: Cannot expand rows to (time, baseline), possibly duplicate values in (time, baseline)")
        txds = xds.copy()

    return txds

###############################
## local helper - flatten (time, baseline) dimensions of xds back to single row
def flatten_xds(xds):
    nan_int = np.array([np.nan]).astype('int32')[0]
    txds = xds.copy()

    # flatten the time x baseline dimensions of main table
    if ('time' in xds.dims) and ('baseline' in xds.dims):
        txds = xds.stack({'row': ('time', 'baseline')}).transpose('row', ...)
        txds = txds.where((txds.STATE_ID != nan_int) & (txds.FIELD_ID != nan_int), drop=True) #.unify_chunks()
        for dv in list(xds.data_vars):
            txds[dv] = txds[dv].astype(xds[dv].dtype)

    return txds


###############################
## local helper
def describe_ms(infile, mode='summary', rowmap=None, verbose=False):
    """
    Summarize the contents of an MS directory in casacore table format

    Parameters
    ----------
    infile : str
        input MS filename
    mode : str
        type of information returned ('summary', 'flat', 'expanded'). 'summary' returns a pandas dataframe
        that is nice for displaying in notebooks etc. 'flat' returns a list of tuples of (ddi, row, chan, pol).
        'expanded' returns a list of tuples of (ddi, time, baseline, chan, pol). These latter two are good for
        trying to determine chunk size for read_ms(expand=True/False).
    rowmap : dict
        Dictionary of DDI to tuple of (row indices, channel indices). Returned by ms_selection function. Default None ignores selections

    Returns
    -------
    pandas.dataframe
    """
    infile = os.path.expanduser(infile)  # does nothing if $HOME is unknown
    assert os.path.isdir(infile), "invalid input filename to describe_ms"
    assert mode in ['summary', 'flat', 'expanded'], "invalid mode, must be summary, flat or expanded"

    # figure out characteristics of main table from select subtables (must all be present)
    spw_xds = read_generic_table(os.path.join(infile, 'SPECTRAL_WINDOW'))
    pol_xds = read_generic_table(os.path.join(infile, 'POLARIZATION'))
    ddi_xds = read_generic_table(os.path.join(infile, 'DATA_DESCRIPTION'))
    ddis = list(ddi_xds.row.values) if rowmap is None else list(rowmap.keys())

    spw_ids = ddi_xds.SPECTRAL_WINDOW_ID.values
    pol_ids = ddi_xds.POLARIZATION_ID.values
    chans = spw_xds.NUM_CHAN.values
    pols = pol_xds.NUM_CORR.values

    summary = []
    if mode == 'summary': summary = pd.DataFrame([])

    for ddi in ddis:
        if verbose: print('processing ddi %i of %i' % (ddi + 1, len(ddis)), end='\r')
        tb_tool = tables.table(infile, readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
        table_tool = tables.taql('select * from $tb_tool where DATA_DESC_ID = %i' % ddi)
        sdf = {'ddi': ddi, 'spw_id': spw_ids[ddi], 'pol_id': pol_ids[ddi], 'rows': table_tool.nrows()}
        if mode in ['expanded', 'summary']:
            times = table_tool.getcol('TIME') if rowmap is None else read_flat_col_chunk(infile, 'TIME', (1,), rowmap[ddi][0], 0, 0)
            baselines = [table_tool.getcol(rr)[:, None] if rowmap is None else read_flat_col_chunk(infile, rr, (1,), rowmap[ddi][0], 0, 0) for rr in ['ANTENNA1', 'ANTENNA2']]
            sdf.update({'times': len(np.unique(times)),
                        'baselines': len(np.unique(np.hstack(baselines), axis=0))})
        sdf.update({'chans': chans[spw_ids[ddi]] if (rowmap is None) or (rowmap[ddi][1] is None) else len(rowmap[ddi][1]), 'pols': pols[pol_ids[ddi]]})
        sdf['size_MB'] = np.ceil((sdf['rows'] * sdf['chans'] * sdf['pols'] * 10) / 1024 ** 2).astype(int)
        if rowmap is not None: sdf['rows'] = len(rowmap[ddi][0])
        if mode == 'summary': summary = pd.concat([summary, pd.DataFrame(sdf, index=[str(ddi)])], axis=0, sort=False)
        elif mode == 'flat': summary += [(ddi, (sdf['rows'], sdf['chans'], sdf['pols']))]
        else: summary += [((ddi, sdf['times'], sdf['baselines'], sdf['chans'], sdf['pols']))]
        table_tool.close()
        tb_tool.close()
    if verbose: print(' ' * 50, end='\r')

    if mode == 'summary': summary = summary.set_index('ddi').sort_index()
    else: summary = dict(summary)
    return summary


###########################################################################
##
## read_generic_table() - read casacore table into memory resident xds
##
###########################################################################
def read_generic_table(infile, subtables=False, timecols=None, ignore=None):
    """
    read generic casacore table format to xarray dataset loaded in memory

    Parameters
    ----------
    infile : str
        Input table filename. To read a subtable simply append the subtable folder name under the main table (ie infile = '/path/mytable.tbl/mysubtable')
    subtables : bool
        Whether or not to include subtables underneath the specified table. If true, an attribute called subtables will be added to the returned xds.
        Default False
    timecols : list
        list of column names to convert to numpy datetime format. Default None leaves times as their original casacore format.
    ignore : list
        list of column names to ignore and not try to read. Default None reads all columns

    Returns
    -------
    xarray.core.dataset.Dataset
    """
    if timecols is None: timecols = []
    if ignore is None: ignore = []

    infile = os.path.expanduser(infile)
    assert os.path.isdir(infile), "invalid input filename to read_generic_table"

    attrs = extract_table_attributes(infile)
    tb_tool = tables.table(infile, readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    if tb_tool.nrows() == 0:
        tb_tool.close()
        return xarray.Dataset(attrs=attrs)

    dims = ['row'] + ['d%i' % ii for ii in range(1, 20)]
    cols = tb_tool.colnames()
    ctype = dict([(col, tb_tool.getcell(col, 0)) for col in cols if (col not in ignore) and (tb_tool.iscelldefined(col, 0))])
    mvars, mcoords, xds = {}, {}, xarray.Dataset()

    tr = tb_tool.row(ignore, exclude=True)[:]

    # extract data for each col
    for col in ctype.keys():
        if tb_tool.coldatatype(col) == 'record': continue  # not supported

        try:
            data = np.stack([rr[col] for rr in tr])  # .astype(ctype[col].dtype)
            if isinstance(tr[0][col], dict):
                data = np.stack([rr[col]['array'].reshape(rr[col]['shape']) if len(rr[col]['array']) > 0 else np.array(['']) for rr in tr])
        except:
            # sometimes the columns are variable, so we need to standardize to the largest sizes
            if len(np.unique([isinstance(rr[col], dict) for rr in tr])) > 1: continue  # can't deal with this case
            mshape = np.array(max([np.array(rr[col]).shape for rr in tr]))
            try:
                data = np.stack([np.pad(rr[col] if len(rr[col]) > 0 else np.array(rr[col]).reshape(np.arange(len(mshape)) * 0),
                                        [(0, ss) for ss in mshape - np.array(rr[col]).shape], 'constant', constant_values=np.array([np.nan]).astype(np.array(ctype[col]).dtype)[0]) for rr in tr])
            except:
                data = []

        if len(data) == 0: continue
        if col in timecols: convert_time(data)
        if col.endswith('_ID'):
            mcoords[col] = xarray.DataArray(data, dims=['d%i_%i' % (di, ds) for di, ds in enumerate(np.array(data).shape)])
        else:
            mvars[col] = xarray.DataArray(data, dims=['d%i_%i' % (di, ds) for di, ds in enumerate(np.array(data).shape)])

    xds = xarray.Dataset(mvars, coords=mcoords)
    xds = xds.rename(dict([(dv, dims[di]) for di, dv in enumerate(xds.dims)]))
    attrs['bad_cols'] = list(np.setdiff1d([dv for dv in tb_tool.colnames()], [dv for dv in list(xds.data_vars) + list(xds.coords)]))

    # if this table has subtables, use a recursive call to store them in subtables attribute
    if subtables:
        stbl_list = sorted([tt for tt in os.listdir(infile) if os.path.isdir(os.path.join(infile, tt)) and tables.tableexists(os.path.join(infile, tt))])
        attrs['subtables'] = []
        for ii, subtable in enumerate(stbl_list):
            sxds = read_generic_table(os.path.join(infile, subtable), subtables=subtables, timecols=timecols, ignore=ignore)
            if len(sxds.dims) != 0: attrs['subtables'] += [(subtable, sxds)]

    xds = xds.assign_attrs(attrs)
    tb_tool.close()

    return xds


################################
# local helper function extract data chunk for each col, this is fed to dask.delayed
def read_flat_col_chunk(infile, col, cshape, ridxs, cstart, pstart):
    tb_tool = tables.table(infile, readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    rgrps = [(rr[0], rr[-1]) for rr in np.split(ridxs, np.where(np.diff(ridxs) > 1)[0] + 1)]
    #try:
    if (len(cshape) == 1) or (col == 'UVW'):  # all the scalars and UVW
        data = np.concatenate([tb_tool.getcol(col, rr[0], rr[1] - rr[0] + 1) for rr in rgrps], axis=0)
    elif len(cshape) == 2:  # WEIGHT, SIGMA
        data = np.concatenate([tb_tool.getcolslice(col, pstart, pstart + cshape[1] - 1, [], rr[0], rr[1] - rr[0] + 1) for rr in rgrps], axis=0)
    elif len(cshape) == 3:  # DATA and FLAG
        data = np.concatenate([tb_tool.getcolslice(col, (cstart, pstart), (cstart + cshape[1] - 1, pstart + cshape[2] - 1), [], rr[0], rr[1] - rr[0] + 1) for rr in rgrps], axis=0)
    #except:
    #    print('ERROR reading chunk: ', col, cshape, cstart, pstart)
    tb_tool.close()
    return data


###############################
# local helper
def read_flat_main_table(infile, ddi, rowidxs=None, chunks=(22000, 512, 2)):

    # get row indices relative to full main table
    if rowidxs is None:
        tb_tool = tables.taql('select rowid() as ROWS from %s where DATA_DESC_ID = %i' % (infile, ddi))
        rowidxs = tb_tool.getcol('ROWS')
        tb_tool.close()

    nrows = len(rowidxs)
    if nrows == 0:
        return xarray.Dataset()

    tb_tool = tables.taql('select * from %s where DATA_DESC_ID = %i' % (infile, ddi))
    cols = tb_tool.colnames()
    ignore = [col for col in cols if (not tb_tool.iscelldefined(col, 0)) or (tb_tool.coldatatype(col) == 'record')]
    cdata = dict([(col, tb_tool.getcol(col, 0, 1)) for col in cols if col not in ignore])
    chan_cnt, pol_cnt = [(cdata[cc].shape[1], cdata[cc].shape[2]) for cc in cdata if len(cdata[cc].shape) == 3][0]

    mvars, mcoords, bvars, xds = {}, {}, {}, xarray.Dataset()
    tb_tool.close()

    # loop over row chunks
    for rc in range(0, nrows, chunks[0]):
        crlen = min(chunks[0], nrows - rc)  # chunk row length
        rcidxs = rowidxs[rc:rc + chunks[0]]

        # loop over each column and create delayed dask arrays
        for col in cdata.keys():
            if col not in bvars: bvars[col] = []

            if len(cdata[col].shape) == 1:
                delayed_array = dask.delayed(read_flat_col_chunk)(infile, col, (crlen,), rcidxs, None, None)
                bvars[col] += [dask.array.from_delayed(delayed_array, (crlen,), cdata[col].dtype)]

            elif col == 'UVW':
                delayed_array = dask.delayed(read_flat_col_chunk)(infile, col, (crlen, 3), rcidxs, None, None)
                bvars[col] += [dask.array.from_delayed(delayed_array, (crlen, 3), cdata[col].dtype)]

            elif len(cdata[col].shape) == 2:
                pol_list = []
                dd = 1 if cdata[col].shape[1] == chan_cnt else 2
                for pc in range(0, cdata[col].shape[1], chunks[dd]):
                    plen = min(chunks[dd], cdata[col].shape[1] - pc)
                    delayed_array = dask.delayed(read_flat_col_chunk)(infile, col, (crlen, plen), rcidxs, None, pc)
                    pol_list += [dask.array.from_delayed(delayed_array, (crlen, plen), cdata[col].dtype)]
                bvars[col] += [dask.array.concatenate(pol_list, axis=1)]

            elif len(cdata[col].shape) == 3:
                chan_list = []
                for cc in range(0, chan_cnt, chunks[1]):
                    clen = min(chunks[1], chan_cnt - cc)
                    pol_list = []
                    for pc in range(0, cdata[col].shape[2], chunks[2]):
                        plen = min(chunks[2], cdata[col].shape[2] - pc)
                        delayed_array = dask.delayed(read_flat_col_chunk)(infile, col, (crlen, clen, plen), rcidxs, cc, pc)
                        pol_list += [dask.array.from_delayed(delayed_array, (crlen, clen, plen), cdata[col].dtype)]
                    chan_list += [dask.array.concatenate(pol_list, axis=2)]
                bvars[col] += [dask.array.concatenate(chan_list, axis=1)]

    # now concat all the dask chunks from each time to make the xds
    mvars = {}
    for kk in bvars.keys():
        if len(bvars[kk]) == 0:
            ignore += [kk]
            continue
        if kk == 'UVW':
            mvars[kk] = xarray.DataArray(dask.array.concatenate(bvars[kk], axis=0), dims=['row', 'uvw_index'])
        elif len(bvars[kk][0].shape) == 2 and (bvars[kk][0].shape[-1] == pol_cnt):
            mvars[kk] = xarray.DataArray(dask.array.concatenate(bvars[kk], axis=0), dims=['row', 'pol'])
        elif len(bvars[kk][0].shape) == 2 and (bvars[kk][0].shape[-1] == chan_cnt):
            mvars[kk] = xarray.DataArray(dask.array.concatenate(bvars[kk], axis=0), dims=['row', 'chan'])
        else:
            mvars[kk] = xarray.DataArray(dask.array.concatenate(bvars[kk], axis=0), dims=['row', 'chan', 'pol'][:len(bvars[kk][0].shape)])

    mvars['TIME'] = xarray.DataArray(convert_time(mvars['TIME'].values), dims=['row']).chunk({'row': chunks[0]})
    attrs = extract_table_attributes(infile)
    attrs['bad_cols'] = ignore
    xds = xarray.Dataset(mvars, coords=mcoords).assign_attrs(attrs)
    return xds


###########################################################################################
##
## read_ms() - read a measuremetset into an mxds of delayed xds's
##
###########################################################################################
def read_ms(infile, subtables=False, expand=False, chunks=None, verbose=False, **kwargs):
    """
    Read legacy format MS to xarray Visibility Dataset

    The MS is partitioned by DDI, which guarantees a fixed data shape per partition. This results in separate xarray
    dataset (xds) partitions contained within a main xds (mxds).

    Parameters
    ----------
    infile : str
        Input MS filename
    rowmap : dict
        Dictionary of DDI to tuple of (row indices, channel indices). Returned by ms_selection function. Default None ignores selections
    subtables : bool
        Also read and include subtables along with main table selection. Default False will omit subtables (faster)
    expand : bool
        Whether or not to return the original flat row structure of the MS (False) or expand the rows to time x baseline dimensions (True).
        Expanding the rows allows for easier indexing and parallelization across time and baseline dimensions, at the cost of some conversion
        time. Default False
    chunks: tuple or list
        Can be used to set a specific chunk shape (with a tuple of ints), or to control the optimization used for automatic chunking (with a list of ints).
        A TUPLE of ints in the form of (row, chan, pol) will use a fixed chunk shape. A LIST or numpy array of ints in the form of [idx1, etc] will trigger
        auto-chunking optimized for the given indices, with row=0, chan=1, pol=2. Default None uses auto-chunking with a best fit across all dimensions
        (probably sub-optimal for most cases).
    verbose : bool
        self explanatory
    **kwargs : str
        Selection parameters from the standard way of making CASA MS selections. Supported keys are: spw, field, scan, baseline, time, scanintent, uvdist,
        polarization, array, observation.  Values are strings.

    Returns
    -------
    xarray.core.dataset.Dataset
      Main xarray dataset of datasets for this visibility set
    """
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    # parse filename to use
    infile = os.path.expanduser(infile)
    assert os.path.isdir(infile), "invalid input filename to read_ms"

    # get the indices of the ms selection (if any)
    rowmap = ms_selection(infile, verbose=verbose, **kwargs)

    # we need the spectral window, polarization, and data description tables for processing the main table
    spw_xds = read_generic_table(os.path.join(infile, 'SPECTRAL_WINDOW'))
    pol_xds = read_generic_table(os.path.join(infile, 'POLARIZATION'))
    ddi_xds = read_generic_table(os.path.join(infile, 'DATA_DESCRIPTION'))

    # each DATA_DESC_ID (ddi) is a fixed shape that may differ from others
    # form a list of ddis to process, each will be placed it in its own xarray dataset and partition
    ddis = np.arange(ddi_xds.row.shape[0]) if rowmap is None else list(rowmap.keys())
    xds_list = []

    # figure out the chunking for each DDI, either one fixed shape or an auto-computed one
    if type(chunks) != tuple:
        mshape = describe_ms(infile, mode='flat', rowmap=rowmap, verbose=False)
        chunks = dict([(ddi, optimal_chunking(didxs=chunks, chunk_size='auto', data_shape=mshape[ddi])) for ddi in mshape])

    ####################################################################
    # process each selected DDI from the input MS, assume a fixed shape within the ddi (should always be true)
    for ddi in ddis:
        rowidxs = None if rowmap is None else rowmap[ddi][0]
        chanidxs = None if rowmap is None else rowmap[ddi][1]
        if ((rowidxs is not None) and (len(rowidxs) == 0)) or ((chanidxs is not None) and (len(chanidxs) == 0)): continue
        if verbose: print('reading DDI %i with chunking %s...' % (ddi, str(chunks[ddi] if type(chunks) == dict else chunks)))

        xds = read_flat_main_table(infile, ddi, rowidxs=rowidxs, chunks=chunks[ddi] if type(chunks) == dict else chunks)
        if len(xds.dims) == 0: continue

        # grab the channel frequency values from the spw table data and pol idxs from the polarization table, add spw and pol ids
        chan = spw_xds.CHAN_FREQ.values[ddi_xds.SPECTRAL_WINDOW_ID.values[ddi], :xds.chan.shape[0]]
        pol = pol_xds.CORR_TYPE.values[ddi_xds.POLARIZATION_ID.values[ddi], :xds.pol.shape[0]]

        coords = {'chan': chan, 'pol': pol, 'spw_id': [ddi_xds['SPECTRAL_WINDOW_ID'].values[ddi]], 'pol_id': [ddi_xds['POLARIZATION_ID'].values[ddi]]}
        xds = xds.assign_coords(coords)  # .assign_attrs(attrs)

        # filter by channel selection
        if (chanidxs is not None) and (len(chanidxs) < len(xds.chan)):
            xds = xds.isel(chan=chanidxs)
            spw_xds['CHAN_FREQ'][ddi_xds.SPECTRAL_WINDOW_ID.values[ddi], :len(chanidxs)] = spw_xds.CHAN_FREQ[ddi_xds.SPECTRAL_WINDOW_ID.values[ddi], chanidxs]

        # expand the row dimension out to (time, baseline)
        if expand:
            xds = expand_xds(xds)

        xds_list += [('xds' + str(ddi), xds)]

    # read other subtables
    xds_list += [('SPECTRAL_WINDOW', spw_xds), ('POLARIZATION', pol_xds), ('DATA_DESCRIPTION', ddi_xds)]
    if subtables:
        skip_tables = ['SORTED_TABLE', 'SPECTRAL_WINDOW', 'POLARIZATION', 'DATA_DESCRIPTION']
        stbl_list = sorted([tt for tt in os.listdir(infile) if os.path.isdir(os.path.join(infile, tt)) and (tt not in skip_tables) and (tables.tableexists(os.path.join(infile, tt)))])
        for ii, subtable in enumerate(stbl_list):
            if verbose: print('reading subtable %s...' % subtable)
            sxds = read_generic_table(os.path.join(infile, subtable), subtables=True, timecols=['TIME'], ignore=[])
            if len(sxds.dims) != 0: xds_list += [(subtable, sxds)]

    # build the master xds to return
    mxds = vis_xds_packager(xds_list)
    return mxds



############################################################################
##
## write_generic_table() - write any xds to generic casacore table format
##
############################################################################
def write_generic_table(xds, outfile, subtable='', cols=None, verbose=False):
    """
    Write generic xds contents back to casacore table format on disk

    Parameters
    ----------
    xds : xarray.Dataset
        Source xarray dataset data
    outfile : str
        Destination filename (or parent main table if writing subtable)
    subtable : str
        Name of the subtable being written, triggers special logic to add subtable to parent table.  Default '' for normal generic writes
    cols : str or list
        List of cols to write. Default None writes all columns
    """
    outfile = os.path.expanduser(outfile)
    if verbose: print('writing %s...' % os.path.join(outfile, subtable))
    if cols is None: cols = list(set(list(xds.data_vars) + [cc for cc in xds.coords if cc not in xds.dims] + (list(xds.attrs['column_descriptions'].keys() if 'column_descriptions' in xds.attrs else []))))
    cols = list(np.atleast_1d(cols))

    max_rows = xds.row.shape[0] if 'row' in xds.dims else 0
    create_table(os.path.join(outfile, subtable), xds, max_rows, infile=None, cols=cols, generic=True)

    tb_tool = tables.table(os.path.join(outfile, subtable), readonly=False, lockoptions={'option': 'permanentwait'}, ack=False)
    try:
        for dv in cols:
            if (dv not in xds) or (np.prod(xds[dv].shape) == 0): continue
            values = xds[dv].values if xds[dv].dtype != 'datetime64[ns]' else revert_time(xds[dv].values)
            tb_tool.putcol(dv, values, 0, values.shape[0], 1)
    except:
        print("ERROR: exception in write generic table - %s, %s, %s, %s" % (os.path.join(outfile,subtable), dv, str(values.shape), tb_tool.nrows()))

    # now we have to add this subtable to the main table keywords (assuming a main table already exists)
    if len(subtable) > 0:
        main_tbl = tables.table(outfile, readonly=False, lockoptions={'option': 'permanentwait'}, ack=False)
        main_tbl.putkeyword(subtable, tb_tool, makesubrecord=True)
        main_tbl.done()
    tb_tool.close()

    # if this table has its own subtables, they need to be written out recursively
    if 'subtables' in xds.attrs:
        for st in list(xds.attrs['subtables']):
            write_generic_table(st[1], os.path.join(outfile, subtable, st[0]), subtable='', verbose=verbose)


###################################
# local helper
def write_main_table_slice(xda, outfile, ddi, col, full_shape, starts):
    """
    Write an xds row chunk to the corresponding main table slice
    """
    # trigger the DAG for this chunk and return values while the table is unlocked
    values = xda.compute().values
    if xda.dtype == 'datetime64[ns]':
        values = revert_time(values)

    tbs = tables.table(outfile, readonly=False, lockoptions={'option': 'permanentwait'}, ack=True)

    # try:
    if (values.ndim == 1) or (col == 'UVW') or (values.shape[1:] == full_shape):  # scalar columns
        tbs.putcol(col, values, starts[0], len(values))
    else:
        if not tbs.iscelldefined(col, starts[0]): tbs.putcell(col, starts[0] + np.arange(len(values)), np.zeros((full_shape)))
        tbs.putcolslice(col, values, starts[1:values.ndim], tuple(np.array(starts[1:values.ndim]) + np.array(values.shape[1:]) - 1), [], starts[0], len(values), 1)
    # except:
    #    print("ERROR: write exception - %s, %s, %s" % (col, str(values.shape), str(starts)))
    tbs.close()

    
    
####################################################################################################
##
## write_ms() - write mxds to casacore MS format on disk
##
####################################################################################################
def write_ms(mxds, outfile, infile=None, subtables=False, modcols=None, verbose=False, execute=True):
    """
    Write ms format xds contents back to casacore table format on disk

    Parameters
    ----------
    mxds : xarray.Dataset
        Source multi-xarray dataset (originally created by read_ms)
    outfile : str
        Destination filename
    infile : str
        Source filename to copy subtables from. Generally faster than reading/writing through mxds via the subtables parameter. Default None
        does not copy subtables to output.
    subtables : bool
        Also write subtables from mxds. Default of False only writes mxds attributes that begin with xdsN to the MS main table.
        Setting to True will write all other mxds attributes to subtables of the main table.  This is probably going to be SLOW!
        Use infile instead whenever possible.
    modcols : list
        List of strings indicating what column(s) were modified (aka xds data_vars). Different logic can be applied to speed up processing when
        a data_var has not been modified from the input. Default None assumes everything has been modified (SLOW)
    verbose : bool
        Whether or not to print output progress. Since writes will typically execute the DAG, if something is
        going to go wrong, it will be here.  Default False
    execute : bool
        Whether or not to actually execute the DAG, or just return it with write steps appended. Default True will execute it
    """
    outfile = os.path.expanduser(outfile)
    if verbose: print('initializing output...')
    start = time.time()

    xds_list = [flatten_xds(mxds.attrs[kk]) for kk in mxds.attrs if kk.startswith('xds')]
    cols = list(set([dv for dx in xds_list for dv in dx.data_vars]))
    if modcols is None: modcols = cols
    modcols = list(np.atleast_1d(modcols))

    # create an empty main table with enough space for all desired xds partitions
    # the first selected xds partition will be passed to create_table to provide a definition of columns and table keywords
    # we first need to add in additional keywords for the selected subtables that will be written as well
    max_rows = np.sum([dx.row.shape[0] for dx in xds_list])
    create_table(outfile, xds_list[0], max_rows=max_rows, infile=infile, cols=cols, generic=False)

    # start a list of dask delayed writes to disk (to be executed later)
    # the SPECTRAL_WINDOW, POLARIZATION, and DATA_DESCRIPTION tables must always be present and will always be written
    delayed_writes = [dask.delayed(write_generic_table)(mxds.SPECTRAL_WINDOW, outfile, 'SPECTRAL_WINDOW', cols=None)]
    delayed_writes += [dask.delayed(write_generic_table)(mxds.POLARIZATION, outfile, 'POLARIZATION', cols=None)]
    delayed_writes += [dask.delayed(write_generic_table)(mxds.DATA_DESCRIPTION, outfile, 'DATA_DESCRIPTION', cols=None)]
    if subtables:  # also write the rest of the subtables
        for subtable in list(mxds.attrs.keys()):
            if subtable.startswith('xds') or (subtable in ['SPECTRAL_WINDOW', 'POLARIZATION', 'DATA_DESCRIPTION']): continue
            if verbose: print('writing subtable %s...' % subtable)
            delayed_writes += [dask.delayed(write_generic_table)(mxds.attrs[subtable], outfile, subtable, cols=None, verbose=verbose)]

    ddi_row_start = 0  # output rows will be ordered by DDI
    for xds in xds_list:
        txds = xds.copy().unify_chunks()
        ddi = txds.DATA_DESC_ID[:1].values[0]

        # serial write entire DDI column first so subsequent delayed writes can find their spot
        if verbose: print('setting up DDI %i...' % ddi)

        # write each chunk of each modified data_var, triggering the DAG along the way
        for col in modcols:
            if col not in txds: continue  # this can happen with bad_cols, should still be created in create_table()
            chunks = txds[col].chunks
            dims = txds[col].dims
            for d0 in range(len(chunks[0])):
                d0start = ([0] + list(np.cumsum(chunks[0][:-1])))[d0]

                for d1 in range(len(chunks[1]) if len(chunks) > 1 else 1):
                    d1start = ([0] + list(np.cumsum(chunks[1][:-1])))[d1] if len(chunks) > 1 else 0

                    for d2 in range(len(chunks[2]) if len(chunks) > 2 else 1):
                        d2start = ([0] + list(np.cumsum(chunks[2][:-1])))[d2] if len(chunks) > 2 else 0

                        starts = [d0start, d1start, d2start]
                        lengths = [chunks[0][d0], (chunks[1][d1] if len(chunks) > 1 else 0), (chunks[2][d2] if len(chunks) > 2 else 0)]
                        slices = [slice(starts[0], starts[0]+lengths[0]), slice(starts[1], starts[1]+lengths[1]), slice(starts[2], starts[2]+lengths[2])]
                        txda = txds[col].isel(dict(zip(dims, slices)), missing_dims='ignore')
                        starts[0] = starts[0] + ddi_row_start  # offset to end of table
                        delayed_writes += [dask.delayed(write_main_table_slice)(txda, outfile, ddi=ddi, col=col, full_shape=txds[col].shape[1:], starts=starts)]

        # now write remaining data_vars from the xds that weren't modified
        # this can be done faster by collapsing the chunking to maximum size (minimum #) possible
        max_chunk_size = np.prod([txds.chunks[kk][0] for kk in txds.chunks if kk in ['row', 'chan', 'pol']])
        for col in list(np.setdiff1d(cols, modcols)):
            if col not in txds: continue  # this can happen with bad_cols, should still be created in create_table()
            col_chunk_size = np.prod([kk[0] for kk in txds[col].chunks])
            col_rows = int(np.ceil(max_chunk_size / col_chunk_size)) * txds[col].chunks[0][0]
            for rr in range(0, txds[col].row.shape[0], col_rows):
                txda = txds[col].isel(row=slice(rr, rr + col_rows))
                delayed_writes += [dask.delayed(write_main_table_slice)(txda, outfile, ddi=ddi, col=col, full_shape=txda.shape[1:], starts=(rr+ddi_row_start,)+(0,)*(len(txda.shape)-1))]

        ddi_row_start += txds.row.shape[0]  # next xds will be appended after this one

    if execute:
        if verbose: print('triggering DAG...')
        zs = dask.compute(delayed_writes)
        if verbose: print('execution time %0.2f sec' % (time.time() - start))
    else:
        if verbose: print('returning delayed task list')
        return delayed_writes

def write_ms_serial(mxds, outfile, infile=None, subtables=False, verbose=False, execute=True, memory_available_in_bytes=500000000000):
    """
    Write ms format xds contents back to casacore table format on disk

    Parameters
    ----------
    mxds : xarray.Dataset
        Source multi-xarray dataset (originally created by read_ms)
    outfile : str
        Destination filename
    infile : str
        Source filename to copy subtables from. Generally faster than reading/writing through mxds via the subtables parameter. Default None
        does not copy subtables to output.
    subtables : bool
        Also write subtables from mxds. Default of False only writes mxds attributes that begin with xdsN to the MS main table.
        Setting to True will write all other mxds attributes to subtables of the main table.  This is probably going to be SLOW!
        Use infile instead whenever possible.
    modcols : list
        List of strings indicating what column(s) were modified (aka xds data_vars). Different logic can be applied to speed up processing when
        a data_var has not been modified from the input. Default None assumes everything has been modified (SLOW)
    verbose : bool
        Whether or not to print output progress. Since writes will typically execute the DAG, if something is
        going to go wrong, it will be here.  Default False
    execute : bool
        Whether or not to actually execute the DAG, or just return it with write steps appended. Default True will execute it
    """
    
    print('*********************')
    outfile = os.path.expanduser(outfile)
    if verbose: print('initializing output...')
    start = time.time()

    xds_list = [flatten_xds(mxds.attrs[kk]) for kk in mxds.attrs if kk.startswith('xds')]
    cols = list(set([dv for dx in xds_list for dv in dx.data_vars]))
    cols = list(np.atleast_1d(cols))
    
    # create an empty main table with enough space for all desired xds partitions
    # the first selected xds partition will be passed to create_table to provide a definition of columns and table keywords
    # we first need to add in additional keywords for the selected subtables that will be written as well
    max_rows = np.sum([dx.row.shape[0] for dx in xds_list])
    create_table(outfile, xds_list[0], max_rows=max_rows, infile=infile, cols=cols, generic=False)

    
    # start a list of dask delayed writes to disk (to be executed later)
    # the SPECTRAL_WINDOW, POLARIZATION, and DATA_DESCRIPTION tables must always be present and will always be written
    write_generic_table(mxds.SPECTRAL_WINDOW, outfile, 'SPECTRAL_WINDOW', cols=None)
    write_generic_table(mxds.POLARIZATION, outfile, 'POLARIZATION', cols=None)
    write_generic_table(mxds.DATA_DESCRIPTION, outfile, 'DATA_DESCRIPTION', cols=None)
    
    
    if subtables:  # also write the rest of the subtables
        #for subtable in list(mxds.attrs.keys()):
        #'OBSERVATION','STATE'
        #['FEED','OBSERVATION','FIELD','ANTENNA','HISTORY','STATE']
        #['FEED','FIELD','ANTENNA','HISTORY']
        #,'FIELD','ANTENNA'
        #for subtable in ['OBSERVATION']:
        for subtable in list(mxds.attrs.keys()):
            if subtable.startswith('xds') or (subtable in ['SPECTRAL_WINDOW', 'POLARIZATION', 'DATA_DESCRIPTION']): continue
            if verbose: print('writing subtable %s...' % subtable)
            #print(subtable)
            #print(mxds.attrs[subtable])
            write_generic_table(mxds.attrs[subtable], outfile, subtable, cols=None, verbose=verbose)
            
    
    from sirius._sirius_utils._ms_utils import _calc_optimal_ms_chunk_shape
    from sirius._sirius_utils._dask_utils import _get_schedular_info
    
    vis_data_shape = mxds.xds0.DATA.shape
    rows_chunk_size = _calc_optimal_ms_chunk_shape(memory_available_in_bytes,vis_data_shape,16,'DATA')
    
    #print(rows_chunk_size)
    #rows_chunk_size = 200000000
    # write each chunk of each modified data_var, triggering the DAG along the way
    tbs = tables.table(outfile, readonly=False, lockoptions={'option': 'permanentwait'}, ack=True)
   
    start_main =time.time()
    for col in cols:
        xda = mxds.xds0[col]
        #print(col,xda.dtype)
        
        for start_row in np.arange(0,vis_data_shape[0],rows_chunk_size):
            end_row = start_row + rows_chunk_size
            if end_row > vis_data_shape[0]:
                end_row = vis_data_shape[0]
            
            start =time.time()
            values = xda[start_row:end_row,].compute().values
            if xda.dtype == 'datetime64[ns]':
                values = revert_time(values)
            #print('1. Time', time.time()-start, values.shape)

            start =time.time()
            tbs.putcol(col, values, start_row, len(values))
            #print('2. Time', time.time()-start)
            
    print('3. Time', time.time()-start_main)
        
    tbs.unlock()
    tbs.close()
    
######################################################################
##
## visplot() - produce matplotlib plots of xarray data arrays
##
######################################################################
def visplot(xda, axis=None, overplot=False, drawplot=True, tsize=250):
    """
    Plot a preview of Visibility xarray DataArray contents

    Parameters
    ----------
    xda : xarray.core.dataarray.DataArray
        input DataArray to plot
    axis : str or list or xarray.core.dataarray.DataArray
        Coordinate(s) within the xarray DataArray, or a second xarray DataArray to plot against. Default None uses range.
        All other coordinates will be maxed across dims
    overplot : bool
        Overlay new plot on to existing window. Default of False makes a new window for each plot
    drawplot : bool
        Display plot window. Should pretty much always be True unless you want to overlay things
        in a Jupyter notebook.
    tsize : int
        target size of the preview plot (might be smaller). Default is 250 points per axis

    Returns
    -------
      Open matplotlib window
    """
    import matplotlib.pyplot as plt
    import xarray
    import numpy as np
    import warnings
    warnings.simplefilter("ignore", category=RuntimeWarning)  # suppress warnings about nan-slices
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    if overplot:
        axes = None
    else:
        fig, axes = plt.subplots(1, 1)

    # fast decimate to roughly the desired size
    thinf = np.ceil(np.array(xda.shape) / tsize)
    txda = xda.thin(dict([(xda.dims[ii], int(thinf[ii])) for ii in range(len(thinf))]))

    # can't plot complex numbers, bools (sometimes), or strings
    if (txda.dtype == 'complex128') or (txda.dtype == 'complex64'):
        txda = (txda.real ** 2 + txda.imag ** 2) ** 0.5
    elif txda.dtype == 'bool':
        txda = txda.astype(int)
    elif txda.dtype.type is np.int32:
        txda = txda.where(txda > np.full((1), np.nan, dtype=np.int32)[0])
    elif txda.dtype.type is np.str_:
        txda = xarray.DataArray(np.unique(txda, return_inverse=True)[1], dims=txda.dims, coords=txda.coords, name=txda.name)

    ######################
    # decisions based on supplied axis to plot against
    # no axis - plot against range of data
    # collapse all but first dimension
    if axis is None:
        collapse = [ii for ii in range(1, txda.ndim)]
        if len(collapse) > 0: txda = txda.max(axis=collapse)
        txda[txda.dims[0]] = np.arange(txda.shape[0])
        txda.plot.line(ax=axes, marker='.', linewidth=0.0)

    # another xarray DataArray as axis
    elif type(axis) == xarray.core.dataarray.DataArray:
        txda2 = axis.thin(dict([(xda.dims[ii], int(thinf[ii])) for ii in range(len(thinf))]))
        if txda2.dtype.type is np.int32: txda2 = txda2.where(txda2 > np.full((1), np.nan, dtype=np.int32)[0])
        xarray.Dataset({txda.name: txda, txda2.name: txda2}).plot.scatter(txda.name, txda2.name)

    # single axis
    elif len(np.atleast_1d(axis)) == 1:
        axis = np.atleast_1d(axis)[0]
        # coord ndim is 1
        if txda[axis].ndim == 1:
            collapse = [ii for ii in range(txda.ndim) if txda.dims[ii] not in txda[axis].dims]
            if len(collapse) > 0: txda = txda.max(axis=collapse)
            txda.plot.line(ax=axes, x=axis, marker='.', linewidth=0.0)

        # coord ndim is 2
        elif txda[axis].ndim == 2:
            collapse = [ii for ii in range(txda.ndim) if txda.dims[ii] not in txda[axis].dims]
            if len(collapse) > 0: txda = txda.max(axis=collapse)
            txda.plot.pcolormesh(ax=axes, x=axis, y=txda.dims[0])

    # two axes
    elif len(axis) == 2:
        collapse = [ii for ii in range(txda.ndim) if txda.dims[ii] not in (txda[axis[0]].dims + txda[axis[1]].dims)]
        if len(collapse) > 0: txda = txda.max(axis=collapse)
        txda.plot.pcolormesh(ax=axes, x=axis[0], y=axis[1])

    plt.title(txda.name)
    if drawplot:
        plt.show()




##################################################################################################
##
## Images
##
##################################################################################################

###################################
# local helper
def read_image_chunk(infile, shapes, starts):
    tb_tool = tables.table(infile, readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    data = tb_tool.getcellslice(tb_tool.colnames()[0], 0, starts, tuple(np.array(starts) + np.array(shapes) - 1))
    tb_tool.close()
    return data


###################################
# local helper
def read_image_array(infile, dimorder, chunks):
    tb_tool = tables.table(infile, readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    cshape = eval(tb_tool.getcolshapestring(tb_tool.colnames()[0])[0])
    cdata = tb_tool.getcellslice(tb_tool.colnames()[0], 0, tuple(np.repeat(0, len(cshape))), tuple(np.repeat(0, len(cshape))))
    tb_tool.close()

    # expand the actual data shape to the full 5 possible dims
    full_shape = cshape + [1 for rr in range(5) if rr >= len(cshape)]
    full_chunks = chunks[::-1] + [1 for rr in range(5) if rr >= len(chunks)]
    d0slices = []
    for d0 in range(0, full_shape[0], full_chunks[0]):
        d0len = min(full_chunks[0], full_shape[0] - d0)
        d1slices = []

        for d1 in range(0, full_shape[1], full_chunks[1]):
            d1len = min(full_chunks[1], full_shape[1] - d1)
            d2slices = []

            for d2 in range(0, full_shape[2], full_chunks[2]):
                d2len = min(full_chunks[2], full_shape[2] - d2)
                d3slices = []

                for d3 in range(0, full_shape[3], full_chunks[3]):
                    d3len = min(full_chunks[3], full_shape[3] - d3)
                    d4slices = []

                    for d4 in range(0, full_shape[4], full_chunks[4]):
                        d4len = min(full_chunks[4], full_shape[4] - d4)

                        shapes = tuple([d0len, d1len, d2len, d3len, d4len][:len(cshape)])
                        starts = tuple([d0, d1, d2, d3, d4][:len(cshape)])
                        delayed_array = dask.delayed(read_image_chunk)(infile, shapes, starts)
                        d4slices += [dask.array.from_delayed(delayed_array, shapes, cdata.dtype)]
                    d3slices += [dask.array.concatenate(d4slices, axis=4)] if len(cshape) > 4 else d4slices
                d2slices += [dask.array.concatenate(d3slices, axis=3)] if len(cshape) > 3 else d3slices
            d1slices += [dask.array.concatenate(d2slices, axis=2)] if len(cshape) > 2 else d2slices
        d0slices += [dask.array.concatenate(d1slices, axis=1)] if len(cshape) > 1 else d1slices

    xda = xarray.DataArray(dask.array.concatenate(d0slices, axis=0), dims=dimorder[::-1]).transpose()
    return xda


#############################################################################
##
## read_image() - read an image from casacore format to delayed xds
##
#############################################################################
def read_image(infile, masks=True, history=True, chunks=None, verbose=False):
    """
    Read casacore format Image to xarray Image Dataset format

    Parameters
    ----------
    infile : str
        Input image filename (.image or .fits format)
    masks : bool
        Also read image masks as additional image data_vars. Default is True
    history : bool
        Also read history log table. Default is True
    chunks: tuple or list
        Can be used to set a specific chunk shape (with a tuple of ints), or to control the optimization used for automatic chunking (with a list of ints).
        A TUPLE of ints in the form of (l, m, chan, pol, component) will use a fixed chunk shape. A LIST or numpy array of ints in the form of [idx1, etc]
        will trigger auto-chunking optimized for the given indices, with l=0, m=1, chan=2, pol=3, component=4. Default None uses auto-chunking with a best
        fit across all dimensions (probably sub-optimal for most cases).

    Returns
    -------
    xarray.core.dataset.Dataset
        new xarray Datasets of Image data contents
    """
    infile = os.path.expanduser(infile)

    IA = ia()
    QA = qa()
    rc = IA.open(infile)
    csys = IA.coordsys()
    ims = IA.shape()  # image shape
    attrs = extract_table_attributes(infile)
    if verbose: print('opening %s with shape %s' % (infile, str(ims)))

    # construct a mapping of dimension names to image indices
    diraxes = [aa.lower().replace(' ', '_') for cc in attrs['coords'].items() if (cc[0][:-1] in ['direction', 'linear']) and len(cc[1]['axes']) >= 2 for aa in cc[1]['axes']]
    dimmap = [(coord[:-1]+str(ii), ci) for coord in attrs['coords'] if coord[:-1] in ['direction', 'stokes', 'spectral', 'linear'] for ii,ci in enumerate(attrs['coords']['pixelmap%s' % coord[-1]])]
    dimmap = [(rr[0].replace('stokes0','pol').replace('spectral0','chan').replace('direction0','l').replace('direction1','m'), rr[1]) for rr in dimmap]
    if ('linear0' in np.vstack(dimmap)[:,0]) and ('linear1' in np.vstack(dimmap)[:,0]):
        dimmap = [(rr[0].replace('linear0', 'l').replace('linear1', 'm'), rr[1]) for rr in dimmap]
    dimmap = [(rr[0].replace('linear0', 'component'), rr[1]) for rr in dimmap if rr[1] >= 0]
    dimmap = dict([(diraxes[int(rr[0][-1])], rr[1]) if rr[0].startswith('linear') or rr[0].startswith('direction') else rr for rr in dimmap])

    # compute world coordinates for spherical dimensions
    sphr_dims = [dimmap['l'], dimmap['m']] if ('l' in dimmap) and ('m' in dimmap) else []
    coord_idxs = np.mgrid[[range(ims[dd]) if dd in sphr_dims else range(1) for dd in range(len(ims))]].reshape(len(ims), -1)
    coord_world = csys.toworldmany(coord_idxs.astype(float))['numeric'][sphr_dims].reshape((-1,) + tuple(ims[sphr_dims]))
    coords = dict([(diraxes[di], (['l', 'm'], coord_world[di])) for di, dd in enumerate(sphr_dims)])

    # compute world coordinates for cartesian dimensions
    for dm in dimmap.items():
        if dm[0] in ['l', 'm']: continue
        coord_idxs = np.mgrid[[range(ims[dd]) if dd == dm[1] else range(1) for dd in range(len(ims))]].reshape(len(ims), -1)
        coord_world = csys.toworldmany(coord_idxs.astype(float))['numeric'][dm[1]].reshape(-1,)
        coords.update({dm[0]: coord_world})

    # assign values to l, m coords based on incr and refpix in metadata
    if len(sphr_dims) > 0:
        sphr_coord = [coord for coord in attrs['coords'] if coord.startswith('direction')]
        sphr_coord = [coord for coord in attrs['coords'] if coord.startswith('linear')][0] if len(sphr_coord) == 0 else sphr_coord[0]
        coords['l'] = np.arange(-attrs['coords'][sphr_coord]['crpix'][0], ims[0]-attrs['coords'][sphr_coord]['crpix'][0]) * attrs['coords'][sphr_coord]['cdelt'][0]
        coords['m'] = np.arange(-attrs['coords'][sphr_coord]['crpix'][1], ims[1]-attrs['coords'][sphr_coord]['crpix'][1]) * attrs['coords'][sphr_coord]['cdelt'][1]

    rc = csys.done()
    rc = IA.close()

    # determine chunking based on type of value passed in
    dimorder = [dd for rr in range(5) for dd in dimmap if (dimmap[dd] is not None) and (dimmap[dd] == rr)]
    if type(chunks) is tuple:  # specific chunk values are given in (l, m, chan, pol) order, rearrange to match the actual data order
        chunks = list(np.array(chunks + (9999999,))[[['l', 'm', 'chan', 'pol', 'component'].index(rr) for rr in dimorder]])
    elif chunks is None:
        chunks = optimal_chunking(didxs=None, chunk_size='auto', data_shape=ims)
    else:  # indices for auto-chunking to optimize are given in (l, m, chan, pol) order, rearrange to match the actual data order
        chunks = [dimmap[['l', 'm', 'chan', 'pol', 'component'][cc]] for cc in chunks if ['l', 'm', 'chan', 'pol', 'component'][cc] in dimmap]
        chunks = optimal_chunking(didxs=chunks, chunk_size='auto', data_shape=ims)
    if verbose: print('chunk shape set to %s' % str(chunks))

    # wrap the actual image data reads in dask delayed calls returned as an xarray dataarray
    xds = xarray.Dataset(coords=coords)
    xda = read_image_array(infile, dimorder, list(chunks))
    xda = xda.rename('IMAGE')
    xds[xda.name] = xda

    # add mask(s) alongside image data
    if masks and 'masks' in attrs:
        for ii, mask in enumerate(list(attrs['masks'].keys())):
            if not os.path.isdir(os.path.join(infile, mask)): continue
            xda = read_image_array(os.path.join(infile, mask), dimorder, list(chunks))
            xda = xda.rename('IMAGE_%s' % mask)
            xds[xda.name] = xda
            attrs[mask+'_column_descriptions'] = extract_table_attributes(os.path.join(infile, mask))['column_descriptions']

    # if also loading history, put it as another xds in the attrs
    if history and os.path.isdir(os.path.join(infile, 'logtable')):
        attrs['history'] = read_generic_table(os.path.join(infile, 'logtable'))

    if 'coords' in attrs: attrs['icoords'] = attrs.pop('coords')  # rename coord table keyword to avoid confusion with xds coords
    xds = xds.assign_attrs(attrs)

    return xds



########################################
# local helper
def write_image_slice(xda, outfile, col, starts):
    """
    Write image xda chunk to the corresponding image table slice
    """
    # trigger the DAG for this chunk and return values while the table is unlocked
    values = xda.compute().values

    tb_tool = tables.table(outfile, readonly=False, lockoptions={'option': 'permanentwait'}, ack=False)
    tb_tool.putcellslice(col, 0, values, starts, tuple(np.array(starts) + np.array(values.shape) - 1))
    tb_tool.close()


######################################################################################################
##
## write_image() - write xds to casacore format image on disk
##
######################################################################################################
def write_image(xds, outfile, portion='IMAGE', masks=True, history=True, verbose=False, execute=True):
    """
    Read casacore format Image to xarray Image Dataset format

    Parameters
    ----------
    xds : xarray.Dataset
        Image xarray dataset to write
    outfile : str
        Output image filename (.image format)
    portion : str
        Name of the data_var in the xds that corresponds to the image data. Default 'IMAGE'
    masks : bool
        Also write the masks to the output. Can be used instead of infile parameter. Default True
    history : bool
        Also write the history log file to the output. Can be used instead of infile paramter. Default True
    verbose : bool
        Whether or not to print output progress. Since writes will typically execute the DAG, if something is
        going to go wrong, it will be here.  Default False
    execute : bool
        Whether or not to actually execute the DAG, or just return it with write steps appended. Default True will execute it
    """
    outfile = os.path.expanduser(outfile)
    start = time.time()
    #xds = xds.copy()

    # initialize list of column names and xda's to be written. The column names are not the same as the data_var names
    cols = [list(xds.attrs['column_descriptions'].keys())[0] if 'column_descriptions' in xds.attrs else list(xds.data_vars.keys())[0]]
    xda_list = [xds[portion]]
    subtable_list = ['']
    if 'icoords' in xds.attrs: xds.attrs['coords'] = xds.attrs.pop('icoords')  # rename back for proper table keyword creation

    # initialize output table (must do it this way since create_table mysteriously throws image tool errors when subsequently opened)
    IA = ia()
    imtype = 'd' if xds[portion].dtype == 'float64' else 'c' if xds[portion].dtype == 'complex64' else 'cd' if xds[portion].dtype == 'complex128' else 'f'
    IA.fromshape(outfile, list(xds[portion].shape), csys=xds.attrs['coords'], overwrite=True, log=False, type=imtype)
    IA.close()

    # write image history to logfile subtable (not delayed)
    if history and ('history' in xds.attrs):
        if verbose: print('writing history log...')
        write_generic_table(xds.history, outfile, subtable='logtable')

    # add masks to the list of xda's to be written
    if masks and ('masks' in xds.attrs):
        for mask in xds.masks:
            if verbose: print('writing %s...' % mask)
            mask_var = '%s_%s' % (portion, mask)
            if (mask + '_column_descriptions' not in xds.attrs) or (mask_var not in xds): continue
            cols += [list(xds.attrs[mask+'_column_descriptions'].keys())[0]]
            xda_list += [xds[mask_var]]
            subtable_list += [mask]
            xds.attrs['masks'][mask]['mask'] = 'Table: %s' % os.path.abspath(os.path.join(outfile, mask))
            xds.attrs[mask+'_column_descriptions'][cols[-1]]['shape'] = list(xds[mask_var].transpose().shape)
            txds = xarray.Dataset({mask_var: xds[mask_var]}).assign_attrs({'column_descriptions': xds.attrs[mask+'_column_descriptions']})
            create_table(os.path.join(outfile, mask), txds, max_rows=1, infile=None, cols=[cols[-1]], generic=True)

    # write xds attribute to output table keywords
    tb_tool = tables.table(outfile, readonly=False, lockoptions={'option': 'permanentwait'}, ack=False)
    for attr in xds.attrs:
        if (attr in ['bad_cols', 'bad_types', 'column_descriptions', 'history', 'subtables', 'info']) or attr.endswith('column_descriptions'): continue
        tb_tool.putkeyword(attr, xds.attrs[attr])
    if 'info' in xds.attrs: tb_tool.putinfo(xds.attrs['info'])
    tb_tool.close()

    # write each xda transposed to disk
    chunks = [rr[0] for rr in xds[portion].chunks][::-1]
    cshapes = xds[portion].shape[::-1]
    dims = xds[portion].dims[::-1]
    delayed_writes = []
    for ii, xda in enumerate(xda_list):
        for d0 in range(0, cshapes[0], chunks[0]):
            d0len = min(chunks[0], cshapes[0] - d0)

            for d1 in range(0, cshapes[1] if len(cshapes) > 1 else 1, chunks[1] if len(chunks) > 1 else 1):
                d1len = min(chunks[1], cshapes[1] - d1) if len(cshapes) > 1 else 0

                for d2 in range(0, cshapes[2] if len(cshapes) > 2 else 1, chunks[2] if len(chunks) > 2 else 1):
                    d2len = min(chunks[2], cshapes[2] - d2) if len(cshapes) > 2 else 0

                    for d3 in range(0, cshapes[3] if len(cshapes) > 3 else 1, chunks[3] if len(chunks) > 3 else 1):
                        d3len = min(chunks[3], cshapes[3] - d3) if len(cshapes) > 3 else 0

                        for d4 in range(0, cshapes[4] if len(cshapes) > 4 else 1, chunks[4] if len(chunks) > 4 else 1):
                            d4len = min(chunks[4], cshapes[4] - d4) if len(cshapes) > 4 else 0

                            starts = [d0, d1, d2, d3, d4][:len(cshapes)]
                            slices = [slice(d0, d0+d0len), slice(d1, d1+d1len), slice(d2, d2+d2len), slice(d3, d3+d3len), slice(d4, d4+d4len)]
                            txda = xda.transpose().isel(dict(zip(dims, slices)), missing_dims='ignore')
                            delayed_writes += [dask.delayed(write_image_slice)(txda, os.path.join(outfile, subtable_list[ii]), col=cols[ii], starts=starts)]

    if execute:
        if verbose: print('triggering DAG...')
        zs = dask.compute(delayed_writes)
        if verbose: print('execution time %0.2f sec' % (time.time() - start))
    else:
        if verbose: print('returning delayed task list')
        return delayed_writes
