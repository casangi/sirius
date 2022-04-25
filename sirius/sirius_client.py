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

import warnings, time, os, psutil, multiprocessing, re
import dask
import copy
from ._parm_utils._check_logger_parms import _check_logger_parms, _check_worker_logger_parms
from sirius._sirius_utils._sirius_logger import _sirius_worker_logger_plugin, _setup_sirius_logger

def initialize_processing(cores=None, memory_limit=None, log_parms={}, worker_log_parms={}):
    '''
    https://github.com/dask/dask/issues/5577
    log_parms['log_to_term'] = True/False
    log_parms['log_file'] = True/False
    log_parms['level'] =
    
    '''
    _log_parms = copy.deepcopy(log_parms)
    _worker_log_parms = copy.deepcopy(worker_log_parms)
    
    assert(_check_logger_parms(_log_parms)), "######### ERROR: initialize_processing log_parms checking failed."
    assert(_check_worker_logger_parms(_worker_log_parms)), "######### ERROR: initialize_processing log_parms checking failed."

    # setup dask.distributed based multiprocessing environment
    if cores is None: cores = multiprocessing.cpu_count()
    if memory_limit is None: memory_limit = str(round(((psutil.virtual_memory().available / (1024 ** 2)) * 0.90) / cores)) + 'MB'
    dask.config.set({"distributed.scheduler.allowed-failures": 10})
    dask.config.set({"distributed.scheduler.work-stealing": False})
    dask.config.set({"distributed.scheduler.unknown-task-duration": '99m'})
    dask.config.set({"distributed.worker.memory.pause": False})
    dask.config.set({"distributed.worker.memory.terminate": False})
    #dask.config.set({"distributed.worker.memory.recent-to-old-time": '999s'})
    dask.config.set({"distributed.comm.timeouts.connect": '3600s'})
    dask.config.set({"distributed.comm.timeouts.tcp": '3600s'})
    dask.config.set({"distributed.nanny.environ.OMP_NUM_THREADS": 1})
    dask.config.set({"distributed.nanny.environ.MKL_NUM_THREADS": 1})
    cluster = dask.distributed.LocalCluster(n_workers=cores, threads_per_worker=1, processes=True, memory_limit=memory_limit) #, silence_logs=logging.ERROR
    #cluster = dask.distributed.LocalCluster(n_workers=1, threads_per_worker=24, processes=True, memory_limit='512GB')
    client = dask.distributed.Client(cluster)
    
    client.get_versions(check=True)
    from distributed.plugins.autorestrictor import install_plugin
    client.run_on_scheduler(install_plugin)
    
    import time
    start = time.time()
    #client.run(init_logging)
    #client.register_worker_callbacks()
    
    _setup_sirius_logger(log_to_term=_log_parms['log_to_term'],log_to_file=_log_parms['log_to_file'],log_file=_log_parms['log_file'], level=_log_parms['log_level'])
    #print('main logger time ', time.time()-start)
    
    
    #start = time.time()
    worker_logger = _sirius_worker_logger_plugin(_worker_log_parms)
    #print('2worker loggers time ', time.time()-start)
    
    #start = time.time()
    client.register_worker_plugin(plugin=worker_logger, name='sirius_worker_logger')
    #print('3worker loggers time ', time.time()-start)
    
    return client
