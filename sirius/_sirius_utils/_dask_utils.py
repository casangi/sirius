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
import logging
import numpy as np

def _get_schedular_info():
    try:
        from dask.distributed import get_client
        scheduler_info = get_client().scheduler_info()
        n_workers = len(scheduler_info['workers'])

        memory_per_worker = np.zeros((n_workers,))
        n_threads_per_worker = np.zeros((n_workers,))
        for i,worker_key in enumerate(scheduler_info['workers']):
            memory_per_worker[i] = scheduler_info['workers'][worker_key]['memory_limit']
            n_threads_per_worker[i] = scheduler_info['workers'][worker_key]['nthreads']

        memory_per_thread = memory_per_worker/n_threads_per_worker
        min_memory_for_a_thread = min(memory_per_thread)
        
        logging.info('Getting info for schedular ' + scheduler_info['address'] + ' with dashboard port ' + str(scheduler_info['services']['dashboard']))
        logging.info('min_memory_for_a_thread ' + str(min_memory_for_a_thread))
        return min_memory_for_a_thread, memory_per_worker, n_threads_per_worker, n_workers
    except Exception as e:
        logging.error('Failed to get schedular info: '+ str(e))
    


'''
import dask
import dask.distributed

print(dask.config.get("distributed.client"))
print('*******')
print(dask.config.config)
'''

