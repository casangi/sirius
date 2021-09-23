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


def calc_a_noise(vis,eval_beam_models,a_noise_parms):
    """
    Add noise to visibilities.
    
    Parameters
    ----------
    vis : np.array
    Returns
    -------
    vis : np.array
    """
    noise = np.zeros(vis.shape,dtype=np.complex)
    
    dish_sizes = get_dish_sizes(eval_beam_models)
    print(dish_sizes)
    

def get_dish_sizes(eval_beam_models):
    dish_sizes = []
    for bm in eval_beam_models:
        if "J" in bm:
            dish_sizes.append(bm.attrs['dish_diam'])
        else:
            dish_sizes.append(bm['dish_diam'])
   
        
    return dish_sizes





