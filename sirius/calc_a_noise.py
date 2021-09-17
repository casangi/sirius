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
c = 299792458
from ._sirius_utils._direction_rotate import _calc_rotation_mats, _cs_calc_rotation_mats, _directional_cosine,  _sin_project
from ._sirius_utils._apply_primary_beam import _apply_casa_airy_pb, _apply_airy_pb, apply_casa_airy_pb, apply_airy_pb
from ._sirius_utils._math_utils import _interp_array, _rot_coord
from ._sirius_utils._array_utils import _find_angle_indx, _find_val_indx
import matplotlib.pyplot as plt
import time
from numba import jit
import numba
from ._sirius_utils._constants import map_mueler_to_pol

def calc_a_noise(vis,eval_beam_models,a_noise_parms):
    print('Hallo')
    
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
    



#def calc_a_noise_jit():
