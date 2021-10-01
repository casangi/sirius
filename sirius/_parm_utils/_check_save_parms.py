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

#ducting - code is complex and might fail after some time if parameters is wrong. Sensable values are also checked. Gives printout of all wrong parameters. Dirty images alone has 14 parameters.

import numpy as np
from  ._check_parms import _check_parms, _check_dataset


def _check_save_parms(save_parms):
    #{'zernike_freq_interp': 'nearest', 'pa_radius': 0.2, 'image_size': array([500, 500])}
    import numbers
    parms_passed = True

    if not(_check_parms(save_parms, 'write_to_ms', [bool],default=False)): parms_passed = False
    if not(_check_parms(save_parms, 'DAG_name_vis_uvw_gen', [str],default=False)): parms_passed = False
    if not(_check_parms(save_parms, 'DAG_name_write', [str],default=False)): parms_passed = False
    if not(_check_parms(save_parms, 'ms_name', [str],default='sirius_sim.ms')): parms_passed = False
    
    return parms_passed
