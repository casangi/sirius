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


def _check_beam_parms(beam_parms):
    #{'zernike_freq_interp': 'nearest', 'pa_radius': 0.2, 'image_size': array([500, 500])}
    import numbers
    parms_passed = True
    arc_sec_to_rad = np.pi / (3600 * 180)
    
    if not(_check_parms(beam_parms, 'fov_scaling', [int],default=15)): parms_passed = False
    if not(_check_parms(beam_parms, 'mueller_selection', [list,np.array], list_acceptable_data_types=[np.int64], default = np.array([ 0, 5, 10, 15]),list_len=-1)): parms_passed = False
    if not(_check_parms(beam_parms, 'zernike_freq_interp', [str], default = 'nearest',acceptable_data=['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'])): parms_passed = False
    if not(_check_parms(beam_parms, 'pa_radius', [numbers.Number], default=0.2,acceptable_range=[0,2*np.pi])): parms_passed = False
    if not(_check_parms(beam_parms, 'image_size', [list,np.array], list_acceptable_data_types=[np.int64], list_len=2, default = np.array([1000,1000]))): parms_passed = False
    
    if parms_passed == True:
        beam_parms['image_size'] = np.array(beam_parms['image_size'])
        beam_parms['mueller_selection'] = np.array(beam_parms['mueller_selection'])

    return parms_passed
