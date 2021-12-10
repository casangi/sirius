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

def _check_noise_parms(noise_parms):
    import numbers
    parms_passed = True
    from sirius_data._constants import t_cmb
    #https://casadocs.readthedocs.io/en/latest/api/tt/casatools.simulator.html#casatools.simulator.simulator.setnoise
    if not(_check_parms(noise_parms, 't_atmos', [numbers.Number], default = 250.0)): parms_passed = False
    
    print("Currently the effect of Zenith Atmospheric Opacity (Tau) is not included in the noise modeling.")
    if not(_check_parms(noise_parms, 'tau', [numbers.Number], default = 0.1)): parms_passed = False
    
    
    if not(_check_parms(noise_parms, 'ant_efficiency', [numbers.Number],default=0.8)): parms_passed = False
    if not(_check_parms(noise_parms, 'spill_efficiency', [numbers.Number],default=0.85)): parms_passed = False
    if not(_check_parms(noise_parms, 'corr_efficiency', [numbers.Number],default=0.88)): parms_passed = False
    if not(_check_parms(noise_parms, 't_receiver', [numbers.Number],default=50.0)): parms_passed = False
    if not(_check_parms(noise_parms, 't_ground', [numbers.Number],default=270.0)): parms_passed = False
    if not(_check_parms(noise_parms, 't_cmb', [numbers.Number],default=t_cmb)): parms_passed = False
    
    if not(_check_parms(noise_parms, 'auto_corr', [bool], default=False)): parms_passed = False
    if not(_check_parms(noise_parms, 'freq_resolution', [numbers.Number])): parms_passed = False
    if not(_check_parms(noise_parms, 'time_delta', [numbers.Number])): parms_passed = False
    return parms_passed
