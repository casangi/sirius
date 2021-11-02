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

#Get 1GHz max rad values from https://open-bitbucket.nrao.edu/projects/CASA/repos/casa6/browse/casatools/src/code/synthesis/TransformMachines/PBMath.cc
#Values is in degrees. Convert to radians.
import numpy as np

vla =  {'pb_func':'casa_airy','dish_diam':24.5,'blockage_diam':0.0,'max_rad_1GHz':0.8564*np.pi/180}
aca =  {'pb_func':'casa_airy','dish_diam':6.25,'blockage_diam':0.75,'max_rad_1GHz':3.568*np.pi/180}
alma =  {'pb_func':'casa_airy','dish_diam':10.7,'blockage_diam':0.75,'max_rad_1GHz':1.784*np.pi/180}
ngvla = {'pb_func':'casa_airy','dish_diam':18.0,'blockage_diam':0.0,'max_rad_1GHz':1.5*np.pi/180}