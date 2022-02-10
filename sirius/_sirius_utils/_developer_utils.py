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

def _clear_pycache():
    '''
    Delete __pycache__ from sirius (useful for numba development).
    '''
    import sirius
    import os
    sirius_dir = os.path.dirname(sirius.__file__)
    os.system('rm -rf ' + sirius_dir + '/__pycache__')
    os.system('rm -rf ' + sirius_dir + '/_sirius_utils/__pycache__')
    os.system('rm -rf ' + sirius_dir + '/_parm_utils/__pycache__')
    os.system('rm -rf ' + sirius_dir.rsplit('/', 1)[0] + '/docs/__pycache__')
    print('__pycache__ removed from sirius.')
