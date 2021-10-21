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

#Do not modify
import numpy as np

map_mueler_to_pol = np.array([[0,0],[0,1],[1,0],[1,1],[0,2],[0,3],[1,2],[1,3],[2,0],[2,1],[3,0],[3,1],[2,2],[2,3],[3,2],[3,3]])

#https://github.com/casacore/casacore/blob/dbf28794ef446bbf4e6150653dbe404379a3c429/measures/Measures/Stokes.h
pol_codes_RL = np.array([5,6,7,8]) #'RR','RL','LR','LL'
pol_codes_XY = np.array([9,10,11,12]) #['XX','XY','YX','YY']
pol_str = np.array(['0','I','Q','U','V','RR','RL','LR','LL','XX','XY','YX','YY'])

k_B = 1.380649*10**-23 #Boltzmann constant J.K**-1
t_cmb = 2.725 #Cosmic Background Temperature
c = 299792458


'''
*****NB Note SiRIUS does not CGS units but MKS units****

https://open-bitbucket.nrao.edu/projects/CASA/repos/casa6/browse/casatools/src/python/constants.py
# %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
# IMPORTS
# %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&

import numpy as _numpy

# %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
# CGS PHYSICAL CONSTANTS
# %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&

c = 2.99792458e10       # speed of light CGS
h = 6.6260755e-27       # Planck's constant CGS
g = 6.67259e-8          # Grav const CGS
kb = 1.380658e-16       # Boltzmann's const CGS
a = 7.56591e-15         # Radiation constant CGS
sb = 5.67051e-5         # sigma (stefan-boltzmann const) CGS
qe =  4.803206e-10      # Charge of electron CGS
ev =  1.60217733e-12    # Electron volt CGS
na =  6.0221367e23      # Avagadro's Number
me =  9.1093897e-28     # electron mass CGS
mp =  1.6726231e-24     # proton mass CGS
mn = 1.674929e-24       # neutron mass CGS
mh = 1.673534e-24       # hydrogen mass CGS
amu =  1.6605402e-24    # atomic mass unit CGS

# %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
# ASTRONOMICAL CONSTANTS
# %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&

# GENERAL
au = 1.496e13           # astronomical unit CGS
pc = 3.0857e18          # parsec CGS
yr = 3.155815e7         # sidereal year CGS
ms = 1.98900e+33        # solar mass CGS
rs = 6.9599e10          # sun's radius CGS
ls = 3.839e33           # sun's luminosity CGS
mm = 7.35000e+25        # moon mass CGS
mer = 5.97400e+27       # earth mass CGS
rer = 6.378e8           # earth's radius CGS
medd = 3.60271e+34      # Eddington mass CGS

# RADIO SPECIFIC
jy = 1.e-23                  # Jansky CGS
restfreq_hi = 1420405751.786 # 21cm transition (Hz)
restfreq_co = 115271201800.  # CO J=1-0 (Hz)
cm2perkkms_hi = 1.823e18     # HI column per intensity (thin)

# OTHER WAVELENGTHS
ksun = 3.28             # abs K mag of sun (Binney & Merrifield)

# %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
# GEOMETRY
# %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&

srdeg = 3283.0          # Degrees squared in a steradian
dtor =  0.0174532925199 # Degrees per radian
pi = 3.14159265359      # Pi

# %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
# BASIC PHYSICAL FUNCTIONS
# %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&

def bb_inu(temp_k=None, lam_cm=None):
    """
    Planck function. Returns f_nu given T and lambda.
    """
    nu = c / lam_cm
    fnu = 2.0*h*nu**3/c**2 /(_numpy.exp(h*nu/(kb*temp_k))-1.0)
    return fnu, "erg/s/cm^2/sr/Hz"

def bb_ilam(temp_k=None, lam_cm=None):
    """
    Planck function. Returns f_lambda given T and lambda.
    """
    flam = 2.0*h*c**2/lam_cm**5 / (_numpy.exp(h*c/(lam_cm*kb*temp_k))-1.0)
    return (flam, "erg/s/cm^2/sr/cm")

def peak_inu(temp_k=None):
    """
    Wavelength for peak of F_nu given T.
    """
    peak_lam = 2.821439372/kb/temp_k/h
    peak_nu = c / peak_lam
    return (peak_lam, "cm")

def peak_ilam(temp_k=None):
    """
    Wavelength for peak of F_lambda given T.
    """
    return h*c/kb/temp_k/4.965114231, "cm"

# %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
# UNIT CONVERSIONS
# %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
'''
