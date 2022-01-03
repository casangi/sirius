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
from sirius_data._constants import c
import scipy.constants
from scipy.special import j1, jn
from numba import jit
import numba_scipy.special

# The twiddle factor is needed to improve agreement between CASA PBMATH airy disk and SiRIUS. 
# This arrises because CASA makes use of truncated constants. The twiddle factor 0.9998277835716939 is very close to 1. 
# Even with the twiddle there will not be machine precission agreement, because CASA calculates a 1D PB at 1 GHz for 10000 points and then frequency scales and interpolates to desired value.
casa_twiddle = (180*7.016*c)/((np.pi**2)*(10**9)*1.566*24.5) # 0.9998277835716939
r_vla_max = 0.8564*np.pi/180 #in radians

#@jit(nopython=True,cache=True,nogil=True)
@jit(nopython=True,nogil=True)
def _casa_airy_pb(lmn,freq_chan,dish_diameter, blockage_diameter, ipower, r_max=r_vla_max, n_sample=10000):
    #dish_diameter = pb_parms['dish_diameter']
    #blockage_diameter = pb_parms['blockage_diameter']
    #ipower = pb_parms['ipower']
    
    #print(lmn,freq_chan,dish_diameter, blockage_diameter, ipower, r_max, n_sample)
    
    k = (2*np.pi*freq_chan)/c
    aperture = dish_diameter/2

    if n_sample is not None:
        r = np.arcsin(np.sqrt(lmn[0]**2 + lmn[1]**2)) #CASA in PBMATH does a small angle approximation.
        r_inc = ((r_max)/(n_sample-1))
        r = (int(np.floor(r/r_inc + 0.5))*r_inc)*aperture*k
        r = r*casa_twiddle
    else:
        r = np.arcsin(np.sqrt(lmn[0]**2 + lmn[1]**2)*k*aperture)
        
    if (r != 0):
        if blockage_diameter==0.0:
            return (2.0*j1(r)/r)**ipower
        else:
            area_ratio = (dish_diameter/blockage_diameter)**2
            length_ratio = (dish_diameter/blockage_diameter)
            return ((area_ratio * 2.0 * j1(r)/r   - 2.0 * j1(r * length_ratio)/(r * length_ratio) )/(area_ratio - 1.0))**ipower
    else:
        return 1

#@jit(nopython=True,cache=True,nogil=True)
@jit(nopython=True,nogil=True)
def _airy_pb(lmn,freq_chan,dish_diameter, blockage_diameter, ipower):
    #print('lmn is',lmn)
    
    if (lmn[0] != 0) or (lmn[1] != 0):
        #dish_diameter = pb_parms['dish_diameter']
        #blockage_diameter = pb_parms['blockage_diameter']
        #ipower = pb_parms['ipower']
        
        k = (2*np.pi*freq_chan)/c
        
        aperture = dish_diameter/2
        r = np.sqrt(lmn[0]**2 + lmn[1]**2)*k*aperture
        
        if blockage_diameter==0.0:
            return (2.0*j1(r)/r)**ipower
        else:
            e = blockage_diameter/dish_diameter
            return (( 2.0 * j1(r)/r   - 2.0 * e * j1(r * e)/r )/(1.0 - e**2))**ipower
            #Changed r_grid to r ^^^
    else:
        return 1
    
    
    
#Non-jitted version:
def _casa_airy_pb_njit(lmn,freq_chan,pb_parms):
    #print('lmn is',lmn)
    
    if (lmn[0] != 0) or (lmn[1] != 0):
        dish_diameter = pb_parms['dish_diameter']
        blockage_diameter = pb_parms['blockage_diameter']
        ipower = pb_parms['ipower']
        
        k = (2*np.pi*freq_chan)/c
        
        aperture = dish_diameter/2
        r = np.sqrt(lmn[0]**2 + lmn[1]**2)*k*aperture
        
        if blockage_diameter==0.0:
            return (2.0*jn(1,r)/r)**ipower
        else:
            area_ratio = (dish_diameter/blockage_diameter)**2
            length_ratio = (dish_diameter/blockage_diameter)
            return ((area_ratio * 2.0 * jn(1,r)/r   - 2.0 * jn(1, r * length_ratio)/(r * length_ratio) )/(area_ratio - 1.0))**ipower
    else:
        return 1


def _airy_pb_njit(lmn,freq_chan,pb_parms):
    #print('lmn is',lmn)
    
    if (lmn[0] != 0) or (lmn[1] != 0):
        dish_diameter = pb_parms['dish_diameter']
        blockage_diameter = pb_parms['blockage_diameter']
        ipower = pb_parms['ipower']
        
        k = (2*np.pi*freq_chan)/c
        
        aperture = dish_diameter/2
        r = np.sqrt(lmn[0]**2 + lmn[1]**2)*k*aperture
        
        if blockage_diameter==0.0:
            return (2.0*jn(1,r)/r)**ipower
        else:
            e = blockage_diameter/dish_diameter
            return (( 2.0 * jn(1,r)/r   - 2.0 * e * jn(1, r * e)/r )/(1.0 - e**2))**ipower
    else:
        return 1
