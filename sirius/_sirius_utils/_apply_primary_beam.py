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
import matplotlib.pyplot as plt
import scipy.constants
from scipy.special import j1, jn
from numba import jit
import numba_scipy.special

#@jit(nopython=True,cache=True,nogil=True)
@jit(nopython=True,nogil=True)
def _apply_casa_airy_pb(lmn,freq_chan,dish_diameter, blockage_diameter, ipower):
    
    if (lmn[0] != 0) or (lmn[1] != 0):
        #dish_diameter = pb_parms['dish_diameter']
        #blockage_diameter = pb_parms['blockage_diameter']
        #ipower = pb_parms['ipower']
        
        c = scipy.constants.c #299792458
        k = (2*np.pi*freq_chan)/c
        
        aperture = dish_diameter/2
        r = np.sqrt(lmn[0]**2 + lmn[1]**2)*k*aperture
        #r = 2.0*np.arcsin(np.sqrt(np.sum(lmn**2))/2.0)*k*aperture # use lmn_rot in calc vis
        #print('r',r)
        
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
def _apply_airy_pb(lmn,freq_chan,dish_diameter, blockage_diameter, ipower):
    #print('lmn is',lmn)
    
    if (lmn[0] != 0) or (lmn[1] != 0):
        #dish_diameter = pb_parms['dish_diameter']
        #blockage_diameter = pb_parms['blockage_diameter']
        #ipower = pb_parms['ipower']
        
        c = scipy.constants.c #299792458
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
def apply_casa_airy_pb(lmn,freq_chan,pb_parms):
    #print('lmn is',lmn)
    
    if (lmn[0] != 0) or (lmn[1] != 0):
        dish_diameter = pb_parms['dish_diameter']
        blockage_diameter = pb_parms['blockage_diameter']
        ipower = pb_parms['ipower']
        
        c = scipy.constants.c #299792458
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


def apply_airy_pb(lmn,freq_chan,pb_parms):
    #print('lmn is',lmn)
    
    if (lmn[0] != 0) or (lmn[1] != 0):
        dish_diameter = pb_parms['dish_diameter']
        blockage_diameter = pb_parms['blockage_diameter']
        ipower = pb_parms['ipower']
        
        c = scipy.constants.c #299792458
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
