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

from numba import jit
import numba
import numpy as np


def _is_subset(arr1, arr2):
    '''
    Is arr2 a subset of arr1.
    '''
    for a in arr2:
        if a not in arr1:
            return False
    return True

def _ndim_list(shape):
    return [_ndim_list(shape[1:]) if len(shape) > 1 else None for _ in range(shape[0])]

#@jit(nopython=True,cache=True,nogil=True)
def _calc_baseline_indx_pair(n_ant,auto_corr=False):
    if auto_corr:
        n_baseline = int((n_ant**2 + n_ant)/2)
        a = 0
    else:
        n_baseline = int((n_ant**2 - n_ant)/2)
        a = 1
        
    antenna1 = np.zeros((n_baseline,),dtype=np.int)
    antenna2 = np.zeros((n_baseline,),dtype=np.int)
    
    k = 0
    for i in range(n_ant-a):
        for j in range(i+a,n_ant):
            antenna1[k] = i
            antenna2[k] = j
            k=k+1
            
    return antenna1, antenna2
    
def _calc_n_baseline(n_ant,auto_corr):
    if auto_corr:
        return int((n_ant**2 + n_ant)/2)
    else:
        return int((n_ant**2 - n_ant)/2)


#@jit("void(i8,i8,i8,i8,i8[:])",nopython=True,cache=True,nogil=True)
@jit(nopython=True,cache=True,nogil=True)
def _combine_indx_permutation(i1,i2,n_i1,n_i2):
    n_comb = n_i1*n_i2
    if n_i1 <= n_i2:
        i_comb = i1*n_i2 + i2
    else:
        i_comb = i1 + i2*n_i1
    
    #result[0] = i_comb
    #result[1] = n_comb
    #print(i1,i2,n_i1,n_i2,i_comb,n_comb)
    return i_comb,n_comb
    
#@jit("void(i8,i8,i8,i8,i8[:])",nopython=True,cache=True,nogil=True)
@jit(nopython=True,cache=True,nogil=True)
def _combine_indx_combination(i1,i2,n_i1,n_i2):
    if n_i1 <= n_i2:
        if i1 > i2:
            temp = i2
            i2 = i1
            i1 = temp
    
        n_comb = n_i1*n_i2 - (n_i1-1)*n_i1//2
        i_comb = ((2*n_i2 -1)*i1 - i1**2)//2 + i2
    else:
        if i1 < i2:
            temp = i2
            i2 = i1
            i1 = temp
    
        n_comb = n_i1*n_i2 - (n_i2-1)*n_i2//2
        i_comb = i1 + ((2*n_i1 -1)*i2 - i2**2)//2
        
    #print(i1,i2,n_i1,n_i2,i_comb,n_comb)
    return i_comb,n_comb

#The next set of functions finds the index of the value in a lists (val_list) that lies closest to a given val. These functions are needed for numba jit code.
@jit(nopython=True,cache=True,nogil=True)
def _find_val_indx(val_list,val):
    min_dif = -42.0 #Dummy value
    for jj in range(len(val_list)):
        ang_dif = np.abs(val-val_list[jj])

        if (min_dif < 0) or (min_dif > ang_dif):
            min_dif = ang_dif
            val_list_indx = jj
            
    return val_list_indx

#ang_list and ang must have the same anfle def. For example [-pi,pi] or [0,2pi]
@jit(nopython=True,cache=True,nogil=True)
def _find_angle_indx(ang_list,ang):
    min_dif = 42.0 #Dummy value
    for jj in range(len(ang_list)):
        #https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
        ang_dif = ang-ang_list[jj]
        ang_dif = np.abs((ang_dif + np.pi)%(2*np.pi) - np.pi)
        
        if min_dif > ang_dif:
            min_dif = ang_dif
            ang_list_indx = jj
            
    return ang_list_indx
    
#Uses an approximation for the distance between two pointings
@jit(nopython=True,cache=True,nogil=True)
def _find_ra_dec_indx(point_list,point):
    min_dis = 42.0 #Dummy value
    for jj in range(len(point_list)):
        #https://stjerneskinn.com/angular-distance-between-stars.htm
        #http://spiff.rit.edu/classes/phys373/lectures/radec/radec.html
        ra = point_list[jj,0]
        dec = point_list[jj,1]
        dis = np.sqrt(((ra-point[0])*np.cos(dec))**2 + (dec-point[1])**2) #approximation
        
        if min_dis > dis:
            min_dis = dis
            point_list_indx = jj

    return point_list_indx


