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

@jit(nopython=True,cache=True,nogil=True)
def bilinear_interpolate(im, x, y):
    """Interpolates image values. 
    Inputs 
    -------------- 
    im: 2-d numpy array (complex?)
    x: 1-d numpy array of fractional indices (float)
    y: 1-d numpy array of fractional indices (float)
    Notes: x and y must be same length. Negative indices not allowed (automatically set to 0).
    -------------- 
    Outputs: 
    -------------- 
    1-d numpy array of interpolated values (float)"""
    
    #x0 rounds down, x1 rounds up. Likewise for y
    x0 = np.floor(x).astype(numba.intc)
    x1 = x0 + 1
    y0 = np.floor(y).astype(numba.intc)
    y1 = y0 + 1
    
    #Safety: makes sure no indices out of bounds
    x0 = np.minimum(im.shape[1]-1, np.maximum(x0, 0))
    x1 = np.minimum(im.shape[1]-1, np.maximum(x1, 0))
    y0 = np.minimum(im.shape[0]-1, np.maximum(y0, 0))
    y1 = np.minimum(im.shape[0]-1, np.maximum(y1, 0))
    
    #Four values around value to be interpolated
    Ia1 = im[y0]
    Ia = Ia1.flatten()[x0]
    Ib1 = im[y1]
    Ib = Ib1.flatten()[x0]
    Ic1 = im[y0]
    Ic = Ic1.flatten()[x1]
    Id1 = im[y1]
    Id = Id1.flatten()[x1]
    
    #See https://en.wikipedia.org/wiki/Bilinear_interpolation
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id


@jit(nopython=True,cache=True,nogil=True)
def interp_array(im_array, l, m, delta_l, delta_m):
    """Interpolates image values.
    Inputs 
    -------------- 
    im_array: 3-d numpy array of shape (pol, image, image)
    l: 1-d numpy array of fractional indices (float)
    m: 1-d numpy array of fractional indices (float)
    delta_l: pixel size for l coordinates (float)
    delta_m: pixel size for m coordinates (float)
    Notes: l and m must be same length.
    -------------- 
    Outputs: 
    -------------- 
    2-d numpy array of interpolated values (float) with shape (pol, len(l))"""
    #Length of image along l 
    n_l = len(im_array[0, :, 0]) #Change to shape?
    #Length of image along m
    n_m = len(im_array[0, 0, :]) 
    #Fractional pixel along l
    x_frac = (l/delta_l) + n_l//2 
    #Fractional pixel along m
    y_frac = (m/delta_m) + n_m//2 
    #Numba-style array creation. Shape is (polarization, coordinates)
    results = np.zeros((len(im_array), len(l)), dtype = numba.complex64) 
    for i in range(len(im_array)):
        #Complex interpolation
        results[i] = bilinear_interpolate(im_array[i].real, x_frac, y_frac) +    1j*bilinear_interpolate(im_array[i].imag, x_frac, y_frac) 
    return results


"""
@jit(nopython=True,cache=True,nogil=True)
def interp_ndim(ndim_array, x, y, dims = (0, 1)):
    Interpolates coordinates of an image defined by specified dimensions of an n-d array.
    Inputs:
    ndim_array: n-dimensional array
    dims: tuple of size 2 containing dimensions which comprise the image
    Outputs:
    Interpolated values
    #Gets the shape of the images which are to be interpolated
    shape = ndim_array.shape
    shape_image = np.zeros(2, dtype = int)
    for i in range(2):
        shape_image[i] = shape[dims[i]]
    
    #Gets the shape of the container of the images
    shape_image_array = np.delete(ndim_array.shape, dims)
    shape_image_tuple = (shape_image_array[0], )
    for i in range(1, len(shape_image_array)):
        shape_image_tuple = shape_image_tuple + (shape_image_array[i], )
    #print(shape_image_tuple)
    
    #Creates flattened container with images inside
    length_flat = 1
    for i in range(len(shape_image_array)):
        length_flat = length_flat*shape_image_array[i]
    flat_shape = np.zeros(1, dtype = int)
    flat_shape = (length_flat, shape_image[0], shape_image[1])
    f_image_array = ndim_array.reshape(flat_shape)
    
    #Creates container with results of interpolation as the innermost dimension
    results = np.zeros((len(f_image_array), len(x)), dtype = float)
    for i in range(len(f_image_array)):
        results[i] = bilinear_interpolate(f_image_array[i], x, y)
    shape_results = shape_image_tuple + (len(x), )
    results = results.reshape(shape_results)
    return results
            
"""

