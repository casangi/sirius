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

#ducting - code is complex and might fail after some time if parameters is wrong (time waisting). Sensable values are also checked. Gives printout of all wrong parameters. Dirty images alone has x parametrs.

import numpy as np
import xarray as xr
import dask.array as da
from ._zernike_polynomials import _generate_zernike_surface
import time
import itertools
  
# Do we need the abs for apeture_parms['zernike_size'].
#    apeture_parms['cell_size'] = 1/(grid_parms['cell_size']*grid_parms['image_size']*apeture_parms['oversampling'])
#    apeture_parms['zernike_size'] = np.floor(np.abs(Dish_Diameter*eta/(apeture_parms['cell_size']*lmbd))) #Why is abs used?
#2.) Currently in ARD-20 the Zernike grid parameters are caculated using (in ZernikeCalc.cc):
#size_zernike = floor((D*eta)/dx)
#delta_zernike = 2.0/size_zernike
#
#By reversing the order the delta of the zernike grid and uv grid can match exactly (no floor operation). The zernike grid will be slightly larger (all cells outside the r=1 will still be nulled).
#delta_zernike = (2.0*dx)/(D*eta)
#size_zernike = ceil((D*eta)/dx)
#Assume ETA is the same for all pol and coef. Only a function of freq. Need to update zpc.zarr format.

def _calc_ant_jones(list_zpc_dataset,j_freq,j_pa,pb_parms,grid_parms):
    pa_prev = -42.0
    freq_prev = -42.0
    i_model_prev = -42
    c = 299792458
    
    j_planes = np.zeros((len(list_zpc_dataset),len(j_pa),len(j_freq),len(pb_parms['needed_pol']),grid_parms['image_size'][0],grid_parms['image_size'][1]),np.complex128)
    j_planes_shape = j_planes.shape
    iter_dims_indx = itertools.product(np.arange(j_planes_shape[0]), np.arange(j_planes_shape[1]),np.arange(j_planes_shape[2]))
    ic = grid_parms['image_size']//2 #image center pixel
    
    
    for i_model, i_pa, i_chan in iter_dims_indx:
        #print(i_model,i_pa,i_chan)
        pa = j_pa[i_pa]
        beam = list_zpc_dataset[i_model]
        freq = j_freq[i_chan]

        if (i_model != i_model_prev) or (freq != freq_prev):
            beam_interp = beam.interp(chan=freq,method=pb_parms['zernike_freq_interp'])

        dish_diam = beam.dish_diam
        lmbd = c/freq
        eta = beam_interp.ETA[0,0].values #Assume ETA is the same for all pol and coef. Only a function of freq. Need to update zpc.zarr format.
        uv_cell_size = 1/(grid_parms['cell_size']*grid_parms['image_size'])
        zernike_cell = (2.0*uv_cell_size*lmbd)/(dish_diam*eta)
        
        
        if (pa != pa_prev) or (freq != freq_prev) :
            pb_parms['parallactic_angle'] = pa
            image_size = (np.ceil(np.abs(2.0/zernike_cell))).astype(int)
            x_grid, y_grid = _compute_rot_coords(image_size,zernike_cell,pa)
            
            r_grid = np.sqrt(x_grid**2 + y_grid**2)
            
            zernike_size = np.array(x_grid.shape)
        
            ic_z = zernike_size//2
            include_last = (zernike_size%2).astype(int)
        
        #assert zernike_size[0] < pb_parms['conv_size'][0] and zernike_size[1] < gcf_parms['conv_size'][1], "The convolution size " + str(gcf_parms['conv_size']) +" is smaller than the aperture image " + zernike_size + " . Increase conv_size"
        
        start = time.time()
        for i_pol,pol in enumerate(pb_parms['needed_pol']):
            a = _generate_zernike_surface(beam_interp.ZC.data[pol,:].compute(),x_grid,y_grid)
            a[r_grid > 1] = 0
            j_planes[i_model, i_pa, i_chan,i_pol,ic[0]-ic_z[0]:ic[0]+ic_z[0]+include_last[0],ic[1]-ic_z[1]:ic[1]+ic_z[1]+include_last[1]] = a
            j_planes[i_model, i_pa, i_chan, i_pol,:,:] = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(j_planes[i_model, i_pa, i_chan,i_pol,:,:])))/(grid_parms['image_size'][0]*grid_parms['image_size'][1])
        #print('One pol set',time.time()-start)
        
        #Normalize Jones
        if 3 not in pb_parms['needed_pol']:
            P_max = np.abs(j_planes[i_model, i_pa, i_chan, np.where(pb_parms['needed_pol']==0),j_planes_shape[4]//2,j_planes_shape[5]//2])
            Q_max = P_max
        elif 0 not in pb_parms['needed_pol']:
            Q_max = np.abs(j_planes[i_model, i_pa, i_chan, np.where(pb_parms['needed_pol']==3),j_planes_shape[4]//2,j_planes_shape[5]//2])
            P_max = Q_max
        else:
            P_max = np.abs(j_planes[i_model, i_pa, i_chan, np.where(pb_parms['needed_pol']==0),j_planes_shape[4]//2,j_planes_shape[5]//2])
            Q_max = np.abs(j_planes[i_model, i_pa, i_chan, np.where(pb_parms['needed_pol']==3),j_planes_shape[4]//2,j_planes_shape[5]//2])

        j_planes[i_model, i_pa, i_chan,:,:,:] = j_planes[i_model, i_pa, i_chan,:,:,:]*2/(P_max+Q_max)
        
        pa_prev = pa
        freq_prev = freq
        i_model_prev = i_model
    return j_planes#np.zeros((1,4,2048,2048),dtype=np.complex128)
    

def _compute_rot_coords(image_size,cell_size,parallactic_angle):
    image_center = image_size//2
    #print(image_size)

    x = np.arange(-image_center[0], image_size[0]-image_center[0])*cell_size[0]
    y = np.arange(-image_center[1], image_size[1]-image_center[1])*cell_size[1]
    xy = np.array([x,y]).T
    x_grid, y_grid = np.meshgrid(x,y,indexing='ij')
    
    if parallactic_angle != 0:
        rot_mat = np.array([[np.cos(parallactic_angle),-np.sin(parallactic_angle)],[np.sin(parallactic_angle),np.cos(parallactic_angle)]]) #anti clockwise
        
        #r = np.einsum('ji, mni -> jmn', rot_mat, np.dstack([x_grid, y_grid]))
        '''
        x_grid_rot = np.cos(parallactic_angle)*x_grid - np.sin(parallactic_angle)*y_grid
        y_grid_rot = np.sin(parallactic_angle)*x_grid + np.cos(parallactic_angle)*y_grid
        '''
        x_grid_rot = np.cos(parallactic_angle)*x_grid + np.sin(parallactic_angle)*y_grid
        y_grid_rot = - np.sin(parallactic_angle)*x_grid + np.cos(parallactic_angle)*y_grid
        
        x_grid = x_grid_rot
        y_grid = y_grid_rot
    
    return x_grid, y_grid








##################Currently Not used functions##################
#@jit(nopython=True,cache=True)
def _outer_product(B1,B2,norm,conj):
    '''
    Input
    B1 2 x 2 x m x n array
    B2 2 x 2 x m x n array
    Output
    M 4 x 4 x m x n
    '''
    
    #assert B1.shape==B2.shape
    
    s = B1.shape
    
    M = np.zeros((4,4,s[2],s[3]),dtype=np.complex128)
    
    indx_b1 = np.array([[[0,0],[0,0],[0,1],[0,1]],[[0,0],[0,0],[0,1],[0,1]],[[1,0],[1,0],[1,1],[1,1]],[[1,0],[1,0],[1,1],[1,1]]])
    indx_b2 = np.array([[[0,0],[0,1],[0,0],[0,1]],[[1,0],[1,1],[1,0],[1,1]],[[0,0],[0,1],[0,0],[0,1]],[[1,0],[1,1],[1,0],[1,1]]])
    #print(indx_b1.shape)
    
    
    for i in range(4):
        for j in range(4):
            #print(indx_b1[i,j,:], ',*,', indx_b2[i,j,:])
            if conj:
                M[i,j,:,:] = B1[indx_b1[i,j,0],indx_b1[i,j,1],:,:] * B2[indx_b2[i,j,0],indx_b2[i,j,1],:,:].conj().T
            else:
                M[i,j,:,:] = B1[indx_b1[i,j,0],indx_b1[i,j,1],:,:] * B2[indx_b2[i,j,0],indx_b2[i,j,1],:,:]
                
            if norm:
                M[i,j,:,:] = M[i,j,:,:]/np.max(np.abs(M[i,j,:,:]))
    
    #print(M.shape)
    return(M)


def _outer_product_conv(B1,B2):
    
#    Input
#    B1 2 x 2 x m x n array
#    B2 2 x 2 x m x n array
#    Output
#    M 4 x 4 x m x n
    
    #assert B1.shape==B2.shape
    
    s = B1.shape
    
    M = np.zeros((4,4,s[2],s[3]),dtype=np.complex128)
    
    indx_b1 = np.array([[[0,0],[0,0],[0,1],[0,1]],[[0,0],[0,0],[0,1],[0,1]],[[1,0],[1,0],[1,1],[1,1]],[[1,0],[1,0],[1,1],[1,1]]])
    indx_b2 = np.array([[[0,0],[0,1],[0,0],[0,1]],[[1,0],[1,1],[1,0],[1,1]],[[0,0],[0,1],[0,0],[0,1]],[[1,0],[1,1],[1,0],[1,1]]])
    
    for i in range(4):
        for j in range(4):
            M[i,j,:,:] = signal.fftconvolve(B1[indx_b1[i,j,0],indx_b1[i,j,1],:,:], B2[indx_b2[i,j,0],indx_b2[i,j,1],:,:],mode='same')
    
    print(M.shape)
    return(M)

    
def _make_flat(B):
    '''
    B 2x2xmxn
    B_flat 2mx2n
    '''
    s = B.shape
    B_flat = np.zeros((s[2]*s[0],s[3]*s[1]),dtype=complex)
    
    
    for i in range(s[0]):
        for j in range(s[1]):
            i_start = i*s[2]
            i_end = (i+1)*s[3]
            j_start = j*s[2]
            j_end = (j+1)*s[3]
            B_flat[i_start:i_end,j_start:j_end] = B[i,j,:,:]
            #print(B[i,j,1024,1024],np.abs(B[i,j,1024,1024]))
    return B_flat
    
    
def _make_flat_casa(B):
    '''
    B mxnx16
    B_flat 4mx4n
    '''
    s = B.shape
    B_flat = np.zeros((s[0]*4,s[1]*4),dtype=complex)
    
    #indx = np.array([[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2],[0,3],[1,3],[2,3],[3,3]])
    indx = np.array([[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]]) #saved as rows
    
    for c,i in enumerate(indx):
        #print(c,i)
        i_start = i[0]*s[0]
        i_end = (i[0]+1)*s[0]
        j_start = i[1]*s[1]
        j_end = (i[1]+1)*s[1]
        B_flat[i_start:i_end,j_start:j_end] = B[:,:,c].T
        #print(B[1024,1024,c],np.abs(B[1024,1024,c]))
    return B_flat
