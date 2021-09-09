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
from ._math_utils import _compute_rot_coords
import time
import itertools
from numba import jit
  
# Do we need the abs for apeture_parms['zernike_size'].
#    apeture_parms['cell_size'] = 1/(beam_parms['cell_size']*beam_parms['image_size']*apeture_parms['oversampling'])
#    apeture_parms['zernike_size'] = np.floor(np.abs(Dish_Diameter*eta/(apeture_parms['cell_size']*lmbd))) #Why is abs used?
#2.) Currently in ARD-20 the Zernike grid parameters are caculated using (in ZernikeCalc.cc):
#size_zernike = floor((D*eta)/dx)
#delta_zernike = 2.0/size_zernike
#
#By reversing the order the delta of the zernike grid and uv grid can match exactly (no floor operation). The zernike grid will be slightly larger (all cells outside the r=1 will still be nulled).
#delta_zernike = (2.0*dx)/(D*eta)
#size_zernike = ceil((D*eta)/dx)
#Assume ETA is the same for all pol and coef. Only a function of freq. Need to update zpc.zarr format.

def _calc_ant_jones(list_zpc_dataset,j_freq,j_pa,beam_parms):
    pa_prev = -42.0
    freq_prev = -42.0
    i_model_prev = -42
    c = 299792458
    
    
    j_planes = np.zeros((len(list_zpc_dataset),len(j_pa),len(j_freq),len(beam_parms['needed_pol']),beam_parms['image_size'][0],beam_parms['image_size'][1]),np.complex128)
    j_planes_shape = j_planes.shape
    iter_dims_indx = itertools.product(np.arange(j_planes_shape[0]), np.arange(j_planes_shape[1]),np.arange(j_planes_shape[2]))
    ic = beam_parms['image_size']//2 #image center pixel
    
    
    for i_model, i_pa, i_chan in iter_dims_indx:
        #print(i_model,i_pa,i_chan)
        pa = j_pa[i_pa]
        beam = list_zpc_dataset[i_model]
        freq = j_freq[i_chan]

        if (i_model != i_model_prev) or (freq != freq_prev):
            beam_interp = beam.interp(chan=freq,method=beam_parms['zernike_freq_interp'])

        dish_diam = beam.dish_diam
        lmbd = c/freq
        eta = beam_interp.ETA[0,0].values #Assume ETA is the same for all pol and coef. Only a function of freq. Need to update zpc.zarr format.
        uv_cell_size = 1/(beam_parms['cell_size']*beam_parms['image_size'])
        zernike_cell = (2.0*uv_cell_size*lmbd)/(dish_diam*eta)
        
        
        if (pa != pa_prev) or (freq != freq_prev) :
            beam_parms['parallactic_angle'] = pa
            image_size = (np.ceil(np.abs(2.0/zernike_cell))).astype(int)
            x_grid, y_grid = _compute_rot_coords(image_size,zernike_cell,pa)
            
            r_grid = np.sqrt(x_grid**2 + y_grid**2)
            
            zernike_size = np.array(x_grid.shape)
        
            ic_z = zernike_size//2
            include_last = (zernike_size%2).astype(int)
        
        #assert zernike_size[0] < beam_parms['conv_size'][0] and zernike_size[1] < gcf_parms['conv_size'][1], "The convolution size " + str(gcf_parms['conv_size']) +" is smaller than the aperture image " + zernike_size + " . Increase conv_size"
        
        start = time.time()
        for i_pol,pol in enumerate(beam_parms['needed_pol']):
            a = _generate_zernike_surface(beam_interp.ZC.data[pol,:].compute(),x_grid,y_grid)
            a[r_grid > 1] = 0
            j_planes[i_model, i_pa, i_chan,i_pol,ic[0]-ic_z[0]:ic[0]+ic_z[0]+include_last[0],ic[1]-ic_z[1]:ic[1]+ic_z[1]+include_last[1]] = a
            j_planes[i_model, i_pa, i_chan, i_pol,:,:] = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(j_planes[i_model, i_pa, i_chan,i_pol,:,:])))/(beam_parms['image_size'][0]*beam_parms['image_size'][1])
        #print('One pol set',time.time()-start)
        
        #Normalize Jones
        if 3 not in beam_parms['needed_pol']:
            P_max = np.abs(j_planes[i_model, i_pa, i_chan, np.where(beam_parms['needed_pol']==0),j_planes_shape[4]//2,j_planes_shape[5]//2])
            Q_max = P_max
        elif 0 not in beam_parms['needed_pol']:
            Q_max = np.abs(j_planes[i_model, i_pa, i_chan, np.where(beam_parms['needed_pol']==3),j_planes_shape[4]//2,j_planes_shape[5]//2])
            P_max = Q_max
        else:
            P_max = np.abs(j_planes[i_model, i_pa, i_chan, np.where(beam_parms['needed_pol']==0),j_planes_shape[4]//2,j_planes_shape[5]//2])
            Q_max = np.abs(j_planes[i_model, i_pa, i_chan, np.where(beam_parms['needed_pol']==3),j_planes_shape[4]//2,j_planes_shape[5]//2])

        j_planes[i_model, i_pa, i_chan,:,:,:] = j_planes[i_model, i_pa, i_chan,:,:,:]*2/(P_max+Q_max)
        
        pa_prev = pa
        freq_prev = freq
        i_model_prev = i_model
    return j_planes#np.zeros((1,4,2048,2048),dtype=np.complex128)
