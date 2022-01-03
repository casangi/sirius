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
import itertools
import time
import xarray as xr
import dask.array as da
import numba
from numba import jit
from sirius._sirius_utils._primary_beam_funcs import _casa_airy_pb, _airy_pb, _casa_airy_pb_njit, _airy_pb_njit
from sirius._sirius_utils._math_utils import _interp_array
from sirius._sirius_utils._coord_transforms import  _rot_coord, _compute_rot_coords
from sirius._sirius_utils._zernike_polynomials import _generate_zernike_surface
from sirius._sirius_utils._array_utils import _find_angle_indx, _find_val_indx
from sirius_data._constants import map_mueler_to_pol, c


def _calc_resolution(pb_freq,dish_diameter,beam_parms):
    fov = beam_parms['fov_scaling']*(1.22 * c / (dish_diameter*pb_freq))
    min_delta = min(min(fov/beam_parms['image_size'][0]),min(fov/beam_parms['image_size'][1]))
    return min_delta

def _beam_models_to_tuple(beam_models,beam_model_map):
    new_beam_model_map = np.zeros(beam_model_map.shape,dtype=np.int)
    beam_types = np.zeros(beam_model_map.shape,dtype=np.int)
    beam_models_list_type0 = []
    beam_models_list_type1 = []
    
    i_t0 = 0
    i_t1 = 0
    for i,bm in enumerate(beam_models):
        if "J" in bm:
            beam_models_list_type0.append(_exstract_arrays_from_bm_xds(bm))
            beam_types[beam_model_map==i] = 0
            new_beam_model_map[beam_model_map==i] = i_t0
            i_t0 = i_t0 + 1
        else:
            beam_models_list_type1.append(_exstract_vals_from_analytic_dict(bm))
            beam_types[beam_model_map==i] = 1
            new_beam_model_map[beam_model_map==i] = i_t1
            i_t1 = i_t1 + 1
            
    #Dummy for sample_j
    if not bool(beam_models_list_type0):
        d_arr_float = np.array([0.])
        d_arr_int = np.array([0])
        d_J = np.zeros((1,1,1,1,1),np.complex128)
        beam_models_list_type0.append((0,d_J,d_arr_float,d_arr_float,
                                        d_arr_int,0.0,0.0,0.0))
    
    #Dummy for sample_j_analytic
    if not bool(beam_models_list_type1):
        beam_models_list_type1.append((1,"none",0.,0.,0.0))
    
    return tuple(beam_models_list_type0), tuple(beam_models_list_type1), tuple(beam_types), tuple(new_beam_model_map)


def _exstract_arrays_from_bm_xds(bm):
    bm_J = bm.J.values 
    #print("bm_J", bm_J.shape)
    pa = bm.pa.values
    chan = bm.chan.values
    pol = bm.pol.values
    delta_l = (bm.l[1].values - bm.l[0].values).astype(float)
    delta_m = (bm.m[1].values - bm.m[0].values).astype(float)
    max_rad_1GHz = bm.attrs['max_rad_1GHz']
    return (0,bm_J,pa,chan,pol,delta_l,delta_m,max_rad_1GHz)
    
def _exstract_vals_from_analytic_dict(bm):
    pb_func = bm['pb_func']
    #dish_diameter = bm['dish_diameter']
    #blockage_diameter = bm['blockage_diameter']
    dish_diameter = bm['dish_diam']
    blockage_diameter = bm['blockage_diam']
    max_rad_1GHz = bm['max_rad_1GHz']
    return (1,pb_func,dish_diameter,blockage_diameter,max_rad_1GHz)


#@jit(nopython=True,cache=True,nogil=True)
@jit(nopython=True,nogil=True)
def _calc_pb_scale(flux, sep1, sep2, bm1_indx,bm2_indx,bm1_type,bm2_type,lmn1,lmn2,beam_models_type0,beam_models_type1,pa,freq,mueller_selection,do_pointing):

    #print(sep1,sep2,bm1_indx,bm2_indx,bm1_type,bm2_type)
    outside_beam = False
    if (bm1_indx == bm2_indx) and (bm1_type == bm2_type) and ~do_pointing: #Antennas are the same
        if bm1_type == 0: #Checks if it is a zernike model
            bm1 = beam_models_type0[bm1_indx]
            max_rad = bm1[7]/(freq/10**9) # scale max_rad_1GHz to freq
            
            if sep1 < max_rad:
                #bm1 = beam_models_type0[bm1_indx]
                J_sampled = _sample_J(bm1[1],bm1[2],bm1[3],bm1[4],bm1[5],bm1[6],pa,freq,lmn1)[:,0]
                
                #J_sampled = np.zeros((4,),dtype=numba.complex128)
                M = _make_mueler_mat(J_sampled, J_sampled, mueller_selection)
                #Add check if J sampled is < 0 and then skip this
                flux_scaled = np.dot(M,flux)
            else:
                outside_beam = True
        else:
            #bm1 = beam_models_type1[bm1_indx]
            bm1 = beam_models_type1[bm1_indx]
            max_rad = bm1[4]/(freq/10**9) # scale max_rad_1GHz to freq
            if sep1 < max_rad:
                J_sampled = _sample_J_analytic(bm1[1],bm1[2],bm1[3],max_rad,lmn1,freq,1)
                #print('J_sampled',J_sampled,bm1[1],bm1[2],bm1[3],lmn1,freq,1)
                flux_scaled = flux*J_sampled**2
            else:
                outside_beam = True
    else:
        if bm1_type == 0:
            bm1 = beam_models_type0[bm1_indx]
            max_rad = bm1[7]/(freq/10**9) # scale max_rad_1GHz to freq
            if sep1 < max_rad:
                J_sampled1 = _sample_J(bm1[1],bm1[2],bm1[3],bm1[4],bm1[5],bm1[6],pa,freq,lmn1)[:,0]
            else:
                outside_beam = True
        else:
            bm1 = beam_models_type1[bm1_indx]
            max_rad = bm1[4]/(freq/10**9) # scale max_rad_1GHz to freq
            if sep1 < max_rad:
                J_sampled1 = _sample_J_analytic(bm1[1],bm1[2],bm1[3],max_rad,lmn1,freq,1)
            else:
                outside_beam = True
        if bm2_type == 0:
            bm2 = beam_models_type0[bm2_indx]
            max_rad = bm2[7]/(freq/10**9) # scale max_rad_1GHz to freq
            if sep2 < max_rad:
                J_sampled2 = _sample_J(bm2[1],bm2[2],bm2[3],bm2[4],bm2[5],bm2[6],pa,freq,lmn2)[:,0]
            else:
                outside_beam =  True
        else:
            bm2 = beam_models_type1[bm2_indx]
            max_rad = bm2[4]/(freq/10**9) # scale max_rad_1GHz to freq
            if sep2 < max_rad:
                J_sampled2 = _sample_J_analytic(bm2[1],bm2[2],bm2[3],max_rad,lmn2,freq,1)
            else:
                outside_beam = True
                
        if not outside_beam:
            M = _make_mueler_mat(J_sampled1, J_sampled2, mueller_selection)
            flux_scaled = np.dot(M,flux)
    if outside_beam:
        flux_scaled = np.zeros(4, dtype = numba.complex128)

    return flux_scaled, outside_beam

@jit(nopython=True,cache=True,nogil=True)
def _pol_code_to_index(pol):
    if pol[0] in [5,6,7,8]:
        return pol-5
    if pol[0] in [9,10,11,12]:
        return pol-9
    assert False # "Unsupported pol " + str(pol) #Numba does not support an explicit error message.
    
#@jit(nopython=True,cache=True,nogil=True)
@jit(nopython=True,nogil=True)
def _sample_J_analytic(pb_func, dish_diameter,blockage_diameter,max_rad, lmn, freq, ipower):
    #pb_parms = bm
    #pb_parms['ipower'] = 1
    
    if pb_func == 'casa_airy':
        J_sampled = _casa_airy_pb(lmn,freq,dish_diameter, blockage_diameter,ipower,max_rad)
        #J_sampled = 0.5
    elif pb_func == 'airy':
        #J_sampled = 0.5
        J_sampled = _airy_pb(lmn,freq,dish_diameter, blockage_diameter, ipower)
    else:
        J_sampled = 1
    
    J_sampled_array = np.zeros(4, dtype = numba.float64)
    J_sampled_array[0] = J_sampled
    J_sampled_array[3] = J_sampled
    #J_sampled = np.array([J_sampled,0,0,J_sampled])
    return J_sampled_array.astype(numba.complex128)
    
@jit(nopython=True,cache=True,nogil=True)
def _sample_J(bm_J,bm_pa,bm_chan, bm_pol, bm_delta_l,bm_delta_m,pa,freq,lmn):
    pa_indx = _find_angle_indx(bm_pa,pa)
    chan_indx = _find_val_indx(bm_chan, freq)
    bm_J_sub = bm_J[pa_indx, chan_indx]

    x_rot, y_rot  = _rot_coord(lmn[0],lmn[1],pa-bm_pa[pa_indx])
    xrot = np.ones(1, dtype = numba.float64)
    xrot[0] = x_rot
    yrot = np.ones(1, dtype = numba.float64)
    yrot[0] = y_rot
    #print((xrot/bm[4]) + len(bm_J_sub[0, :, 0])//2)
    #print((yrot/bm[5]) + len(bm_J_sub[0, 0, :])//2)
    
    J_temp = _interp_array(bm_J_sub, xrot, yrot, bm_delta_l, bm_delta_m)
    
    J = np.zeros((4,J_temp.shape[1]),dtype=numba.complex128) + 1
    for i,p in enumerate(bm_pol):
        J[p] = J_temp[i,:]
        
    return J

#Non-numba versions:
def _sample_J_analytic_og(bm,lmn,freq):
    pb_parms = bm
    pb_parms['ipower'] = 1
    
    if pb_parms['pb_func'] == 'casa_airy':
        J_sampled = _casa_airy_pb_njit(lmn,freq,pb_parms)
    elif pb_parms['pb_func'] == 'airy':
        J_sampled = _airy_pb_njit(lmn,freq,pb_parms)
    else:
        J_sampled = 1
    J_sampled = np.array([J_sampled,0,0,J_sampled])
    return J_sampled


def _sample_J_og(bm,lmn,freq,pa):
# sample_J(J,pa_,chan_,lmn,chan,pa)
    bm_sub = bm
    if len(bm.pa) > 1:
        pa_indx= _find_angle_indx(bm_sub.pa.values,pa)
        bm_sub = bm_sub.isel(pa=pa_indx)
    else:
        bm_sub = bm_sub.isel(pa=0)
    if len(bm.chan) > 1:
        bm_sub = bm.J.interp(chan=freq,method='nearest')
    else:
        bm_sub = bm_sub.isel(chan=0)
        
    #print('pa values',bm_sub)
    x_rot, y_rot  = _rot_coord(lmn[0],lmn[1],pa-bm_sub.pa.values)
    
#    print(bm_sub)
#    print(lmn,pa,bm_sub.pa.values )
#    print(x_rot,y_rot)
#    print(bm_sub.J.isel(model=0).interp(l=x_rot,m=y_rot,method='linear'))
#    print(bm_sub.J.interp(l=x_rot,m=y_rot,method='linear').values)
    
    return bm_sub.J.interp(l=x_rot,m=y_rot,method='linear').values[0]

@jit(nopython=True,cache=True,nogil=True)
def _make_mueler_mat(J1, J2, mueller_selection):

    #M = np.zeros((4,4),dtype=np.complex)
    M = np.zeros((4,4),dtype=numba.complex128)
    
    for m_flat_indx in mueller_selection:
        #print(m_flat_indx//4,m_flat_indx - 4*(m_flat_indx//4))
        #print(np.where(map_mueler_to_pol[m_flat_indx][0] == pol)[0][0])
        #print(pol, map_mueler_to_pol[m_flat_indx][0])
        #M[m_flat_indx//4,m_flat_indx - 4*(m_flat_indx//4)] = J1[np.where(map_mueler_to_pol[m_flat_indx][0] == pol)[0][0]]*np.conj(J2[np.where(map_mueler_to_pol[m_flat_indx][1] == pol)[0][0]])
        
        M[m_flat_indx//4,m_flat_indx - 4*(m_flat_indx//4)] = J1[map_mueler_to_pol[m_flat_indx,0]]*np.conj(J2[map_mueler_to_pol[m_flat_indx,1]])
            
    return M
    
    #map_mueler_to_pol = np.array([[0,0],[0,1],[1,0],[1,1],[0,2],[0,3],[1,2],[1,3],[2,0],[2,1],[3,0],[3,1],[2,2],[2,3],[3,2],[3,3]])

'''
    #print('pol',pol)
    if inv:
        map_mueler_to_pol = np.array([[3, 3],[3, 2],[2, 3],[2, 2],[3, 1],[3, 0],[2, 1],[2, 0],[1, 3],[1, 2],[0, 3],[0, 2],[1, 1],[1, 0],[0, 1],[0, 0]]) # np.flip(map_mueler_to_pol,axis=0)
        #map_mueler_to_pol = np.array([ [[3, 3],[3, 2],[2, 3],[2, 2]],[[3, 1],[3, 0],[2, 1],[2, 0]],[[1, 3],[1, 2],[0, 3],[0, 2]],[[1, 1],[1, 0],[0, 1],[0, 0]]])
    else:
        map_mueler_to_pol = np.array([[0,0],[0,1],[1,0],[1,1],[0,2],[0,3],[1,2],[1,3],[2,0],[2,1],[3,0],[3,1],[2,2],[2,3],[3,2],[3,3]])
        #map_mueler_to_pol = np.array([[[0,0],[0,1],[1,0],[1,1]],[[0,2],[0,3],[1,2],[1,3]],[[2,0],[2,1],[3,0],[3,1]],[[2,2],[2,3],[3,2],[3,3]]])
        
'''

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
  
def _calc_ant_jones(zpc_dataset,j_freq,j_pa,beam_parms):
    pa_prev = -42.0
    freq_prev = -42.0
    
    j_planes = np.zeros((len(j_pa),len(j_freq),len(beam_parms['needed_pol']),beam_parms['image_size'][0],beam_parms['image_size'][1]),np.complex128)
    j_planes_shape = j_planes.shape
    iter_dims_indx = itertools.product(np.arange(j_planes_shape[0]),np.arange(j_planes_shape[1]))
    ic = beam_parms['image_size']//2 #image center pixel
    
    for i_pa, i_chan in iter_dims_indx:
        #print(i_pa,i_chan)
        pa = j_pa[i_pa]
        beam = zpc_dataset
        freq = j_freq[i_chan]

        if(freq != freq_prev):
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
        
        #print('x_grid',x_grid)
        #print('y_grid',y_grid)
        
        
        start = time.time()
        for i_pol,pol in enumerate(beam_parms['needed_pol']):
            a = _generate_zernike_surface(beam_interp.ZC.data[pol,:],x_grid,y_grid)
            a[r_grid > 1] = 0
            j_planes[i_pa, i_chan,i_pol,ic[0]-ic_z[0]:ic[0]+ic_z[0]+include_last[0],ic[1]-ic_z[1]:ic[1]+ic_z[1]+include_last[1]] = a
            j_planes[i_pa, i_chan, i_pol,:,:] = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(j_planes[i_pa, i_chan,i_pol,:,:])))/(beam_parms['image_size'][0]*beam_parms['image_size'][1])
        #print('One pol set',time.time()-start)
        
        #Normalize Jones
        if 3 not in beam_parms['needed_pol']:
            P_max = np.abs(j_planes[i_pa, i_chan, np.where(beam_parms['needed_pol']==0),j_planes_shape[3]//2,j_planes_shape[4]//2])
            Q_max = P_max
        elif 0 not in beam_parms['needed_pol']:
            Q_max = np.abs(j_planes[i_pa, i_chan, np.where(beam_parms['needed_pol']==3),j_planes_shape[3]//2,j_planes_shape[4]//2])
            P_max = Q_max
        else:
            P_max = np.abs(j_planes[i_pa, i_chan, np.where(beam_parms['needed_pol']==0),j_planes_shape[3]//2,j_planes_shape[4]//2])
            Q_max = np.abs(j_planes[i_pa, i_chan, np.where(beam_parms['needed_pol']==3),j_planes_shape[3]//2,j_planes_shape[4]//2])

        j_planes[i_pa, i_chan,:,:,:] = j_planes[i_pa, i_chan,:,:,:]*2/(P_max+Q_max)
        
        pa_prev = pa
        freq_prev = freq
    return j_planes#np.zeros((1,4,2048,2048),dtype=np.complex128)
