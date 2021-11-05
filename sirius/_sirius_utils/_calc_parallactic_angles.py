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

import numpy as np
import xarray as xr
import dask.array as da
import time
import matplotlib.pyplot as plt
from numba import jit
import numba

def _calc_parallactic_angles(times, site_pos, phase_center):
    """
    Computes parallactic angles per timestep for the given
    reference antenna position and field centre.
    
    Based on https://github.com/ska-sa/codex-africanus/blob/068c14fb6cf0c6802117689042de5f55dc49d07c/africanus/rime/parangles_astropy.py
    """
    from astropy.coordinates import (EarthLocation, SkyCoord,
                                     AltAz, CIRS)
    from astropy.time import Time
    from astropy import units
    import numpy as np
    import astropy.units as u
    import astropy.coordinates as coord
    
    #if site=='EVLA': site='VLA'
    #observing_location = EarthLocation.of_site(site)
    observing_location = coord.EarthLocation(x=site_pos[0]['m0']['value']*u.m, y=site_pos[0]['m1']['value']*u.m, z=site_pos[0]['m2']['value']*u.m)
    phase_center = SkyCoord(ra=phase_center[:,0]*u.rad, dec=phase_center[:,1]*u.rad, frame='fk5')
    
    pole = SkyCoord(ra=0, dec=90, unit=units.deg, frame='fk5')

    cirs_frame = CIRS(obstime=times)
    pole_cirs = pole.transform_to(cirs_frame)
    phase_center_cirs = phase_center.transform_to(cirs_frame)

    altaz_frame = AltAz(location=observing_location, obstime=times)
    pole_altaz = pole_cirs.transform_to(altaz_frame)
    phase_center_altaz = phase_center_cirs.transform_to(altaz_frame)
    
    #print('the zen angle is',phase_center_altaz.zen)
    #print('the zen angle is',pole_altaz.zen)
        
    return phase_center_altaz.position_angle(pole_altaz).value

@jit(nopython=True,cache=True,nogil=True)
def _find_optimal_set_angle(nd_vals,val_step):
    vals_flat = np.ravel(nd_vals)
    n_vals = len(vals_flat)
    neighbours = np.zeros((n_vals,n_vals),numba.b1)

    for ii in range(n_vals):
        for jj in range(n_vals):
            #https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
            ang_dif = vals_flat[ii]-vals_flat[jj]
            ang_dif = np.abs((ang_dif + np.pi)%(2*np.pi) - np.pi)
            
            #neighbours_dis[ii,jj] = ang_dif
            
            if ang_dif <= val_step:
                neighbours[ii,jj] = True
             
    neighbours_rank = np.sum(neighbours,axis=1)
    vals_centers = [42.0] #Dummy value to let numba know what dtype of list is
    lonely_neighbour = True
    while lonely_neighbour:
        #if True:
        neighbours_rank = np.sum(neighbours,axis=1)
        highest_ranked_neighbour_indx = np.argmax(neighbours_rank)
        
        if neighbours_rank[highest_ranked_neighbour_indx]==0:
            lonely_neighbour = False
        else:
            group_members = np.where(neighbours[highest_ranked_neighbour_indx,:]==1)[0]
            vals_centers.append(vals_flat[highest_ranked_neighbour_indx]) #no outliers
            #vals_centers.append(np.median(vals_flat[neighbours[highest_ranked_neighbour_indx,:]])) #best stats
            #vals_centers.append(np.mean(vals_flat[neighbours[highest_ranked_neighbour_indx,:]])) #?
            
            for group_member in group_members:
                for ii in range(n_vals):
                    neighbours[group_member,ii] = 0
                    neighbours[ii,group_member] = 0
                    
    vals_centers.pop(0)
    vals_centers = np.array(vals_centers)
    

    n_time = nd_vals.shape[0]
    n_beam = nd_vals.shape[1]
    vals_dif = np.zeros(nd_vals.shape,numba.f8)
    
    for ii in range(n_time):
        for kk in range(n_beam):
            min_dif = 42.0 #Dummy value to let numba know what dtype of list is
            group_indx = -1
            for jj in range(len(vals_centers)):
                #https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
                ang_dif = nd_vals[ii,kk]-vals_centers[jj]
                ang_dif = np.abs((ang_dif + np.pi)%(2*np.pi) - np.pi)
                
                if min_dif > ang_dif:
                    min_dif = ang_dif
            
            vals_dif[ii,kk] = min_dif
            

    
    return vals_centers, vals_dif
    ##############################################################################################################

