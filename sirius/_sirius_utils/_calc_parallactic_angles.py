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
#
# Four different implementations for calculating the parallactic angle have been included. Each version produces a slightly different answer.
# The reason for the differences has not been verified, however, we suspect they have to do with different earth models and earth movement corrections that are applied by the different software packages.
# Here is an interesting discussion about different packages giving differing parallactic angles: https://github.com/ratt-ru/codex-africanus/issues/21


import numpy as np
from numba import jit
import numba

#AZEL vs AZELGEO
#Dir FK5 (J2000) vs ICRS


def _calc_parallactic_angles_astropy(times, observing_location, direction, dir_frame='FK5', zenith_frame='FK5'):
    """
    Converts a direction and zenith (frame FK5) to a topocentric Altitude-Azimuth (https://docs.astropy.org/en/stable/api/astropy.coordinates.AltAz.html) frame centered at the observing_location (frame ITRF) for a UTC time. The parallactic angles is calculated as the position angle of the Altitude-Azimuth direction and zenith.
    
    Parameters
    ----------
    times: str np.array, [n_time], 'YYYY-MM-DDTHH:MM:SS.SSS'
        UTC time series. Example '2019-10-03T19:00:00.000'.
    observing_location: int np.array, [3], [x,y,z], meters
        ITRF geocentric coordinates.
    direction: float np.array, [n_time,2], [time,ra,dec], radians
        The pointing direction.
    Returns
    -------
    parallactic_angles: float np.array, [n_time], radians
        An array of parallactic angles.
    """
    import astropy.units as u
    import astropy.coordinates as coord
    
    observing_location = coord.EarthLocation.from_geocentric(x=observing_location[0]*u.m, y=observing_location[1]*u.m, z=observing_location[2]*u.m)
    
    direction = coord.SkyCoord(ra=direction[:,0]*u.rad, dec=direction[:,1]*u.rad, frame=dir_frame.lower())
    zenith = coord.SkyCoord(0, 90, unit=u.deg, frame=zenith_frame.lower())
    
    altaz_frame = coord.AltAz(location=observing_location, obstime=times)
    zenith_altaz = zenith.transform_to(altaz_frame)
    direction_altaz = direction.transform_to(altaz_frame)
    
    return direction_altaz.position_angle(zenith_altaz).value
    
    '''
    cirs_frame = coord.CIRS(obstime=times)
    zenith_cirs = zenith.transform_to(cirs_frame)
    direction_cirs = direction.transform_to(cirs_frame)

    altaz_frame = coord.AltAz(location=observing_location, obstime=times)
    zenith_altaz = zenith_cirs.transform_to(altaz_frame)
    direction_altaz = direction_cirs.transform_to(altaz_frame)
    
    return direction_altaz.position_angle(zenith_altaz).value
    '''
    
def _calc_parallactic_angles_astropy2(times, observing_location, direction, dir_frame='FK5'):
    """
    Calculates a set of parallactic angles for a array of time strings for a given earth obeserving location and direction.
    Based on the Astropy implementation in https://github.com/ARDG-NRAO/plumber/blob/master/plumber/sky.py.

    Parameters
    ----------
    times: str np.array, [n_time], 'YYYY-MM-DDTHH:MM:SS.SSS'
        UTC time series. Example '2019-10-03T19:00:00.000'.
    observing_location: int np.array, [3], [x,y,z], meters
        ITRF geocentric coordinates.
    direction: float np.array, [n_time,2], [time,ra,dec], radians
        The pointing direction.
    Returns
    -------
    parallactic_angles: float np.array, [n_time], radians
        An array of parallactic angles.
    """
    from astroplan import Observer
    import astropy.units as u
    import astropy.coordinates as coord
    from astropy.time import Time
    
    from casatools import measures
    me = measures()
    
    observing_location = coord.EarthLocation.from_geocentric(x=observing_location[0]*u.m, y=observing_location[1]*u.m, z=observing_location[2]*u.m)
    
    direction = coord.SkyCoord(ra=direction[:,0]*u.rad, dec=direction[:,1]*u.rad, frame=dir_frame.lower())
    
    observer = Observer(location=observing_location, name='tel', timezone='UTC')
    
    times = Time(times)

    return observer.parallactic_angle(times, direction).value
    

def _calc_parallactic_angles_casa(times, observing_location, direction, frame='AZEL', dir_frame='FK5', zenith_frame='HADEC'):
    """
    Calculates a set of parallactic angles for a array of time strings for a given earth obeserving location and direction.
    Makes use of casacore measures posangle function (https://casacore.github.io/python-casacore/casacore_measures.html#casacore.measures.measures.posangle).

    Parameters
    ----------
    times: str np.array, [n_time], 'YYYY-MM-DDTHH:MM:SS.SSS'
        UTC time series. Example '2019-10-03T19:00:00.000'.
    observing_location: int np.array, [3], [x,y,z], meters
        ITRF geocentric coordinates.
    direction: float np.array, [n_time,2], [time,ra,dec], radians
        The pointing direction.
    Returns
    -------
    parallactic_angles: float np.array, [n_time], radians
        An array of parallactic angles.
    """
    from casacore.measures import measures
    from casacore.quanta import quantity as qq
    me = measures()
    
    n_time = len(times)
    f_pc_time = n_time if direction.shape[0] == 1 else 1
    
    observing_location = me.position('ITRF',
                    qq(observing_location[0],'m'),
                    qq(observing_location[1],'m'),
                    qq(observing_location[2],'m'))
    
    if dir_frame=='FK5':
        dir_frame = 'J2000' #Change to CASA convention.
        
    if zenith_frame=='FK5':
        zenith_frame = 'J2000'
        
    if frame == 'FK5':
        frame='J2000'
    
    parallactic_angles = np.zeros(n_time)
    for i,t in enumerate(times):
        me = measures()
        me.doframe(observing_location)
        me.doframe(me.epoch('utc', t))
        
        zenith = me.measure(me.direction(rf=zenith_frame, v0='0deg', v1='90deg'),frame)
        pointing_dir = me.measure(me.direction(dir_frame,qq(direction[i//f_pc_time,0], 'rad'),qq(direction[i//f_pc_time,1], 'rad')),frame)
        
        #print(pointing_dir)
 
        parallactic_angles[i] = me.posangle(pointing_dir, zenith).get_value("rad")
 
    return parallactic_angles
    
def _calc_parallactic_angles_casa2(times, observing_location, direction, dir_frame='FK5'):
    """
    Calculates a set of parallactic angles for a array of time strings for a given earth obeserving location and direction.
    Based on the Casa implementation in https://github.com/ARDG-NRAO/plumber/blob/master/plumber/sky.py.

    Parameters
    ----------
    times: str np.array, [n_time], 'YYYY-MM-DDTHH:MM:SS.SSS'
        UTC time series. Example '2019-10-03T19:00:00.000'.
    observing_location: int np.array, [3], [x,y,z], meters
        ITRF geocentric coordinates.
    direction: float np.array, [n_time,2], [time,ra,dec], radians
        The pointing direction.
    Returns
    -------
    parallactic_angles: float np.array, [n_time], radians
        An array of parallactic angles.
    """
    from casacore.measures import measures
    from casacore.quanta import quantity as qq
    me = measures()
    
    n_time = len(times)
    f_pc_time = n_time if direction.shape[0] == 1 else 1
    
    lat = np.arcsin(observing_location[2]/np.sqrt(np.sum(observing_location**2)))
    observing_location = me.position('ITRF',
                    qq(observing_location[0],'m'),
                    qq(observing_location[1],'m'),
                    qq(observing_location[2],'m'))
                    
    parallactic_angles = np.zeros(n_time)
    
    if dir_frame=='FK5':
        dir_frame = 'J2000' #Change to CASA convention.
    
    for i,t in enumerate(times):
        me = measures()
        me.doframe(observing_location)
        tm = me.epoch('utc', t)
        last = me.measure(tm, 'LAST')['m0']['value']
        pointing_dir = me.direction(dir_frame,qq(direction[i//f_pc_time,0], 'rad'),qq(direction[i//f_pc_time,1], 'rad'))
        
        ha = (last - np.floor(last))*24 - (pointing_dir['m0']['value']*12/np.pi) # hours
        ha = ha*np.pi/12.  # rad
        dec = pointing_dir['m1']['value']
        parallactic_angles[i] = np.arctan2(np.cos(lat)*np.sin(ha),(np.sin(lat) * np.cos(dec) - np.cos(lat)*np.sin(dec)*np.cos(ha)))
        
    return parallactic_angles

    
    
def _calc_parallactic_angles_casa3(times, observing_location, direction, frame='AZEL', dir_frame='FK5', zenith_frame='HADEC'):
    """
    Calculates a set of parallactic angles for a array of time strings for a given earth obeserving location and direction.
    Based on: casacore/ms/MSOper/MSDerivedValues.cc::parAngle(),casacore/casa/Quanta/MVDirection.cc::positionAngle.
    This casacore code is used by awproject in CASA tclean. Note that the casacore/measures/Measures/ParAngleMachine.cc is an independant implementation (see _calc_parallactic_angles_casa).

    Parameters
    ----------
    times: str np.array, [n_time], 'YYYY-MM-DDTHH:MM:SS.SSS'
        UTC time series. Example '2019-10-03T19:00:00.000'.
    observing_location: int np.array, [3], [x,y,z], meters
        ITRF geocentric coordinates.
    direction: float np.array, [n_time,2], [time,ra,dec], radians
        The pointing direction.
    Returns
    -------
    parallactic_angles: float np.array, [n_time], radians
        An array of parallactic angles.
    """
    from casacore.measures import measures
    from casacore.quanta import quantity as qq
    from sirius._sirius_utils._coord_transforms import _directional_cosine
    me = measures()
    
    n_time = len(times)
    f_pc_time = n_time if direction.shape[0] == 1 else 1
    
    lat = np.arcsin(observing_location[2]/np.sqrt(np.sum(observing_location**2)))
    observing_location = me.position('ITRF',
                    qq(observing_location[0],'m'),
                    qq(observing_location[1],'m'),
                    qq(observing_location[2],'m'))
    me.doframe(observing_location)
    
    parallactic_angles = np.zeros(n_time)
    
    if dir_frame=='FK5':
        dir_frame = 'J2000' #Change to CASA convention.
        
    if zenith_frame=='FK5':
        zenith_frame = 'J2000'
        
    if frame == 'FK5':
        frame='J2000'
    
    
    for i,t in enumerate(times):
        me = measures()
        me.doframe(observing_location)
        reference_time = me.epoch('utc', t)
        me.doframe(reference_time)
        
        #Create Pole direction
        zenith = me.measure(me.direction(rf=zenith_frame, v0='0deg', v1='90deg'),frame)
        zenith_cosine = _directional_cosine(np.array([[zenith['m0']['value'],zenith['m1']['value']]]))[0,:]

        pointing_dir = me.measure(me.direction(dir_frame,qq(direction[i//f_pc_time,0], 'rad'),qq(direction[i//f_pc_time,1], 'rad')), frame)
        pointing_cosine=_directional_cosine(np.array([[pointing_dir['m0']['value'],pointing_dir['m1']['value']]]))[0,:]
        
        zenith_long = np.arctan2(zenith_cosine[1],zenith_cosine[0])
        pointing_long = np.arctan2(pointing_cosine[1],pointing_cosine[0])
        
        pc_slat1 = pointing_cosine[2];
        zenith_slat2 = zenith_cosine[2];
        zenith_clat2 = np.sqrt(np.abs(1 - zenith_slat2**2))
        
        s1 = (-zenith_clat2 * np.sin(pointing_long-zenith_long))
        c1 = (np.sqrt(np.abs(1.0 - pc_slat1**2))*zenith_slat2 - pc_slat1*zenith_clat2*np.cos(pointing_long-zenith_long));
        parallactic_angles[i] = np.arctan2(s1,c1)
    return parallactic_angles
    

@jit(nopython=True,cache=True,nogil=True)
def _find_optimal_set_angle(ang_vals,val_step,calc_dif=False):
    """
    For an array of angles (ang_vals) find an "optimal" (ie good) subset of angles (subset_ang_vals) so that these angles all lie within val_step of all the values in val_step.
    
    Parameters
    ----------
    ang_vals: float np.array, [n_vals], radians
    val_step: float, radians
    calc_dif: bool
    Returns
    -------
    subset_ang_vals: float np.array, [n_subset_vals], radians
    vals_dif: float np.array, [n_vals], radians
        Optional output if calc_dif is True. The minimum angular distance for each ang_vals with subset_ang_vals.
    vals_indx: int np.array, [n_vals], radians
        Optional output if calc_dif is True. The index for each ang_vals into subset_ang_vals so that the angular distance is a minimum.
    """
    n_vals = len(ang_vals)
    neighbours = np.identity(n_vals,numba.b1)
    #neighbours = np.identity(n_vals,int)

    # Find all the neighbours. A boolean n_vals x n_vals array, called neighbours, is created.
    # If two angles are within val_step they are considered neighbours.
    for ii in range(0,n_vals-1):
        for jj in range(ii+1,n_vals):
            #https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
            ang_dif = ang_vals[ii]-ang_vals[jj]
            ang_dif = np.abs((ang_dif + np.pi)%(2*np.pi) - np.pi)
            
            if ang_dif <= val_step:
                neighbours[ii,jj] = True
                neighbours[jj,ii] = True
                
    # Create neighbour groups by iteratively selecting the val with the highest neighbours_rank and setting that val and all its neigbours to False (0) in neighbours. Currently, the value of the neighbour with the highest neighbours_rank is used as the center value for the group (other options are commented out: median and mean). When all values are False (0) in neighbours the process ends.
    subset_ang_vals = [42.0] #Dummy value to let numba know what dtype of list is.
    lonely_neighbour = True
    while lonely_neighbour:
        #if True:
        neighbours_rank = np.sum(neighbours,axis=1) #Each val has a neighbours_rank equal to all the vals that are within val_step.
        highest_ranked_neighbour_indx = np.argmax(neighbours_rank)
        
        if neighbours_rank[highest_ranked_neighbour_indx]==0:
            lonely_neighbour = False
        else:
            group_members = np.where(neighbours[highest_ranked_neighbour_indx,:]==1)[0]
            subset_ang_vals.append(ang_vals[highest_ranked_neighbour_indx])
            #subset_ang_vals.append(np.median(ang_vals[neighbours[highest_ranked_neighbour_indx,:]])) #best stats
            #subset_ang_vals.append(np.mean(ang_vals[neighbours[highest_ranked_neighbour_indx,:]])) #?
            
            for group_member in group_members:
                for ii in range(n_vals):
                    neighbours[group_member,ii] = 0
                    neighbours[ii,group_member] = 0
                    
    subset_ang_vals.pop(0)
    subset_ang_vals = np.array(subset_ang_vals)
    
    if calc_dif:
        vals_dif = np.zeros(n_vals,numba.f8)
        vals_indx = np.zeros(n_vals,numba.f8)
        
        # Calculate the min angular distance between subset_ang_vals and ang_vals for each value in ang_vals.
        # The index of the value in subset_ang_vals that gives the min angular distance is also stored in vals_indx.
        for ii in range(n_vals):
            min_dif = 42.0 #Dummy value to let numba know the dtype.
            indx = -42 #Dummy value to let numba know the dtype.
            for jj in range(len(subset_ang_vals)):
                #https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
                ang_dif = ang_vals[ii]-subset_ang_vals[jj]
                ang_dif = np.abs((ang_dif + np.pi)%(2*np.pi) - np.pi)
                
                if min_dif > ang_dif:
                    min_dif = ang_dif
                    indx = jj
            
            vals_dif[ii] = min_dif
            vals_indx[ii] = indx
        return subset_ang_vals, vals_dif, vals_indx
    else:
        return subset_ang_vals, np.array([0.0]), np.array([0.0])



