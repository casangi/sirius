import os
import xarray as xr
from astropy.time import Time
import numpy as np
import astropy.coordinates as coord
import astropy.time
import astropy.units as u
import matplotlib.pyplot as plt

# Can do this to get updated IERS B values into astropy
from astropy.utils import iers
from astropy.utils import data
from astropy.time import Time
iers_b = iers.IERS_B.open(data.download_file(iers.IERS_B_URL, cache=True))
iers_a = iers.IERS_A.open(data.download_file(iers.IERS_A_URL, cache=True))
iers_auto = iers.IERS_Auto.open()
from calc11 import almacalc

################################################################################

def calc_uvw_astropy(ant_pos, mjd, site, pointing_direction):
    """
    Parameters
    ----------
    ant_pos : numpy.array (n_antx3), Geocentric ITRF, m
    mjd : numpy.array (n_time), Modified Julian Day, UTC
    site : str
        Site name
    pointing_direction : string
        Define the UVW frame relative to a certain point on the sky.
    Returns
    -------
    ant_uvw
    """

    # VLA B-config positions, ITRF, m
    ant_pos = ant_pos* u.m

    # Time of observation:
    t = astropy.time.Time(mjd, format='mjd', scale='utc')

    # Format antenna positions and VLA center as EarthLocation.
    antpos_ap = coord.EarthLocation(x=ant_pos[:,0], y=ant_pos[:,1], z=ant_pos[:,2])
    tel_site = coord.EarthLocation.of_site(site)

    # Convert antenna pos terrestrial to celestial.  For astropy use
    # get_gcrs_posvel(t)[0] rather than get_gcrs(t) because if a velocity
    # is attached to the coordinate astropy will not allow us to do additional
    # transformations with it (https://github.com/astropy/astropy/issues/6280)
    tel_site_p, tel_site_v = tel_site.get_gcrs_posvel(t)
    antpos_c_ap = coord.GCRS(antpos_ap.get_gcrs_posvel(t)[0],
            obstime=t, obsgeoloc=tel_site_p, obsgeovel=tel_site_v)

    pointing_direction = coord.SkyCoord(pointing_direction[:,0]*u.rad, pointing_direction[:,1]*u.rad, frame='icrs')
    
    #frame_uvw = pointing_direction.skyoffset_frame() # ICRS
    frame_uvw = pointing_direction.transform_to(antpos_c_ap).skyoffset_frame() # GCRS

    # Rotate antenna positions into UVW frame.
    antpos_uvw_ap = antpos_c_ap.transform_to(frame_uvw).cartesian
    
    ant_uvw = np.array([antpos_uvw_ap.y,antpos_uvw_ap.z,antpos_uvw_ap.x]).T
  
    return ant_uvw

################################################################################

def find_nearest(array, values):
    indices = np.abs(np.subtract.outer(array, values)).argmin(0)
    return indices
     
def calc_uvw_CALC(jpx_de421, ant_pos, mjd, pointing_direction,time_obj,delta = 0.00001):
    """
    Parameters
    ----------
    ant_pos : numpy.array (n_antx3), Geocentric ITRF, m
    mjd : numpy.array (n_time), Modified Julian Day, UTC
    pointing_direction : string
        Define the UVW frame relative to a certain point on the sky.
    Returns
    -------
    ant_uvw
    """
    
    #from calc11 import almacalc
    ref_antenna  = 0 #Choosing the first antenna as the reference
    n_times = len(mjd)

    ########################################################################################################
    #Geocentric (ITRF) position of each antenna.
    #ant_pos #n_ant x 3
    ant_x = np.ascontiguousarray(ant_pos[:,0])
    ant_y = np.ascontiguousarray(ant_pos[:,1])
    ant_z = np.ascontiguousarray(ant_pos[:,2])
    n_ant = ant_x.shape[0]
    #Geocentric position of the array reference point (ITRF).
    ref_x = ant_x[ref_antenna]
    ref_y = ant_y[ref_antenna]
    ref_z = ant_z[ref_antenna]
    ########################################################################################################
    # Temperature (deg. C), Pressure (hPa/mbar), and humidity (0-1) at each antenna
    #      REAL*8 temp(nant), pressure(nant), humidity(nant)
    # Only effects dry and wet delay
    temp = np.array([-1.68070068]*n_ant)  #To deg C, n_ant
    pressure = np.array([555.25872803]*n_ant) #Pressure (hPa/mbar), n_ant
    humidity = np.array([0.054894]*n_ant) #humidity (0-1), n_ant

    ########################################################################################################
    #pointing_direction radians, n_time x 2
    ra = np.ascontiguousarray(pointing_direction[:,0])
    dec = np.ascontiguousarray(pointing_direction[:,1])

    ssobj = np.zeros(n_times, dtype=bool) #True if the source is a solar system object.
    #Earth orientation parameters at each time (arc-sec, arc-sec, sec)
    iers_b = iers.IERS_B.open()
    dx_dy = np.ascontiguousarray(np.array(iers_b.pm_xy(time_obj))) #2 x n_time
    dx = np.ascontiguousarray(np.array(dx_dy[0,:]))
    dy = np.ascontiguousarray(np.array(dx_dy[1,:]))
    dut  = np.ascontiguousarray(np.array(iers_b.ut1_utc(time_obj)))
    print(ant_x)
    #print(n_times)

    leapsec = 35
    axisoff = np.zeros(n_ant)
    
    sourcename = np.array(['P'] * n_times) # source names, for future use with solar system objects
    #jpx_de421 = '~/sirius/sirius/DE421_little_Endian' #Path name of the JPL ephemeris
    
    #Calculate uvw using same math as DiFX (The software is availble at https://www.atnf.csiro.au/vlbi/dokuwiki/doku.php/difx/installation see Applications/calcif2/src/difxcalc.c::callCalc for math)
    geodelay, drydelay, wetdelay = almacalc(ref_x, ref_y, ref_z, ant_x, ant_y,ant_z, temp, pressure, humidity, mjd, ra, dec, ssobj,dx, dy, dut, leapsec, axisoff,sourcename, jpx_de421)
    ra_x = ra - delta/np.cos(dec)
    geodelay_x, drydelay_x, wetdelay_x = almacalc(ref_x, ref_y, ref_z, ant_x, ant_y,ant_z, temp, pressure, humidity, mjd, ra_x, dec, ssobj,dx, dy, dut, leapsec, axisoff,sourcename, jpx_de421)
    dec_y = dec + delta
    geodelay_y, drydelay_y, wetdelay_y = almacalc(ref_x, ref_y, ref_z, ant_x, ant_y,ant_z, temp, pressure, humidity, mjd, ra, dec_y, ssobj,dx, dy, dut, leapsec, axisoff,sourcename, jpx_de421)

    c = 299792458
    
    u = (c/delta)*(geodelay-geodelay_x)[:,0]
    v = (c/delta)*(geodelay_y-geodelay)[:,0]
    w = (c*geodelay)[:,0]
    
    '''
    total_delay = geodelay + drydelay + wetdelay
    total_delay_x = geodelay_x + drydelay_x + wetdelay_x
    total_delay_y = geodelay_y + drydelay_y + wetdelay_y

    u = (c/delta)*(total_delay-total_delay_x)[:,0]
    v = (c/delta)*(total_delay_y-total_delay)[:,0]
    w = (c*total_delay)[:,0]
    '''
    
    uvw = np.array([u,v,w]).T
    return uvw