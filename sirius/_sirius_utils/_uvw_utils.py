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

'''
Calculating of uvw coordinates using different packages.
'''

from astropy.utils import iers
from astropy.utils import data
iers_b = iers.IERS_B.open(data.download_file(iers.IERS_B_URL, cache=True))
iers_a = iers.IERS_A.open(data.download_file(iers.IERS_A_URL, cache=True))
iers_auto = iers.IERS_Auto.open()
    
from astropy.time import Time
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from sirius_data._constants import c

def _calc_uvw_astropy(tel_xds, time_str, phase_center_ra_dec, antenna1, antenna2):
    n_time = len(time_str)
    n_ant = tel_xds.dims['ant_name']

    # Time of observation:
    time_str = np.tile(time_str[:,None],(1,n_ant))
    mjd = Time(Time(time_str, scale='utc'), format='mjd', scale='utc')

    # Format antenna positions and array center as EarthLocation.
    ant_pos = np.tile(tel_xds.ANT_POS.values[None,:,:]*u.m,(n_time,1,1))
    antpos_ap = coord.EarthLocation(x=ant_pos[:,:,0], y=ant_pos[:,:,1], z=ant_pos[:,:,2])
    #tel_site = coord.EarthLocation.of_site(site)
    tel_site = coord.EarthLocation(x=tel_xds.site_pos[0]['m0']['value']*u.m, y=tel_xds.site_pos[0]['m1']['value']*u.m, z=tel_xds.site_pos[0]['m2']['value']*u.m)
    
    # Convert antenna pos terrestrial to celestial.  For astropy use
    # get_gcrs_posvel(t)[0] rather than get_gcrs(t) because if a velocity
    # is attached to the coordinate astropy will not allow us to do additional
    # transformations with it (https://github.com/astropy/astropy/issues/6280)
    tel_site_p, tel_site_v = tel_site.get_gcrs_posvel(mjd)
    antpos_c_ap = coord.GCRS(antpos_ap.get_gcrs_posvel(mjd)[0],
            obstime=mjd, obsgeoloc=tel_site_p, obsgeovel=tel_site_v)

    phase_center_ra_dec = coord.SkyCoord(phase_center_ra_dec[:,0]*u.rad, phase_center_ra_dec[:,1]*u.rad, frame='icrs')

    #frame_uvw = phase_center_ra_dec.skyoffset_frame() # ICRS
    frame_uvw = phase_center_ra_dec.transform_to(antpos_c_ap).skyoffset_frame() # GCRS

    # Rotate antenna positions into UVW frame.
    antpos_uvw_ap = antpos_c_ap.transform_to(frame_uvw).cartesian
    
    ant_uvw = np.array([antpos_uvw_ap.y,antpos_uvw_ap.z,antpos_uvw_ap.x])
    ant_uvw = np.moveaxis(ant_uvw, 0, -1)
    
    uvw = np.ascontiguousarray(ant_uvw[:,antenna1,:] - ant_uvw[:,antenna2,:])
    return uvw
             

def _calc_uvw_casacore(tel_xds, time_str, phase_center_ra_dec,antenna1, antenna2):
    from casacore.measures import measures
    from casacore.quanta import quantity as qq
    
    n_time = len(time_str)
    n_baseline = len(antenna1)
    
    f_pc_time = n_time if phase_center_ra_dec.shape[0] == 1 else 1

    me = measures()

    tel_site = tel_xds.site_pos[0] #me.observatory(name=tel_xds.telescope_name)
    
    ant_pos = tel_xds.ANT_POS.values
    # Format antenna positions for CASA:
    antpos_casa = me.position('ITRF',
                qq(ant_pos[:,0],'m'),
                qq(ant_pos[:,1],'m'),
                qq(ant_pos[:,2],'m'))
    
    uvw = np.zeros((n_time,n_baseline,3))
    
    for i,t in enumerate(time_str):
        ra_dec = phase_center_ra_dec[i//f_pc_time,:]
        me = measures()
        #me.set_data_path(casa_data_dir)
        me.do_frame(tel_site)
        me.do_frame(me.epoch('utc', str(t)))
        me.do_frame(me.direction('J2000',qq(ra_dec[0], 'rad'),qq(ra_dec[1], 'rad')))
        
        # Converts from ITRF to "J2000":
        antpos_c_casa = me.as_baseline(antpos_casa)
        # Rotate into UVW frame
        
        antpos_uvw_casa = me.to_uvw(antpos_c_casa)['measure']
        
        ant_uvw = _casa_to_astropy(antpos_uvw_casa)
        ant_uvw = ant_uvw.xyz.value.T

        uvw[i,:,:] = np.ascontiguousarray(ant_uvw[antenna1,:] - ant_uvw[antenna2,:])
    return uvw


def _calc_uvw_casa(tel_xds, time_str, phase_center_ra_dec,antenna1, antenna2):
    print("Warning CASA uvw code is not thread safe.")
    import casatools

    n_time = len(time_str)
    n_baseline = len(antenna1)
    
    f_pc_time = n_time if phase_center_ra_dec.shape[0] == 1 else 1
    
    qa = casatools.quanta()
    qq = qa.quantity
    
    me = casatools.measures()
    
    tel_site = tel_xds.site_pos[0]#me.observatory(tel_xds.telescope_name)
    
    ant_pos = tel_xds.ANT_POS.values
    # Format antenna positions for CASA:
    antpos_casa = me.position('ITRF',
                qq(ant_pos[:,0],'m'),
                qq(ant_pos[:,1],'m'),
                qq(ant_pos[:,2],'m'))
    me.done()
    
    uvw = np.zeros((n_time,n_baseline,3))
    
    for i,t in enumerate(time_str):
        ra_dec = phase_center_ra_dec[i//f_pc_time,:]
        me = casatools.measures()
        me.doframe(tel_site)
        me.doframe(me.epoch('utc', str(t)))
        me.doframe(me.direction('J2000',qq(ra_dec[0], 'rad'),qq(ra_dec[1], 'rad')))
        
        # Converts from ITRF to "J2000":
        antpos_c_casa = me.asbaseline(antpos_casa)
        # Rotate into UVW frame
        antpos_uvw_casa = me.touvw(antpos_c_casa)[0]
        me.done()
        
        ant_uvw = _casa_to_astropy(antpos_uvw_casa)
        ant_uvw = ant_uvw.xyz.value.T

        uvw[i,:,:] = np.ascontiguousarray(ant_uvw[antenna1,:] - ant_uvw[antenna2,:])
    return uvw
     

def _casa_to_astropy(c):
    """Convert CASA spherical coords to astropy CartesianRepresentation"""
    sph = coord.SphericalRepresentation(
            lon=c['m0']['value']*u.Unit(c['m0']['unit']),
            lat=c['m1']['value']*u.Unit(c['m1']['unit']),
            distance=c['m2']['value']*u.Unit(c['m2']['unit']))
    return sph.represent_as(coord.CartesianRepresentation)

'''
def _calc_uvw_CALC(jpx_de421, ant_pos, mjd, phase_center_ra_dec,time_obj,delta = 0.00001):
    """
    Parameters
    ----------
    ant_pos : numpy.array (n_antx3), Geocentric ITRF, m
    mjd : numpy.array (n_time), Modified Julian Day, UTC
    phase_center_ra_dec : string
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
    #phase_center_ra_dec radians, n_time x 2
    ra = np.ascontiguousarray(phase_center_ra_dec[:,0])
    dec = np.ascontiguousarray(phase_center_ra_dec[:,1])

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

 
    
    u = (c/delta)*(geodelay-geodelay_x)[:,0]
    v = (c/delta)*(geodelay_y-geodelay)[:,0]
    w = (c*geodelay)[:,0]
    

#    total_delay = geodelay + drydelay + wetdelay
#    total_delay_x = geodelay_x + drydelay_x + wetdelay_x
#    total_delay_y = geodelay_y + drydelay_y + wetdelay_y
#
#    u = (c/delta)*(total_delay-total_delay_x)[:,0]
#    v = (c/delta)*(total_delay_y-total_delay)[:,0]
#    w = (c*total_delay)[:,0]

    
    uvw = np.array([u,v,w]).T
    return uvw
'''
 
