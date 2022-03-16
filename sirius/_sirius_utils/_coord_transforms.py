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

from scipy.spatial.transform import Rotation as R
import numpy as np
from numba import jit, types
import numba
from astropy.wcs import WCS

@jit(nopython=True,cache=True,nogil=True)
def _func_R_z(phi):
    result = np.zeros((3, 3), dtype = numba.float64)
    result[0, 0] = np.cos(phi)
    result[0, 1] = -np.sin(phi)
    result[1, 0] = np.sin(phi)
    result[1, 1] = np.cos(phi)
    result[2, 2] = 1
    return result

#def _func_R_z(phi):
#    return np.array([[np.cos(phi), -np.sin(phi), 0],[np.sin(phi), np.cos(phi), 0],[0, 0, 1]])

@jit(nopython=True,cache=True,nogil=True)
def _func_R_x(theta):
    result = np.zeros((3, 3), dtype = numba.float64)
    result[0, 0] = 1
    result[1, 1] = np.cos(theta)
    result[1, 2] = -np.sin(theta)
    result[2, 1] = np.sin(theta)
    result[2, 2] = np.cos(theta)
    return result

#def _func_R_x(theta):
#    return np.array([[1, 0, 0],[0, np.cos(theta), -np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])

@jit(nopython=True,cache=True,nogil=True)
def _func_R_y(psi):
    result = np.zeros((3, 3), dtype = numba.float64)
    result[0, 0] = np.cos(psi)
    result[0, 2] = np.sin(psi)
    result[1, 1] = 1
    result[2, 0] = -np.sin(psi)
    result[2, 2] = np.cos(psi)
    return result
    
#def _func_R_y(psi):
#    return np.array([[np.cos(psi), 0, np.sin(psi)],[0, 1, 0],[-np.sin(psi), 0, np.cos(psi)]])

@jit(nopython=True,cache=True,nogil=True)
def _directional_cosine(ra_dec):
    """
    The coordinate definition of the measurement set V2/3 is used (https://casacore.github.io/casacore-notes/264.pdf see page 12 and https://drive.google.com/file/d/1a-eUwNrfnYjaUQTjJDfOjJCa8ZaSzZcn/view?usp=sharing). Note this does not follow the convention in chapter 4 of https://link.springer.com/book/10.1007/978-3-319-44431-4 or CASACORE.
    
    Celestial coordinate system is defined by:
       - Z pointing to NCP (dec = pi/2),
       - X pointing to the East (ra = pi/2, dec = 0),
       - (-Y) pointing to the vernal equinox (ra = 0, dec = 0).
    The uvw coordinates are aligned with XYZ when the source is located at the NCP (ra=0 and dec=pi/2).
    A good overview of coordinates systems [1] http://www.danfleisch.com/maxwell/CoordinateSystemReview.pdf
    Using the physics convention for spherical coordinates:
       - Polar angle      theta = pi/2 - dec (1)
       - Azimuthal Angle  phi   = 3*pi/2 + ra (2)
    Substituting (1) and (2) into equations (8)-(10) of [1] yields lmn.
    
    Parameters
    ----------
    ra_dec: float np.array, [n_pos,2], radians
        Right ascension and declination coordinates.
    Returns
    -------
    lmn : float np.array, [n_pos,3], radians
        Directional cosine coordinates.
    """
    ra_dec_shape = ra_dec.shape
    lmn = np.zeros((ra_dec_shape[0],3),dtype = numba.float64)
    
    for i in range(ra_dec_shape[0]):
        lmn[i,0] = np.sin(ra_dec[i,0])*np.cos(ra_dec[i,1])
        lmn[i,1] = -np.cos(ra_dec[i,0])*np.cos(ra_dec[i,1])
        lmn[i,2] = np.sin(ra_dec[i,1])

    return lmn
    
@jit(nopython=True,cache=True,nogil=True)
def _sin_project(ra_dec_o,ra_dec):
    """
    Orthographic/Synthesis projection of right ascension and declination to a tangential plane with center at ra_dec_o.
    See equations 10 in http://tdc-www.harvard.edu/wcstools/aips27.pdf.
    
    Parameters
    ----------
    ra_dec_o: float np.array, [2], radians
        Right ascension and declination of the image center on the celestial sphere.
    ra_dec: float np.array, [n_pos,2], radians
        Right ascension and declination of a point on the celestial sphere.
    Returns
    -------
    lm : float np.array, [n_pos,2], radians
        Right ascension and declination (ra_dec) projected onto the image plane.
    """   
    ra_dec_shape = ra_dec.shape
    lm = np.zeros(ra_dec_shape,dtype = numba.float64)
    ra_o = ra_dec_o[0]
    dec_o = ra_dec_o[1]
    
    for i in range(ra_dec_shape[0]):
        ra = ra_dec[i,0]
        dec = ra_dec[i,1]
        lm[i,0] = np.cos(dec)*np.sin(ra-ra_o)
        lm[i,1] = np.sin(dec)*np.cos(dec_o) - np.cos(dec)*np.sin(dec_o)*np.cos(ra-ra_o)
    return lm
    
@jit(nopython=True,cache=True,nogil=True)
def _inv_sin_project(ra_dec_o,lm):
    """
    Inverse Orthographic/Synthesis projection from the image (tangential) plane with center at ra_dec_o to the celestial sphere. See section 3.2.2 in http://tdc-www.harvard.edu/wcstools/aips27.pdf.
    
    Parameters
    ----------
    ra_dec_o: float np.array, [2], radians
        Right ascension and declination of the image center on the celestial sphere.
    lm : float np.array, [n_pos,2], radians
        Coordinates of a point on the image plane.
    Returns
    -------
    ra_dec: float np.array, [n_pos,2], radians
        Image plane coordinates (lm) projected onto the celestial sphere.
    """
    lm_shape = lm.shape
    ra_dec = np.zeros(lm_shape,dtype = numba.float64)
    #ra_dec = np.zeros(lm_shape)
    ra_o = ra_dec_o[0]
    dec_o = ra_dec_o[1]
    
    for i in range(lm_shape[0]):
        l = lm[i,0]
        m = lm[i,1]
        n = np.sqrt(1 - l**2 - m**2)
        ra_dec[i,0] = ra_o + np.arctan2(l,n*np.cos(dec_o)-m*np.sin(dec_o))
        ra_dec[i,1] = np.arcsin(m*np.cos(dec_o) + n*np.sin(dec_o))
    return ra_dec
    
def _sin_pixel_to_celestial_coord(ra_dec_o,image_size,delta,pixel):
    """
    Converts a pixel coordinate of grid defined on the image plane to right ascension and declination coordinate on the celestial sphere.
    
    Parameters
    ----------
    ra_dec_o: float np.array, [n_pos,2], radians
        Array with the right ascension and declination of the image center on the celestial sphere.
    image_size : int np.array, [2]
        Size of image plane.
    delta: float np.array, [2], radians
        The size of a single pixel.
    pixel: float np.array, [n_pos,2]
        Pixel coordinate on the grid defined by image_size and delta. Fractional pixel coordinates are allowed.
    Returns
    -------
    ra_dec: float np.array, [n_pos,2], radians
        Image plane coordinates (lm) projected onto the celestial sphere.
    """
    lm = (pixel - image_size//2)*delta
    return _inv_sin_project(ra_dec_o,lm)
    
    
def _celestial_coord_to_sin_pixel(ra_dec_o,image_size,delta,ra_dec):
    """
    Converts a right ascension and declination coordinate on the celestial sphere to a pixel coordinate of grid defined on the image plane.
    
    Parameters
    ----------
    ra_dec_o: float np.array, [2], radians
        Array with the right ascension and declination of the image center on the celestial sphere.
    image_size : int np.array, [2]
        Size of image plane.
    delta: float np.array, [2], radians
        The size of a single pixel.
    ra_dec: float np.array, [np_pos,2], radians
        Right ascension and declination of a point on the celestial sphere.
    Returns
    -------
    pixel: float np.array, [n_pos,2]
        Pixel coordinate on the grid defined by image_size and delta.
    """
    lm = _sin_project(ra_dec_o,ra_dec)
    return (lm/delta + image_size//2)
    
    
@jit(nopython=True,cache=True,nogil=True)
def _tan_project(ra_dec_o,ra_dec):
    #Gnomonic projection of right ascension and declination to a tangential plane with center at ra_dec_o.
    #Equations 9 http://tdc-www.harvard.edu/wcstools/aips27.pdf
    ra = ra_dec[0]
    dec = ra_dec[1]
    ra_o = ra_dec_o[0]
    dec_o = ra_dec_o[1]
    lm = np.zeros((2,),dtype = numba.float64)
    div_term = np.sin(dec)*np.sin(dec_o) + np.cos(dec)*np.cos(dec_o)*np.cos(ra-ra_o)
    lm[0] = np.cos(dec)*np.sin(ra-ra_o)/div_term
    lm[1] = (np.sin(dec)*np.cos(dec_o) - np.cos(dec)*np.sin(dec_o)*np.cos(ra-ra_o))/div_term
    return lm
    
@jit(nopython=True,cache=True,nogil=True)
def _local_spherical_coords(ra_dec_o,ra_dec):
    #Equation 6-8 http://tdc-www.harvard.edu/wcstools/aips27.pdf
    ra = ra_dec[0]
    dec = ra_dec[1]
    ra_o = ra_dec_o[0]
    dec_o = ra_dec_o[1]
    theta_phi = np.zeros((2,),dtype = numba.float64)
    theta_phi[0] = np.arccos(np.sin(dec)*np.sin(dec_o) + np.cos(dec)*np.cos(dec_o)*np.cos(ra-ra_o))
    theta_phi[1] = np.arctan2(np.cos(dec)*np.sin(ra-ra_o),np.sin(dec)*np.cos(dec_o) + np.cos(dec)*np.cos(dec_o)*np.cos(ra-ra_o))
    
    return theta_phi
    

@jit(nopython=True,cache=True,nogil=True)
def _calc_rotation_mats(ra_dec_o,ra_dec):
    
    #The default
    #Rotates input system to output system.
    uvw_rotmat = _func_R_x(-(np.pi/2-ra_dec_o[1]))@_func_R_z(ra_dec[0]-ra_dec_o[0])@_func_R_x(np.pi/2 - ra_dec[1])
    
    #Rotates output to Standard System
    out_rotmat = _func_R_x(-(np.pi/2-ra_dec[1]))@_func_R_z(-ra_dec[0])
    lmn_out = _directional_cosine(ra_dec.reshape(1,-1))[0,:]
    lmn_in = _directional_cosine(ra_dec_o.reshape(1,-1))[0,:]
    
    lmn_rot = out_rotmat@(lmn_out - lmn_in) # out_rotmat(lmn_out - lmn_in)= [0,0,1] - out_rotmat@[l_in,m_in,n_in]
    
    return uvw_rotmat, lmn_rot


@jit(types.containers.UniTuple(numba.float64, 2)(numba.float64, numba.float64, numba.float64), nopython=True,cache=True,nogil=True)
def _rot_coord(x,y,angle):
    x_rot = np.cos(angle)*x + np.sin(angle)*y
    y_rot = - np.sin(angle)*x + np.cos(angle)*y
    return x_rot,y_rot


def _compute_rot_coords(image_size,cell_size,angle):
    image_center = image_size//2
    #print(image_size)

    x = np.arange(-image_center[0], image_size[0]-image_center[0])*cell_size[0]
    y = np.arange(-image_center[1], image_size[1]-image_center[1])*cell_size[1]
    #xy = np.array([x,y]).T
    x_grid, y_grid = np.meshgrid(x,y,indexing='ij')
    
    if angle != 0:
        #rot_mat = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]) #anti clockwise
        
        #r = np.einsum('ji, mni -> jmn', rot_mat, np.dstack([x_grid, y_grid]))
        '''
        x_grid_rot = np.cos(angle)*x_grid - np.sin(angle)*y_grid
        y_grid_rot = np.sin(angle)*x_grid + np.cos(angle)*y_grid
        '''
        x_grid_rot = np.cos(angle)*x_grid + np.sin(angle)*y_grid
        y_grid_rot = - np.sin(angle)*x_grid + np.cos(angle)*y_grid
        
        x_grid = x_grid_rot
        y_grid = y_grid_rot
    
    return x_grid, y_grid


def _convert_latlong_to_xyz(site_pos):
    '''
    Only works for frame=ITRF (not for WGS84).
    '''
    import numpy as np
    x = site_pos['m2']['value']*np.cos(site_pos['m1']['value'])*np.cos(site_pos['m0']['value'])
    y = site_pos['m2']['value']*np.cos(site_pos['m1']['value'])*np.sin(site_pos['m0']['value'])
    z = site_pos['m2']['value']*np.sin(site_pos['m1']['value'])
    
    site_pos['m0']['unit'] = 'm'
    site_pos['m0']['value'] = x
    site_pos['m1']['unit'] = 'm'
    site_pos['m1']['value'] = y
    site_pos['m2']['value'] = z
    
def _convert_xyz_to_latlong(site_pos):
    '''
    Only works for frame=ITRF (not for WGS84).
    '''
    import numpy as np
    p = np.array([site_pos['m0']['value'],site_pos['m1']['value'],site_pos['m2']['value']])
    r = np.sqrt(np.sum(p**2))
    lon = np.arctan2(p[1],p[0])
    lat =  np.arcsin(p[2]/r)
    height = r
    
    site_pos['m0']['unit'] = 'rad'
    site_pos['m0']['value'] = lon
    site_pos['m1']['unit'] = 'rad'
    site_pos['m1']['value'] = lat
    site_pos['m2']['value'] = height





################################################
#CASACORE derived functions, not used

#cs functions: casa versions.
def _cs_directional_cosine(ra_dec):
    # In https://arxiv.org/pdf/astro-ph/0207413.pdf see equation 160
    #ra_dec (RA,DEC)
    #lmn = np.zeros((3,),dtype=numba.f8)
    lmn = np.zeros((3,))
    lmn[0] = np.cos(ra_dec[0])*np.cos(ra_dec[1])
    lmn[1] = np.sin(ra_dec[0])*np.cos(ra_dec[1])
    lmn[2] = np.sin(ra_dec[1])
    return lmn


def _cs_calc_rotation_mats(ra_dec_in,ra_dec_out,rotation_parms):
    #ra_dec_in
    #ra_dec_out
    #rotation_parms
    #'common_tangent_reprojection'
    #'reproject'
    #https://github.com/casacore/casacore/blob/dbf28794ef446bbf4e6150653dbe404379a3c429/measures/Measures/UVWMachine.cc
    #https://github.com/casacore/casacore/blob/dbf28794ef446bbf4e6150653dbe404379a3c429/measures/Measures/UVWMachine.h


    #The rotation matrix from a system that has a pole towards output
    #direction, into the standard system.
    out_rotmat = R.from_euler('XZ',[[- ra_dec_out[1] + np.pi/2, - ra_dec_out[0] + np.pi/2]]).as_matrix()[0]
    #print('rot3_p',out_rotmat)
    out_dir_cosine = _cs_directional_cosine(ra_dec_out)
    
    uvw_rotmat = np.zeros((3,3),np.double)
    phase_rotation = np.zeros((3,),np.double)
    
    # Define rotation to a coordinate system with pole towards in-direction
    # and X-axis W; by rotating around z-axis over -(90-long); and around
    # x-axis (lat-90).
    in_rotmat = R.from_euler('ZX',[[ra_dec_in[0] - np.pi/2 , ra_dec_in[1] - np.pi/2]]).as_matrix()[0]
    
#    print(-np.pi/2, ra_dec_in[0])
#    aR_z = R.from_euler('Z',[[-np.pi/2 + ra_dec_in[0]]]).as_matrix()[0]
#    print('R_z',aR_z)
#
#    aR_x = R.from_euler('X',[[- np.pi/2 + ra_dec_in[1]]]).as_matrix()[0]
#    print('R_x ',aR_x)
#    #print('rot1_p',in_rotmat)
#    print('compare',in_rotmat-aR_z@aR_x)
    R_zx_si = _func_R_z(ra_dec_in[0] - np.pi/2)@_func_R_x(ra_dec_in[1] - np.pi/2) #in_rotmat
    R_xz_is = _func_R_x(-ra_dec_in[1] + np.pi/2)@_func_R_z(- ra_dec_in[0] + np.pi/2) # R_xz_is = R_zx_si.T
    #print('@@@@@@@@@@@in_rotmat',mat_dis(R_zx_si,in_rotmat))
    
    R_xz_os = _func_R_x(- ra_dec_out[1] + np.pi/2)@_func_R_z(- ra_dec_out[0] + np.pi/2) #out_rotmat
    R_zx_so = _func_R_z(ra_dec_out[0] - np.pi/2)@_func_R_x(ra_dec_out[1] - np.pi/2)
    #print('@@@@@@@@@@@out_rotmat',mat_dis(R_xz_os,out_rotmat))
    
    uvw_rotmat = np.matmul(out_rotmat,in_rotmat).T
    
    #print('@@@@@@@@@@@uvw_rotmat',mat_dis(uvw_rotmat,(R_xz_os@R_zx_si ).T))
    #print('@@@@@@@@@@@uvw_rotmat',mat_dis(uvw_rotmat,(R_xz_is@R_zx_so)))
    
    #print('uvrot_p',uvw_rotmat[:,:])
    uvw_proj_rotmat = None
    
    if rotation_parms['reproject'] == True:
        #Get the rotation matrix which re-projects an uv-plane onto another reference direction:
        # around x-axis (out-lat - 90)
        # around z-axis (out-long - in-long)
        # around x-axis (90 - in-lat) and normalise
        proj_out_rotmat = np.eye(3)
        temp = R.from_euler('XZX',[[-np.pi/2 + ra_dec_out[1], ra_dec_out[0] - ra_dec_in[0], np.pi/2 - ra_dec_in[1] ]]).as_matrix()[0]
        proj_out_rotmat[0,0] = temp[1,1]/temp[2,2]
        proj_out_rotmat[1,1] = temp[0,0]/temp[2,2]
        proj_out_rotmat[0,1] = temp[1,0]/temp[2,2]
        proj_out_rotmat[1,0] = temp[0,1]/temp[2,2]
        uvw_proj_rotmat = np.matmul(uvw_rotmat,proj_out_rotmat)
    #print('rot4_p',proj_out_rotmat)
    
    #print('uvproj_p',uvw_rotmat[:,:])
    
    if rotation_parms['common_tangent_reprojection'] == True:
        uvw_rotmat[2,0:2] = 0.0 # (Common tangent rotation needed for joint mosaics, see last part of FTMachine::girarUVW in CASA)
    
    in_dir_cosine = _cs_directional_cosine(ra_dec_in)
    #print("i_field, field, new",i_field,field_phase_center_cosine,new_phase_center_cosine)
    phase_rotation = np.matmul(out_rotmat,(out_dir_cosine - in_dir_cosine))
    
    print('in_rotmat',in_rotmat)
    print('out_rotmat',out_rotmat)
    print('out_dir_cosine',out_dir_cosine)
    print('in_dir_cosine',in_dir_cosine)
    print('phase_rotation',phase_rotation)
    print((out_rotmat@(out_dir_cosine - in_dir_cosine)).shape)
    print('out_rotmat,out_dir_cosine',np.matmul(out_rotmat,out_dir_cosine))
    print('out_rotmat,-in_dir_cosine',np.matmul(out_rotmat,-in_dir_cosine))
    
    #print('phrot_p',phase_rotation)
    
    return uvw_rotmat, uvw_proj_rotmat, phase_rotation

