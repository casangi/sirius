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

#@jit(nopython=True,cache=True,nogil=True)
def _directional_cosine(ra_dec):
   '''
   # In https://arxiv.org/pdf/astro-ph/0207413.pdf see equation 160
   ra_dec (RA,DEC)
   '''
   
   #lmn = np.zeros((3,),dtype=numba.f8)
   lmn = np.zeros((3,))
   lmn[0] = np.cos(ra_dec[0])*np.cos(ra_dec[1])
   lmn[1] = np.sin(ra_dec[0])*np.cos(ra_dec[1])
   lmn[2] = np.sin(ra_dec[1])
   return lmn
    
def _calc_rotation_mats(ra_dec_in,ra_dec_out,rotation_parms):
    '''
    ra_dec_in
    ra_dec_out

    rotation_parms
    'common_tangent_reprojection'
    'reproject'

    #https://github.com/casacore/casacore/blob/dbf28794ef446bbf4e6150653dbe404379a3c429/measures/Measures/UVWMachine.cc
    #https://github.com/casacore/casacore/blob/dbf28794ef446bbf4e6150653dbe404379a3c429/measures/Measures/UVWMachine.h
    '''

    #The rotation matrix from a system that has a pole towards output
    #direction, into the standard system.
    out_rotmat = R.from_euler('XZ',[[np.pi/2 - ra_dec_out[1], - ra_dec_out[0] + np.pi/2]]).as_matrix()[0]
    #print('rot3_p',out_rotmat)
    out_dir_cosine = _directional_cosine(ra_dec_out)
    
    uvw_rotmat = np.zeros((3,3),np.double)
    phase_rotation = np.zeros((3,),np.double)
    
    # Define rotation to a coordinate system with pole towards in-direction
    # and X-axis W; by rotating around z-axis over -(90-long); and around
    # x-axis (lat-90).
    in_rotmat = R.from_euler('ZX',[[-np.pi/2 + ra_dec_in[0],- np.pi/2 + ra_dec_in[1]]]).as_matrix()[0]
    
    
    print(-np.pi/2, ra_dec_in[0])
    print('R_z',R.from_euler('Z',[[-np.pi/2 + ra_dec_in[0]]]).as_matrix()[0])
    
    print('R_x ',R.from_euler('X',[[- np.pi/2 + ra_dec_in[1]]]).as_matrix()[0])
    #print('rot1_p',in_rotmat)
    uvw_rotmat = np.matmul(out_rotmat,in_rotmat).T
    
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
    
    in_dir_cosine = _directional_cosine(ra_dec_in)
    #print("i_field, field, new",i_field,field_phase_center_cosine,new_phase_center_cosine)
    phase_rotation = np.matmul(out_rotmat,(out_dir_cosine - in_dir_cosine))
    
    print('in_rotmat',in_rotmat)
    print('out_rotmat',out_rotmat)
    print('out_dir_cosine',out_dir_cosine)
    print('in_dir_cosine',in_dir_cosine)
    print('phase_rotation',phase_rotation)
    print('out_rotmat,out_dir_cosine',np.matmul(out_rotmat,out_dir_cosine))
    print('out_rotmat,-in_dir_cosine',np.matmul(out_rotmat,-in_dir_cosine))
    
    #print('phrot_p',phase_rotation)
    
    return uvw_rotmat, uvw_proj_rotmat, phase_rotation
