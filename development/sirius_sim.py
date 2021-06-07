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
c = 299792458
#from direction_rotate import _calc_rotation_mats
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import xarray as xr
    import dask.array as da
    import cngi.conversion.convert_ms as convert_ms
    import cngi.conversion.convert_image as convert_image
    from cngi.dio import read_vis
    import numpy as np
    from astropy.coordinates import SkyCoord
    from sirius import calc_vis
    from sympy import *
    from astropy.wcs import WCS
    import time

    #Load Data from converted CASA simulated ms (steal uvw values)
    #ms_file = 'point_source_sim_vis/point_source_sim_dovp_True.vis.zarr' #remember to change pb_parms['pb_func'] = 'casa_airy'
    ms_file = 'point_source_sim_vis/point_source_sim_dovp_False.vis.zarr' #remember to change pb_parms['pb_func'] = 'none'
    #ms_file = 'zenith_point_source_sim_dovp_False.vis.zarr'
    mxds = read_vis(ms_file)
    print(mxds)
    vis_xds = read_vis(ms_file).xds0
    cas_vis_data = vis_xds.DATA.data.compute()
    uvw = vis_xds.UVW.data.compute()
    freq_chan = vis_xds.chan.values
    
    #Sirius Simulation
    
    #Setup pointing centre
    #pointing_ra_dec:  [n_time, n_baseline, 2]                   (singleton: n_time, n_baseline)
    pointing_skycoord = SkyCoord(ra='19h59m28.5s',dec='+40d44m01.5s',frame='fk5') #sim
    #pointing_skycoord = SkyCoord(ra='0h0m0.0s',dec='90d00m00.0s',frame='fk5') #zenith
    pointing_ra_dec = np.array([pointing_skycoord.ra.rad,pointing_skycoord.dec.rad])[None,None,:]
    
    #Setup point source skycoord
    #point_source_ra_dec:  [n_time, n_point_sources, 2]          (singleton: n_time)
    point_source_skycoord = SkyCoord(ra='19h59m0.0s',dec='+40d51m01.5s',frame='fk5') #sim
    #point_source_skycoord = SkyCoord(ra='19h59m0.0s',dec='+89d54m01.5s',frame='fk5') #zenith
    point_source_ra_dec = np.array([point_source_skycoord.ra.rad,point_source_skycoord.dec.rad])[None,None,:]

    #Source flux
    #point_source_flux: [n_time, n_chan, n_pol, n_point_sources] (singleton: n_time, n_chan, n_pol)
    point_source_flux = np.array([2.17])[None,None,None,:]
    
    #Primary Beam Model parameters
    pb_parms = {}
    #pb_parms['pb_func'] = 'casa_airy' # casa_airy/airy/none
    pb_parms['pb_func'] = 'none'
    pb_parms['dish_diameter'] = 24.5
    pb_parms['blockage_diameter'] = 0.0
    pb_parms['ipower'] = 2
    
    start = time.time()
    vis_data = calc_vis(uvw,cas_vis_data.shape,point_source_flux,point_source_ra_dec,pointing_ra_dec,freq_chan,pb_parms)
    print('Compute Time', time.time()-start)
    

    abs_dif = np.ravel(np.abs(cas_vis_data[:,:,:,0]-vis_data[:,:,:,0])/np.abs(vis_data[:,:,:,0]))
    
    plt.figure()
    plt.plot(abs_dif)
    plt.xlabel('Vis Number (ravel)')
    plt.ylabel('Relative Change (out of 1)')

    ##############################################
    #Create Image using ngCASA
    ##############################################
    from cngi.vis import apply_flags
    from ngcasa.imaging import make_imaging_weight
    from ngcasa.imaging import make_image
    from ngcasa.imaging import make_pb
    from cngi.dio import write_image
    import dask
    import dask.array as da
    rad_to_deg =  180/np.pi
    deg_to_rad = np.pi/180
    arcsec_to_deg = 1/3600
    arcsec_to_rad = np.pi/(180*3600)

    mxds = apply_flags(mxds, 'xds0', flags='FLAG')
    mxds.attrs['xds1'] = mxds.attrs['xds0'].isel(pol=slice(0,1))
    mxds.attrs['xds1']['DATA'] = xr.DataArray(da.from_array(vis_data[:,:,:,0][:,:,:,None],chunks=mxds.attrs['xds1']['DATA'].chunks),coords=mxds.attrs['xds1']['DATA'].coords)

    imaging_weights_parms = {}
    imaging_weights_parms['weighting'] = 'natural'

    sel_parms = {}
    sel_parms['xds'] = 'xds1'
    sel_parms['data_group_in_id'] = 0
    
    grid_parms = {}
    grid_parms['chan_mode'] = 'cube'
    grid_parms['image_size'] = [200,400]
    grid_parms['cell_size'] = [20,20]
    grid_parms['phase_center'] = pointing_ra_dec[0,0,:]#mxds.FIELD.PHASE_DIR[0,0,:].data.compute()

    mxds = make_imaging_weight(mxds, imaging_weights_parms, grid_parms, sel_parms)

    vis_sel_parms = {}
    vis_sel_parms['xds'] = 'xds1'
    vis_sel_parms['data_group_in_id'] = 0

    img_sel_parms = {}
    img_sel_parms['data_group_out_id'] = 0

    img_xds = xr.Dataset() #empty dataset
    img_xds = make_image(mxds, img_xds, grid_parms, vis_sel_parms, img_sel_parms)

    #Select chan 1
    chan = 1

    plt.figure()
    plt.imshow(img_xds.IMAGE.isel(chan=chan,time=0,pol=0))
    
    ##############################################
    #Calculate pixel position
    ##############################################
    phase_center = grid_parms['phase_center']
    w = WCS(naxis=2)
    w.wcs.crpix = np.array(grid_parms['image_size'])//2
    w.wcs.cdelt = np.array([-20,20])*arcsec_to_deg
    w.wcs.crval = phase_center*rad_to_deg
    w.wcs.ctype = ['RA---SIN','DEC--SIN']

    lm_pix_pos = w.all_world2pix(point_source_ra_dec[0,:,:]*rad_to_deg, 1)
    print('source pix pos',lm_pix_pos)

    cell_size = np.array(grid_parms['cell_size'])*arcsec_to_rad
    cell_size[0] = -cell_size[0]
    image_center = np.array(grid_parms['image_size'])//2
    source_lm_pos = lm_pix_pos*cell_size - image_center*cell_size
    print('source lm pos',source_lm_pos)
    
    ##############################################
    #Plot Primary Beam
    ##############################################
    from sirius._sirius_utils._make_pb_symmetric import _casa_airy_disk
    
    grid_parms = {}
    grid_parms['chan_mode'] = 'continuum'
    grid_parms['image_size'] = np.array([200,400])
    grid_parms['cell_size'] = np.array([-20,20])*arcsec_to_rad
    grid_parms['phase_center'] = pointing_ra_dec[0,0,:]
    pol = [1]
    grid_parms['image_center'] = grid_parms['image_size']//2
    
    pb_parms = {}
    pb_parms['pb_func'] = 'casa_airy'
    pb_parms['list_dish_diameters'] = [24.5]
    pb_parms['list_blockage_diameters'] = [0.0]
    pb_parms['ipower'] = 2
    
    pb = _casa_airy_disk(freq_chan,pol,pb_parms,grid_parms)
    
    plt.figure()
    plt.imshow(pb[:,:,1,0,0])
    
    plt.show()
    
    
    
