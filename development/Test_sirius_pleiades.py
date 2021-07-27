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
    from astropy.wcs import WCS
    import time
    deg_to_rad = np.pi/180
    from sirius import make_ant_sky_jones, simulation
    from sirius._sirius_utils._ant_jones_term import _compute_rot_coords
    from sirius._sirius_utils._calc_parallactic_angles import _calc_parallactic_angles, _find_optimal_set_angle
    
    from astropy.time import Time
    from astropy.coordinates import (EarthLocation, SkyCoord,
                                     AltAz, CIRS)
    import astropy.units as u
    
    import pkg_resources

    tel_dir = pkg_resources.resource_filename('casadata', 'data/telescope_layout/data/vla.d.tel.zarr')
    tel_xds = xr.open_zarr(tel_dir)
    
    
    mxds = read_vis('point_source_sim_vis/mosaic_pleiades_sim_dovp_True.vis.zarr')

    
    mxds.attrs['xds0'] = mxds.attrs['xds0']#.isel(time=slice(80,81))
    print(mxds.xds0.time)
    
    #(_)_s dimension can be singleton (value is reused)
    
    #########Setup parameters for uvw calculation###########
    ant_pos = mxds.ANTENNA.POSITION.values # [n_ant x 3]
    time_str = mxds.xds0.time.values        # [n_time]
    site = 'VLA'
    
    phase_center_ra_dec = mxds.FIELD.PHASE_DIR.sel(field_id=mxds.xds0.FIELD_ID.isel(baseline=0).values).values[:,0,:] #n_time x 2
    print('phase_center_ra_dec',phase_center_ra_dec)
    print('field ids',mxds.xds0.FIELD_ID.isel(baseline=0).values)
    
    ant1 = mxds.xds0.ANTENNA1.values
    ant2 = mxds.xds0.ANTENNA2.values
    
    #########################################################
    
    uvw_parms = {}
    uvw_parms['calc_method'] = 'astropy'
    uvw_parms['site'] = 'vla'
    uvw_parms['auto_corr'] = False
    
    uvw = mxds.xds0.UVW.values
    print(uvw.shape)
    
    ##################### Setup Beam Models ##############################
    airy_disk_parms =  {'pb_func':'casa_airy','dish_diameter':24.5,'blockage_diameter':0.0}
    beam_models = [airy_disk_parms]
    n_ant = len(ant_pos)
    beam_model_map = np.zeros(n_ant,dtype=int)

    beam_parms = {}
    beam_parms['pa_radius'] = 0.2
    beam_parms['mueller_selection'] = np.array([0,15])

    freq_chan = mxds.xds0.chan.values
    pol = mxds.xds0.pol.values
    ######################
    
    #####################Setup sources
    #pointing_ra_dec:  [n_time, n_ant, 2]          (singleton: n_time, n_ant)
    pointing_ra_dec = None #np.zeros((1, 1, 2)) #Singleton
    
    #point_source_ra_dec:  [n_time, n_point_sources, 2]          (singleton: n_time)
    ra = ['03h47m29.1s','03h49m09.7s','03h46m19.6s','  03h45m49.6s', '03h44m52.5s']
    dec = ['+24d06m18.5s','+24d03m12.3s','+23d56m54.1s','+24d22m03.9s','+24d06m48.0s']
    point_source_skycoord = SkyCoord(ra=ra,dec=dec,frame='fk5') #sim
    
    '''
    03h47m29.1s +24d06m18.5s
    03h49m09.7s +24d03m12.3s
    03h46m19.6s +23d56m54.1s
    03h45m49.6s +24d22m03.9s
    03h44m52.5s +24d06m48.0s
    '''
    point_source_ra_dec = np.array([point_source_skycoord.ra.rad,point_source_skycoord.dec.rad]).T[None,:,:]
    
  
    
    
    # NB n_pol is no longer singleton
    #point_source_flux: [n_time, n_chan, n_pol, n_point_sources] (singleton: n_time, n_chan)
    point_source_flux = np.array([2.17, 0, 0, 2.17])[None,None,:,None] #has to be in instrument polarization: [RR,RL,LR,LL] or [XX,XY,YX,YY]. All 4 values are needed.
    point_source_flux = np.tile(point_source_flux,(1,1,1,5))
    
    pb_limit = 0.0045
    
    ###############################
    vis_data, uvw = simulation(point_source_flux, point_source_ra_dec, pointing_ra_dec, phase_center_ra_dec, beam_parms,beam_models,beam_model_map,uvw_parms, ant_pos, time_str, freq_chan, pol, ant1, ant2, pb_limit, uvw) #None can be changed to uvw
    
    #print(vis_data)
    
    casa_vis_data = mxds.xds0.DATA.data.compute()
    #print(casa_vis_data)
    
    
    casa_vis_data = mxds.xds0.DATA.data.compute()
    dif = np.ravel((vis_data[:,:,:,0]-casa_vis_data[:,:,:,0])) #/np.abs(casa_vis_data[:,:,:,0])
    
    plt.figure()
    plt.plot(np.abs(dif))
    plt.xlabel('Vis Number (ravel)')
    plt.ylabel('Abs Dif')
    
    plt.figure()
    plt.plot(np.imag(dif))
    plt.xlabel('Vis Number (ravel)')
    plt.ylabel('Imag Dif')
   
    plt.figure()
    plt.plot(np.real(dif))
    plt.xlabel('Vis Number (ravel)')
    plt.ylabel('Real Dif')
    
    plt.figure()
    plt.plot(np.abs(np.ravel(vis_data[:,:,:,0])),label='Sirius')
    plt.plot(np.abs(np.ravel(casa_vis_data[:,:,:,0])),label='CASA')
    plt.legend()
    plt.xlabel('Vis Number (ravel)')
    plt.ylabel('abs Vis')
    
    
    plt.show()
    
    #from cngi.vis import apply_flags
    from ngcasa.imaging import make_imaging_weight
    from ngcasa.imaging import make_image
    from ngcasa.imaging import make_pb
    from ngcasa.imaging import make_gridding_convolution_function
    from cngi.dio import write_image
    import dask
    import dask.array as da
    
    rad_to_deg =  180/np.pi
    deg_to_rad = np.pi/180
    arcsec_to_deg = 1/3600
    arcsec_to_rad = np.pi/(180*3600)
    
    #mxds = apply_flags(mxds, 'xds0', flags='FLAG')
    mxds.attrs['xds1'] = mxds.attrs['xds0'].isel(pol=slice(0,1))
    mxds.attrs['xds1']['DATA']
    xr.DataArray(da.from_array(vis_data[:,:,:,0][:,:,:,None],chunks=mxds.attrs['xds1']['DATA'].chunks),coords=mxds.attrs['xds1']['DATA'].coords)
    from ngcasa.imaging import direction_rotate
    
    grid_parms = {}
    grid_parms['chan_mode'] = 'cube'
    grid_parms['image_size'] = [1600,800]
    grid_parms['cell_size'] = [5,5]

    point_source_skycoord = SkyCoord(ra='03h47m29.1s',dec='+24d06m18.5s',frame='fk5')
    grid_parms['phase_center'] = np.array([point_source_skycoord.ra.rad,point_source_skycoord.dec.rad])
    
    
    sel_parms = {}
    sel_parms['xds'] = 'xds1' #KEEP USING XDS1 THROUGHOUT
    sel_parms['data_group_in_id'] = 0 #CHANGE TO DATA GROUP 2 AFTER ROTATION

    rotation_parms = {}
    rotation_parms['new_phase_center'] = grid_parms['phase_center']
    rotation_parms['common_tangent_reprojection'] = True
    rotation_parms['single_precision'] = False

    mxds = direction_rotate(mxds, rotation_parms, sel_parms)
    
    imaging_weights_parms = {}
    imaging_weights_parms['weighting'] = 'natural'
    
    sel_parms = {}
    sel_parms['xds'] = 'xds1'
    sel_parms['data_group_in_id'] = 2
    
    mxds = make_imaging_weight(mxds, imaging_weights_parms, grid_parms, sel_parms)
    
    vis_sel_parms = {}
    vis_sel_parms['xds'] = 'xds1'
    vis_sel_parms['data_group_in_id'] = 2
    
    img_sel_parms = {}
    img_sel_parms['data_group_out_id'] = 0
    
    from cngi.dio import write_image
    write_imagegrid_parms = {}
    grid_parms['chan_mode'] = 'cube'
    grid_parms['image_size'] = [1600,800]
    grid_parms['cell_size'] = [5,5]
    
    point_source_skycoord = SkyCoord(ra='03h47m29.1s',dec='+24d06m18.5s',frame='fk5')
    grid_parms['phase_center'] = np.array([point_source_skycoord.ra.rad,point_source_skycoord.dec.rad])
    gcf_parms = {}
    gcf_parms['function'] = 'alma_airy'
    gcf_parms['list_dish_diameters'] = np.array([24.5])
    gcf_parms['list_blockage_diameters'] = np.array([0])
    #gcf_parms['max_support'] = [24, 24]
    #gcf_parms['oversampling'] = [10, 10]
    gcf_parms['max_support'] = [48, 48]
    gcf_parms['oversampling'] = [10, 10]
    unique_ant_indx = mxds.ANTENNA.DISH_DIAMETER.values
    unique_ant_indx[unique_ant_indx == 25.0] = 0
    mxds.ANTENNA.DISH_DIAMETER.values
    gcf_parms['unique_ant_indx'] = unique_ant_indx.astype(int)
    gcf_parms['phase_center'] = grid_parms['phase_center']
    sel_parms = {}
    sel_parms['xds'] = 'xds1'
    sel_parms['data_group_in_id'] = 2
    gcf_xds = make_gridding_convolution_function(mxds, gcf_parms, grid_parms, sel_parms)
    dask.compute(gcf_xds)
    
    from ngcasa.imaging import make_image_with_gcf
    from ngcasa.imaging import make_mosaic_pb
    img_xds = xr.Dataset() #empty dataset
    vis_sel_parms = {}
    vis_sel_parms['xds'] = 'xds1'
    vis_sel_parms['data_group_in_id'] = 2

    img_xds = make_mosaic_pb(mxds,gcf_xds,img_xds,vis_sel_parms,img_sel_parms,grid_parms)

    vis_select_parms = {}
    vis_select_parms['xds'] = 'xds1'
    vis_select_parms['data_group_in_id'] = 2

    img_select_parms = {}
    img_select_parms['data_group_in_id'] = 0
    img_select_parms['data_group_out_id'] = 0

    norm_parms = {}
    norm_parms['norm_type'] = 'flat_sky'

    img_xds = make_image_with_gcf(mxds,gcf_xds, img_xds, grid_parms, norm_parms, vis_select_parms, img_select_parms)
    #Need make_image_with_gcf instead of make_image, add make_gridding_convolutional_function, direction_rotate
    #Note: if no primary beam, use 'flat_noise' instead of 'flat_sky'
        
        
    #Select chan 1
    chan = 1
    ngcasa_image_name = 'IMAGE'
    pb_limit = 0.2
    #extent = extent=(np.min(casa_img_xds.m),np.max(casa_img_xds.m),np.min(casa_img_xds.l),np.max(casa_img_xds.l))

    mosaic_pb = img_xds.PB.isel(chan=chan)
    mosaic_img = img_xds[ngcasa_image_name].isel(chan=chan)
    mosaic_img = mosaic_img.where(mosaic_pb > np.max(mosaic_pb)*pb_limit,other=np.nan)
    
    plt.figure()
    #plt.imshow(img_xds.IMAGE.isel(chan=chan,time=0,pol=0))
    plt.imshow(mosaic_img[:, :, 0, 0])


    '''
    print(point_source_ra_dec.shape)
    
    
    #####################################
    print(mxds.xds0.chan.values)
    
    zpc_xds = xr.open_zarr('../../../cngi_reference/dish_models/data/EVLA_avg_zcoeffs_SBand_lookup.zpc.zarr')
    
    print(zpc_xds)
    
    list_zpc_dataset = [zpc_xds]
    pb_parms = {}
    pb_parms['fov_scaling'] = 15
    #pb_parms['mueller_selection'] = np.array([0])#np.arange(16) #np.array([0,5,10,15])#np.arange(16)
    pb_parms['mueller_selection'] = np.arange(16)
    pb_parms['zernike_freq_interp'] = 'nearest'
    pb_parms['freq'] = mxds.xds0.chan.values
    
    ##########PA Calc
    telescope_name = mxds.attrs['OBSERVATION'].TELESCOPE_NAME.values[0]
    if telescope_name=='EVLA': telescope_name='VLA'
    observing_location = EarthLocation.of_site(telescope_name)
    
    obs_time = mxds.xds0.time.values.astype(str)[:,None]
    obs_time = Time(obs_time, format='isot', scale='utc', location=observing_location)
    print(obs_time)

    phase_center = SkyCoord(ra=array_phase_center_ra_dec[:,0]*u.rad, dec=array_phase_center_ra_dec[:,1]*u.rad, frame='fk5')
    
    pa = _calc_parallactic_angles(mxds.xds0.time.values,observing_location,phase_center)
    #print(pa.shape)
    val_step = 0.2
    pa_subset,vals_dif = _find_optimal_set_angle(pa[:,None],val_step)
    #print(pa_subset)
    #print(vals_dif)
    

    
    ##########PA Calc

    pb_parms['pa'] =  pa_subset#np.array([0.987*np.pi/4, 0.0, np.pi/4, np.pi/2, 0.986*np.pi/4])
    #pb_parms['pa'] = np.array([0.0, 0.987*np.pi/4, np.pi/4, np.pi/2])
    grid_parms = {}
    #grid_parms['image_size'] = np.array([1000,1000])
    grid_parms['image_size'] = np.array([2000,2000])
    
    print(zpc_xds.dish_diam,pb_parms['freq'])
    # time, ant, chan, pol, l, m
    J_xds = make_ant_sky_jones(list_zpc_dataset,pb_parms,grid_parms) #[None,None,:,:,:,:]
    
    #M = make_mueler_mat(J, inv=False)
    
    #M_inv = make_mueler_mat(J, inv=True)
    parallactic_angle = 0.987*np.pi/4
    cell_size = np.array([J_xds.l[1]-J_xds.l[0],J_xds.m[1]-J_xds.m[0]])
    print(cell_size)
    image_size = grid_parms['image_size']
    
    
    x_grid, y_grid = _compute_rot_coords(image_size,cell_size,parallactic_angle)
    
    print(J_xds)
    '''
  

    
    
    
    
    '''
    #x = (39)*cell_size[0]
    #y = (40)*cell_size[1]
    pol = 0
    x = (39)*cell_size[0]
    y = (40)*cell_size[1]
    pa_sel = pb_parms['pa'][0]#0.987*np.pi/4
    
    r0 = J_xds.J.isel(model=0,pa=0,chan=0,pol=pol).interp(l=x,m=y).values
    print('pa_sel from generated rotated',r0)
    
    for i in [1,2,3,4]:
        x_rot, y_rot  = rot_coord(x,y,pa_sel-pb_parms['pa'][i])
        r = J_xds.J.isel(model=0,pa=i,chan=0,pol=pol).interp(l=x_rot,m=y_rot,method='linear').values
        i_dif = 100*np.imag(r0-r)/np.imag(r0)
        r_dif = 100*np.real(r0-r)/np.real(r0)
        print('pa '+str(i)+'                                            ',r,i_dif,r_dif)
    
    print(pb_parms['pa']-pa_sel)
    
    plt.figure()
    plt.imshow(np.abs(J_xds.J[0,0,0,pol,:,:]))
    
    plt.figure()
    plt.imshow(np.abs(J_xds.J[0,1,0,pol,:,:]))
    
    plt.figure()
    plt.imshow(np.abs(J_xds.J[0,3,0,pol,:,:]))
    
    plt.figure()
    plt.imshow(np.abs(J_xds.J[0,4,0,pol,:,:]))
    '''
    
    
    '''
    x_rot_1, y_rot_1  = rot_coord(x,y,pa_sel-pb_parms['pa'][1])
    x_rot_2, y_rot_2  = rot_coord(x,y,pa_sel-pb_parms['pa'][2])
    x_rot_3, y_rot_3  = rot_coord(x,y,pa_sel-pb_parms['pa'][3])
    x_rot_4, y_rot_4  = rot_coord(x,y,pa_sel-pb_parms['pa'][4])
    
   
    r1 = J_xds.J.isel(model=0,pa=1,chan=0,pol=0).interp(l=x_rot_1,m=y_rot_1,method='linear').values
    print('pa 1                                            ',r1,)
    
    r2 = J_xds.J.isel(model=0,pa=2,chan=0,pol=0).interp(l=x_rot_2,m=y_rot_2,method='linear').values
    print('pa 2                                            ',r2,)
    
    r3 = J_xds.J.isel(model=0,pa=3,chan=0,pol=0).interp(l=x_rot_3,m=y_rot_3,method='linear').values
    print('pa 3                                            ',r3,)
    
    r4 = J_xds.J.isel(model=0,pa=4,chan=0,pol=0).interp(l=x_rot_4,m=y_rot_4,method='linear').values
    print('pa 4                                            ',r4,)
    '''
    
    '''
    
 
    J_rot = np.zeros(J_xds.J.data.shape,dtype=complex)
    
    #print(x_grid)
    #print(y_grid)
    #x = (139- image_size[0]//2)*cell_size[0]
    #y = (140- image_size[1]//2)*cell_size[1]
    x = (39)*cell_size[0]
    y = (40)*cell_size[1]
    parallactic_angle = 0.987*np.pi/4

#    rot_mat = np.array([[np.cos(parallactic_angle),-np.sin(parallactic_angle)],[np.sin(parallactic_angle),np.cos(parallactic_angle)]])
#    x_rot = np.cos(parallactic_angle)*x + np.sin(parallactic_angle)*y
#    y_rot = - np.sin(parallactic_angle)*x + np.cos(parallactic_angle)*y
    x_rot, y_rot  = rot_coord(x,y,parallactic_angle)
    
    x_rot_90, y_rot_90  = rot_coord(x,y,parallactic_angle-np.pi/2)
    
    x_rot_45, y_rot_45  = rot_coord(x,y,parallactic_angle-np.pi/4)
    
    print(x_rot,y_rot)
    print('x,y rot coord',x_grid[139,140],y_grid[139,140])
    
    a1 = J_xds.J.isel(model=0,pa=1,chan=0,pol=0).interp(l=x,m=y).values
    print('J_xds.J 1 ',a1)
        
    a2 = J_xds.J.isel(model=0,pa=0,chan=0,pol=0).interp(l=x_rot,m=y_rot,method='linear').values
    print('J_xds.J 0 rot coords ',a2)
    
    funsc = interp2d(J_xds.m.values,J_xds.l.values,np.real(J_xds.J.isel(model=0,pa=0,chan=0,pol=0).values), kind='quintic')
    print('scipy ', funsc(y_rot,x_rot))
    
    a3 = J_xds.J.isel(model=0,pa=3,chan=0,pol=0).interp(l=x_rot_90,m=y_rot_90).values
    print('J_xds.J 3 rot coords ',a3)
    
    a4 = J_xds.J.isel(model=0,pa=2,chan=0,pol=0).interp(l=x_rot_45,m=y_rot_45).values
    print('J_xds.J 4 rot coords ',a4)
    
    #pb_parms['pa'] = np.array([0.0, 0.987*np.pi/4, np.pi/4, np.pi/2])
    
    
    print(a1-a2)
    print(100*(np.abs(a1-a2)/np.abs(a1)))
    '''
    
    
    
    '''
    for i in range(1):
        for x in range(J_xds.J.shape[2]):
            print(x)
            for y in range(J_xds.J.shape[3]):
                J_rot[0,i,x,y] =J_xds.J.isel(pb_parm_indx=0,pol=i).interp(l=x_grid[x,y],m=y_grid[x,y]).values


    print('J_rot',J_rot[0,0,139,140])
    print('J_xds.J 0 ',J_xds.J[0,0,139,140])
    print('J_xds.J 1 ',J_xds.J[1,0,139,140])
    print('x,y coord',J_xds.l[139],J_xds.m[140])
    print('x,y rot coord',x_grid[139,140],y_grid[139,140])
    
    
    plt.figure()
    plt.imshow(np.abs(J_xds.J[0,0,:,:]))

    plt.figure()
    plt.imshow(np.abs(J_rot[0,0,:,:]))

    plt.figure()
    plt.imshow(np.abs(J_xds.J[1,0,:,:]))

    plt.figure()
    plt.imshow((np.abs(J_xds.J[1,0,:,:]-J_rot[0,0,:,:])/np.abs(J_xds.J[1,0,:,:]))[75:125,75:125])
    plt.colorbar()
    
    
    plt.figure()
    plt.imshow((np.abs(J_xds.J[1,0,:,:]-J_xds.J[0,0,:,:])/np.abs(J_xds.J[1,0,:,:]))[75:125,75:125])
    plt.colorbar()
    '''
    
    '''
    #n = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(M[0,0,:,:] + M[0,5,:,:] + M[0,10,:,:] + M[0,15,:,:])))
    n = ifft_2d(fft_2d(M[0,0,:,:]) + fft_2d(M[0,5,:,:]) + fft_2d(M[0,10,:,:]) + fft_2d(M[0,15,:,:]))
    
    Y = M_inv*M
    
    plt.figure()
    plt.imshow(np.abs(M_inv[0,0,:,:]))
    
    plt.figure()
    plt.imshow(np.abs(M_inv[0,5,:,:]))
    
    plt.figure()
    plt.imshow(np.abs(M_inv[0,10,:,:]))
    
    plt.figure()
    plt.imshow(np.abs(M_inv[0,15,:,:]))
    
    plt.figure()
    plt.imshow(np.abs(M[0,0,:,:]))
    
    plt.figure()
    plt.imshow(np.abs(M[0,5,:,:]))
    
    plt.figure()
    plt.imshow(np.abs(M[0,10,:,:]))
    
    plt.figure()
    plt.imshow(np.abs(M[0,15,:,:]))
    '''
    
    '''
    plt.figure()
    plt.imshow(np.real(Y[0,0,:,:]/Y[0,0,:,:]))
    
    plt.figure()
    plt.imshow(np.imag(Y[0,0,:,:]))
    
    
    plt.figure()
    plt.imshow(np.real(Y[0,1,:,:]))
    
    plt.figure()
    plt.imshow(np.imag(Y[0,1,:,:]))
    
    plt.figure()
    plt.imshow(np.real(n))
    
    plt.figure()
    plt.imshow(np.imag(n))
    '''
    
    '''
    plt.figure()
    plt.imshow(np.abs(J.J[0,0,:,:]))
    
    plt.figure()
    plt.imshow(np.abs(J.J[0,1,:,:]))
    
    plt.figure()
    plt.imshow(np.abs(J.J[0,2,:,:]))
    
    plt.figure()
    plt.imshow(np.abs(J.J[0,3,:,:]))

    print(J)
    
    parallactic_angle = 0.987*np.pi/2
    cell_size = np.array([J.l[1]-J.l[0],J.m[1]-J.m[0]])
    print(cell_size)
    image_size = grid_parms['image_size']
    l_r, m_r = _compute_rot_coords(image_size,cell_size,parallactic_angle)
    
    J_rot = J.loc()
    
 
    print(ant_pb_planes.shape)
    
    # time, ant, chan, pol, l, m -> time, unique_baselines , chan, pol, l, m
    M = make_mueler_mat(ant_pb_planes, inv=False)
    
    M_inv = make_mueler_mat(ant_pb_planes, inv=True)
    
    n = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(M_inv[0,0,0,0,:,:] + M_inv[0,0,0,5,:,:] + M_inv[0,0,0,10,:,:] + M_inv[0,0,0,15,:,:])))
    
    print(M.shape)
    
    plt.figure()
    plt.imshow(np.abs(n))
    
    plt.figure()
    plt.imshow(np.real(n))
    
    plt.figure()
    plt.imshow(np.imag(n))
    
    plt.figure()
    plt.imshow(np.abs(M[0,0,0,0,:,:]*M_inv[0,0,0,0,:,:]))
    
    plt.figure()
    plt.imshow(np.real(M[0,0,0,0,:,:]*M_inv[0,0,0,0,:,:]))
    
    i_m = 0
    #lm to uv
    A = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(M[0,0,0,i_m,:,:]*M_inv[0,0,0,i_m,:,:])))
    
    plt.figure()
    plt.imshow(np.abs(A/n))

    plt.figure()
    plt.imshow(np.real(A/n))
    
    plt.figure()
    plt.imshow(np.imag(A/n))
    '''
    
#
#    plt.figure()
#    plt.imshow(np.real(pb_planes[1,0,:,:]))
#
#    plt.figure()
#    plt.imshow(np.imag(pb_planes[0,0,:,:]))
#
#    plt.figure()
#    plt.imshow(np.imag(pb_planes[1,0,:,:]))
#
#
#
#    print(pb_planes[1,0,945,1095])
#
#    x_grid = 55
#    y_grid = 95
#    parallactic_angle = -0.987*np.pi/2
#
#    rot_mat = np.array([[np.cos(parallactic_angle),-np.sin(parallactic_angle)],[np.sin(parallactic_angle),np.cos(parallactic_angle)]])
#    x_grid_rot = np.cos(parallactic_angle)*x_grid + np.sin(parallactic_angle)*y_grid
#    y_grid_rot = - np.sin(parallactic_angle)*x_grid + np.cos(parallactic_angle)*y_grid
#
#    print(x_grid_rot)
#    print(y_grid_rot)
#
#    print(pb_planes[0,0,945,1095])
#    print(pb_planes[0,0,1094,1057])
#    print(pb_planes[1,0,945,1095])
    

  
    
#    plt.figure()
#    plt.imshow(np.imag(q))
#
#    plt.figure()
#    plt.imshow(np.imag(q))
    
    plt.show()



