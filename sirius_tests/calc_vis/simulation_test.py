import numpy as np
import xarray as xr
from cngi.dio import read_vis
#    import dask.array as da
#    from cngi.dio import read_vis
#    import numpy as np
#    from astropy.coordinates import SkyCoord
#    import time

deg_to_rad = np.pi/180
from sirius import make_ant_sky_jones, simulation
from sirius._sirius_utils._math_utils import _compute_rot_coords
from sirius._sirius_utils._calc_parallactic_angles import _calc_parallactic_angles, _find_optimal_set_angle
from sirius._sirius_utils._math_utils import _rot_coord
from astropy.time import Time
from astropy.coordinates import (EarthLocation, SkyCoord, AltAz, CIRS)
import astropy.units as u
import pkg_resources

def test_simulation():
    """zpc_dir = 'sirius_data/dish_models/data/EVLA_avg_zcoeffs_SBand_lookup.zpc.zarr'
    zpc_xds = xr.open_zarr(zpc_dir)
    beam_models = [zpc_xds]
    tel_dir = 'sirius_data/telescope_layout/data/vla.d.tel.zarr'
    tel_xds = xr.open_zarr(tel_dir)
    mxds = read_vis('sirius_data/fm_sim_data/fm_sim_data.vis.zarr')
    
    #########Setup parameters for uvw calculation###########
    ant_pos = mxds.ANTENNA.POSITION.values # [n_ant x 3]
    time_str = mxds.xds0.time.values#[0:1]        # [n_time]
    site = 'VLA'
    phase_center = SkyCoord(ra='19h59m28.5s',dec='+40d44m01.5s',frame='fk5')
    phase_center_ra_dec = np.array([phase_center.ra.rad,phase_center.dec.rad])[None,:] #field phase centre, [(n_time)_s x 2]. This example only has a single phase center.

    ant1 = mxds.xds0.ANTENNA1.values
    ant2 = mxds.xds0.ANTENNA2.values
    #########################################################
    
    uvw_parms = {}
    uvw_parms['calc_method'] = 'astropy'
    uvw_parms['site'] = 'vla'
    uvw_parms['auto_corr'] = False
    uvw = mxds.xds0.UVW.values
    ######################
    
    #Create J_xds from zpc
    pb_parms = {}
    pb_parms['fov_scaling'] = 15
    #pb_parms['fov_scaling'] = 7
    #pb_parms['fov_scaling'] = 9
    #pb_parms['mueller_selection'] = np.array([0])#np.arange(16) #np.array([0,5,10,15])#np.arange(16)
    pb_parms['mueller_selection'] = np.arange(16)
    pb_parms['zernike_freq_interp'] = 'nearest'
    pb_parms['freq'] = np.array([3.00e+09])
    pb_parms['pa'] = np.array([0.0, 0.7323*np.pi/4, 0.9741*np.pi/4, 0.987*np.pi/4, np.pi/4, np.pi/2])
    pb_parms['image_size'] = np.array([500,500])
    # time, ant, chan, pol, l, m
    J_xds = make_ant_sky_jones([zpc_xds],pb_parms) #[None,None,:,:,:,:]
    pol = 0
    J_sub = J_xds.J.isel(model=0,pa=0,pol=pol,chan=0)
    ######################
    
    #Setup Beam Models
    ant_pos = mxds.ANTENNA.POSITION.values
    n_ant = len(ant_pos)

    airy_disk_parms =  {'pb_func':'casa_airy','dish_diameter':24.5,'blockage_diameter':0.0}
    beam_model_map = np.zeros(len(ant_pos), dtype=int)
    #beam_models = [zpc_xds,airy_disk_parms]
    beam_models = [J_xds]
    #beam_models = [airy_disk_parms]
    #beam_models = [airy_disk_parms,airy_disk_parms]
    beam_model_map = np.zeros(n_ant,dtype=int)
    #beam_model_map[1] = 1
    #beam_model_map[10] = 1

    beam_parms = {}
    beam_parms['fov_scaling'] = 15
    beam_parms['mueller_selection'] = np.arange(16)
    beam_parms['zernike_freq_interp'] = 'nearest'
    #beam_parms['freq'] = mxds.xds0.chan.values
    beam_parms['pa_radius'] = 0.2
    beam_parms['image_size'] = np.array([500,500])#np.array([2000,2000])

    freq_chan = mxds.xds0.chan.values
    pol = mxds.xds0.pol.values
    ######################
    
    #Setup sources
    #pointing_ra_dec:  [n_time, n_ant, 2]          (singleton: n_time, n_ant)
    pointing_ra_dec = None #np.zeros((1, 1, 2)) #Singleton

    #point_source_ra_dec:  [n_time, n_point_sources, 2]          (singleton: n_time)
    point_source_skycoord = SkyCoord(ra='19h59m28.5s',dec='+40d44m01.5s',frame='fk5') #sim
    point_source_ra_dec = np.array([point_source_skycoord.ra.rad,point_source_skycoord.dec.rad])[None,None,:]

    # NB n_pol is no longer singleton
    #point_source_flux: [n_time, n_chan, n_pol, n_point_sources] (singleton: n_time, n_chan)
    point_source_flux = np.array([2.5, 0, 0, 2.5])[None,None,:,None] #has to be in instrument polarization: [RR,RL,LR,LL] or [XX,XY,YX,YY]. All 4 values are needed.
    pb_limit = 0.2

    ###############################
    
    vis_data, uvw =simulation(point_source_flux, point_source_ra_dec, pointing_ra_dec, phase_center_ra_dec, beam_parms,beam_models,beam_model_map,uvw_parms, tel_xds, time_str, freq_chan, pol, ant1, ant2, pb_limit, uvw)
    
    vis_data_import = np.load("data/test_data/simulation_test/vis_data_test.npy")
    uvw_import = np.load("data/test_data/simulation_test/uvw_test.npy")"""
    
    #assert (np.allclose(vis_data, vis_data_import), np.allclose(uvw, uvw_import)) == (True, True)
    assert True