

def sim_sirius_evla(ms_name):
    import pkg_resources
    import xarray as xr
    import numpy as np
    import os
    from sirius._sirius_utils._coord_transforms import _sin_pixel_to_celestial_coord
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from sirius_data import _constants as cnt

    ########## Get telescope layout ##########
    tel_dir = pkg_resources.resource_filename('sirius_data', 'telescope_layout/data/vla.d.tel.zarr')
    tel_xds = xr.open_zarr(tel_dir,consolidated=False)
    n_ant = tel_xds.dims['ant_name']
    tel_xds.attrs['telescope_name'] = 'EVLA'

    # If using uvw_parms['calc_method'] = 'casa' .casarc must have directory of casadata.
    import pkg_resources
    casa_data_dir = pkg_resources.resource_filename('casadata', '__data__')
    rc_file = open(os.path.expanduser("~/.casarc"), "a+")  # append mode
    rc_file.write("\n measures.directory: " + casa_data_dir)
    rc_file.close()
    uvw_parms = {}
    uvw_parms['calc_method'] = 'casa' #'astropy' or 'casa'

    #https://github.com/casacore/casacore/blob/dbf28794ef446bbf4e6150653dbe404379a3c429/measures/Measures/Stokes.h
    # ['RR','RL','LR','LL'] => [5,6,7,8], ['XX','XY','YX','YY'] => [9,10,11,12]
    pol = [5,6,7,8]

    from sirius.dio import make_time_xda
    time_xda = make_time_xda(time_start='2019-10-03T21:21:40.151',time_delta=7200,n_samples=6,n_chunks=1)

    from sirius.dio import make_chan_xda
    spw_name = 'SBand'
    chan_xda = make_chan_xda(freq_start = 2.052*10**9, freq_delta = 0.1*10**9, freq_resolution=0.02*10**9, n_channels=1, n_chunks=1)

    image_size = np.array([1024,1024])
    cell_size = np.array([[-4.0,4.0]])
    cell_size = cell_size*cnt.arcsec_to_rad
    phase_center = SkyCoord(ra='19h59m28.5s',dec='+40d44m01.5s',frame='fk5')
    phase_center_ra_dec = np.array([phase_center.ra.rad,phase_center.dec.rad])[None,:]
    phase_center_names = np.array(['field1'])

    pixel = np.array([[512,612]])
    point_source_ra_dec = _sin_pixel_to_celestial_coord(phase_center_ra_dec[0,:],image_size,cell_size,pixel)[None,:,:] #[n_time, n_point_sources, 2] (singleton: n_time)
    point_source_skycoord = SkyCoord(ra=point_source_ra_dec[0,0,0]*u.rad,dec=point_source_ra_dec[0,0,1]*u.rad,frame='fk5')

    point_source_flux = np.array([1.314, 0, 0, 1.314])[None,None,None,:]

    pointing_ra_dec = None #No pointing offsets

    noise_parms = None
    #noise_parms ={}
    
    #If Zernike Polynomial should be used:
    zpc_dir = pkg_resources.resource_filename('sirius_data', 'aperture_polynomial_coefficient_models/data/EVLA_avg_zcoeffs_SBand_lookup.apc.zarr')
    zpc_xds = xr.open_zarr(zpc_dir,consolidated=False)

    beam_model_map = np.zeros(n_ant,dtype=int)
    beam_models = [zpc_xds]
    beam_parms = {}
    
    from sirius import simulation
    save_parms = {'ms_name':ms_name,'write_to_ms':True,}
    ms_xds = simulation(point_source_flux,
                         point_source_ra_dec,
                         pointing_ra_dec,
                         phase_center_ra_dec,
                         phase_center_names,
                         beam_parms,beam_models,
                         beam_model_map,uvw_parms,
                         tel_xds, time_xda, chan_xda, pol, noise_parms, save_parms)
    return ms_xds

