from sirius import calc_vis_point, calc_uvw

def simulation(make_uvw, uvw, uvw_tuple, calc_vis_tuple):
    if make_uvw == "astropy":
        ant_pos, mjd, site, pointing_direction = uvw_tuple
        uvw = calc_uvw.calc_uvw_astropy(ant_pos, mjd, site, pointing_direction)
        print(uvw)
    
    shape, point_source_flux, point_source_ra_dec, pointing_ra_dec, phase_center_ra_dec, antenna1, antenna2, n_ant, freq_chan, pb_parms = calc_vis_tuple
    
    result = calc_vis_point.calc_vis(uvw,shape,point_source_flux,point_source_ra_dec,pointing_ra_dec,phase_center_ra_dec,antenna1,antenna2,n_ant,freq_chan,pb_parms)
    
    return result
    
    