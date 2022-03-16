import time
import os
import xarray as xr
from astropy.timeseries import TimeSeries
from astropy.time import Time
from casatools import simulator
from casatasks import mstransform, fixvis
from astropy import units as u
from collections import Counter
import pkg_resources
from sirius._sirius_utils._coord_transforms import _sin_pixel_to_celestial_coord
from astropy.coordinates import SkyCoord
from astropy import units as u
from sirius_data import _constants as cnt
    
from casatools import simulator, image, table, coordsys, measures, componentlist, quanta, ctsys
from casatasks import tclean, ft, imhead, listobs, exportfits, flagdata, bandpass, applycal
from casatasks.private import simutil

import os
import pylab as pl
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# Instantiate all the required tools
sm = simulator()
ia = image()
tb = table()
cs = coordsys()
me = measures()
qa = quanta()
cl = componentlist()
mysu = simutil.simutil()


def sim_awproject_evla(ms_name):
    
    os.system('rm -rf ' + ms_name.split('.')[0]+'*')
    
    integration_time = 7200
    image_size = np.array([1024,1024])
    cell_size = np.array([-4.0,4.0])
    cell_size = cell_size*cnt.arcsec_to_rad

    phase_center_skycoord = SkyCoord(ra='19h59m28.5s',dec='40d44m01.5s',frame='fk5')
    phase_center_ra_dec = np.array([phase_center_skycoord.ra.rad,phase_center_skycoord.dec.rad])
    phase_center_dict = {'m0': {'unit': 'rad', 'value': phase_center_ra_dec[0]}, 'm1': {'unit': 'rad', 'value': phase_center_ra_dec[1]}, 'refer': 'J2000', 'type': 'direction'}

    pixel = np.array([[512,612]])
    point_source_ra_dec = _sin_pixel_to_celestial_coord(phase_center_ra_dec,image_size,cell_size,pixel)[0]
    point_source_pos_dict = {'m0': {'unit': 'rad', 'value': point_source_ra_dec[0]}, 'm1': {'unit': 'rad', 'value': point_source_ra_dec[1]}, 'refer': 'J2000', 'type': 'direction'}

    #### Make ms frame (creates main table with UVW coordinates and DATA initialised to 0).
    make_ms_frame(ms_name,phase_center_dict,integration_time)
    print('Created ms frame')

    #### Create component list
    cl_name = ms_name.split('.')[0] + '.cl'
    os.system('rm -rf ' + cl_name)
    create_component_list(cl_name,point_source_pos_dict)
    
    image_name_model =  ms_name.split('.')[0] + '_model'
    make_empty_image(image_name_model,image_size,cell_size,phase_center_dict)
    eval_comp_list(cl_name,image_name_model)
    
    if ms_name.rfind('/'):
        split_data_dir = ms_name[:ms_name.rfind('/')+1]
        split_ms_name = ms_name[ms_name.rfind('/')+1:]
    else:
        split_data_dir = './'

    image_name = split_data_dir + 'img_' + split_ms_name.split('.')[0]
    os.system('rm -rf ' + image_name+'*')
    predict_imager(cell_size,image_size,ms_name,image_name,image_name_model)
    copy_model_to_data(ms_name)

    
## Use an input model sky image - widefield gridders
def predict_imager(cell_size,image_size,ms_name,image_name,
                  image_name_model):
    
    cfcache = image_name.split('.')[0] + '.cf'
    cell = np.array([str(-cell_size[0]) + 'rad',str(cell_size[1]) + 'rad'])
    
    # Run tclean in predictModel mode.
    tclean(vis=ms_name,
       startmodel=image_name_model,
       imagename=image_name,
       savemodel='modelcolumn',
       imsize=image_size,
       cell=cell,
       specmode='mfs',
       stokes='I',
       interpolation='nearest',
       #start='1.0GHz',
       #width='0.1GHz',
       #nchan=10,
       #reffreq='1.5Hz',
       gridder='awproject',
       normtype='flatsky',  # sky model is flat-sky
       cfcache=cfcache,
       wbawp=True,      # ensure that gridders='mosaic' and 'awproject' do freq-dep PBs
       pblimit=0.05,
       conjbeams=False,
       calcres=False,
       calcpsf=True,
       niter=0,
       wprojplanes=1,
       computepastep=0.1)
   
    '''
        tclean(vis='test2.ms',
       imagename='x_image',
       imsize=1024,
       cell='4.0arcsec',
       specmode='mfs',
       stokes='IQUV',
       interpolation='nearest',
       gridder='awproject',
       normtype='flatnoise',
       cfcache='img_test2.cf',
       wbawp=True,
       pblimit=0.05,
       conjbeams=False,
       calcres=True,
       calcpsf=True,
       niter=0,
       wprojplanes=1,
       computepastep=0.1)
   '''
    
    
def eval_comp_list(cl_name,image_name_model):
    ##  Evaluate a component list
    cl.open(cl_name)
    ia.open(image_name_model)
    ia.modify(cl.torecord(),subtract=False)
    ia.close()
    cl.done()
    
def make_empty_image(image_name_model,image_size,cell_size,phase_center_dict):
    #image_size nx,ny,pol,chan
    image_shape =  np.zeros((4,))
    image_shape[0:2] = image_size
    image_shape[2] = 1 #stokes I
    image_shape[3] = 1
    
    ## Make the image from a shape
    ia.close()
    ia.fromshape(image_name_model,image_shape,overwrite=True) #[256,256,1,10]
    
    ## Make a coordinate system
    cs=ia.coordsys()
    cs.setunits(['rad','rad','','Hz'])
    #cell_rad=qa.convert(qa.quantity('8.0arcsec'),"rad")['value']
    #cs.setincrement([-cell_rad,cell_rad],'direction')
    cell = np.array([str(cell_size[0]) + 'arcsec',str(cell_size[1]) + 'arcsec'])
    cs.setincrement(cell_size,'direction')
    
    cs.setreferencevalue([phase_center_dict['m0']['value'],phase_center_dict['m1']['value']],type="direction")
    #cs.setreferencevalue(dir)
    
    #cs.setreferencevalue('1.0GHz','spectral')
    #cs.setreferencepixel([0],'spectral')
    #cs.setincrement('0.1GHz','spectral')
    
    cs.setreferencevalue('2.052GHz','spectral')
    cs.setreferencepixel([0],'spectral')
    cs.setincrement('0.1GHz','spectral')
    
    ## Set the coordinate system in the image
    ia.setcoordsys(cs.torecord())
    ia.setbrightnessunit("Jy/pixel")
    ia.set(0.0)
    ia.close()
    
def make_ms_frame(ms_name,phase_center_dict,integration_time):
    tel_dir = pkg_resources.resource_filename('sirius_data', 'telescope_layout/data/evla.d.tel.zarr')
    tel_xds = xr.open_zarr(tel_dir,consolidated=False)
    n_ant = tel_xds.dims['ant_name']

    sm = simulator()
    
    ant_pos = tel_xds.ANT_POS.values
    sm.open(ms=ms_name);

    ## Set the antenna configuration
    sm.setconfig(telescopename= tel_xds.telescope_name,
                    x=ant_pos[:,0],
                    y=ant_pos[:,1],
                    z=ant_pos[:,2],
                    dishdiameter=tel_xds.DISH_DIAMETER.values,
                    mount=['alt-az'],
                    antname=list(tel_xds.ant_name.values),  #CASA can't handle an array of antenna names.
                    coordsystem='global',
                    referencelocation=tel_xds.site_pos[0]);
                    
    ## Set the polarization mode (this goes to the FEED subtable)
    sm.setfeed(mode='perfect R L', pol=['']);
    
    sm.setspwindow(spwname='SBand',
                freq='2.052GHz',
                deltafreq='0.1GHz',
                freqresolution='0.2GHz',
                nchannels=1,
                refcode='LSRK',
                stokes='RR LL');

    sm.setauto(autocorrwt=0.0)

    sm.settimes(integrationtime=integration_time,
                usehourangle=True,
                referencetime=me.epoch('UTC','2019/10/4/00:00:00'));
        
    fields_set = []

    sm.setfield(sourcename='field1',sourcedirection=phase_center_dict)
    sm.observe(sourcename='field1',
                spwname='SBand',
                starttime= '-6h',
                stoptime= '6h')
    sm.close()
    
    flagdata(vis=ms_name,mode='unflag')
    fixvis(ms_name,reuse=False) #Needed so that uvw agree with SiRIUS

    
def create_component_list(clname,point_source_pos_dict):
    # Create compoennt list
    # Make sure the cl doesn't already exist. The tool will complain otherwise.
    cl.done()
    cl.addcomponent(dir=point_source_pos_dict,
                    flux=[1.314,0.0,0.0,0.0],
                    polarization='stokes',
                    fluxunit='Jy',
                    freq='2.052GHz',
                    shape='point',       ## Point source
                    spectrumtype="spectral index",
                    index=-1.0)
    
    cl.rename(filename=clname)
    cl.done()


### Copy visibilities from the MODEL column to the data columns
### This is required when predicting using tclean or ft as they will only write to the MODEL column
def copy_model_to_data(msname):
    tb.open(msname,nomodify=False);
    moddata = tb.getcol(columnname='MODEL_DATA');
    tb.putcol(columnname='DATA',value=moddata);
    #tb.putcol(columnname='CORRECTED_DATA',value=moddata);
    moddata.fill(0.0);
    tb.putcol(columnname='MODEL_DATA',value=moddata);
    tb.close();
