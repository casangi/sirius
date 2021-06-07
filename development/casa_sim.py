
def ps_sim(ms_name,im_name,conf_file,time_interval,dovp,gridder):
    import os
    #CASA5
    '''
    from simutil import simutil
    casalog.filter('DEBUG2')
    mysu = simutil()
    '''
    #CASA6
    # Import required tools/tasks
    from casatools import simulator, image, table, coordsys, measures, componentlist, quanta, ctsys
    from casatasks import tclean, ft, imhead, listobs, exportfits, flagdata, bandpass, applycal
    from casatasks.private import simutil
    from cngi.conversion import convert_ms, convert_image
    
    import os
    import pylab as pl
    import numpy as np
    #from astropy.io import fits
    #from astropy.wcs import WCS

    from casatools import simulator
    from casatasks import tclean
    # Instantiate all the required tools
    sm = simulator()
    ia = image()
    tb = table()
    cs = coordsys()
    me = measures()
    qa = quanta()
    cl = componentlist()
    mysu = simutil.simutil()

    os.system('rm -rf ' + ms_name)
     
    ## Open the simulator
    sm.open(ms=ms_name);
    
    q = mysu.readantenna(conf_file)
    
    #CASA5
    #(x,y,z,d,an,an2,telname, obspos) = mysu.readantenna(conf_file)
    
    #CASA6
    (x,y,z,d,an,an2,telname, obspos) = mysu.readantenna(conf_file)
     
    ## Set the antenna configuration
    sm.setconfig(telescopename=telname,
                        x=x,
                        y=y,
                        z=z,
                        dishdiameter=d,
                        mount=['alt-az'],
                        antname=an,
                        coordsystem='local',
                        referencelocation=me.observatory(telname));

    ## Set the polarization mode (this goes to the FEED subtable)
    sm.setfeed(mode='perfect R L', pol=['']);

    '''
    sm.setspwindow(spwname="SBand",
                   freq='3.0GHz',
                   deltafreq='0.4GHz',
                   freqresolution='0.01GHz',
                   nchannels=3,
                   refcode='LSRK',
                   stokes='RR RL LR LL');
    '''
                   
    sm.setspwindow(spwname="SBand",
                   freq='3.0GHz',
                   deltafreq='0.4GHz',
                   freqresolution='0.01GHz',
                   nchannels=3,
                   refcode='LSRK',
                   stokes='RR LL');

    sm.setfield( sourcename="fake",sourcedirection=me.direction(rf='J2000', v0='19h59m28.5s',v1='+40d44m01.5s'))
    #sm.setfield( sourcename="fake",sourcedirection=me.direction(rf='J2000', v0='0h0m0.0s',v1='+90d00m0.0s')) #Zenith

    ## Leave autocorrelations out of the MS.
    sm.setauto(autocorrwt=0.0)
        
    sm.settimes(integrationtime=integration_time,
                 usehourangle=True,
                 referencetime=me.epoch('UTC','2019/10/4/00:00:00'));


    sm.observe(sourcename="fake",
               spwname='SBand',
               starttime=time_interval[0],
               stoptime=time_interval[1]);

    ## Close the simulator
    sm.close()
    
    clname=ms_name.split('.')[0] + '.cl'
    os.system('rm -rf '+ clname)

    # Add sources, one at a time.
    # Call multiple times to add multiple sources. ( Change the 'dir', obviously )
#    cl.addcomponent(dir='J2000 19h59m28.5s +40d44m01.5s',
#                        flux=5.42,            # For a gaussian, this is the integrated area.
#                        fluxunit='Jy',
#                        freq='LSRK 3.0GHz',
#                        shape='point',       ## Point source
#    #                    shape='gaussian',   ## Gaussian
#    #                    majoraxis="5.0arcmin",
#    #                    minoraxis='2.0arcmin',
#                        spectrumtype="constant",
#                        index=-1.0)
    #Sim
    cl.addcomponent(dir='J2000 19h59m0.0s +40d51m01.5s',
                        flux=2.17,            # For a gaussian, this is the integrated area.
                        fluxunit='Jy',
                        freq='LSRK 3.0GHz',
                        shape='point',       ## Point source
    #                    shape='gaussian',   ## Gaussian
    #                    majoraxis="5.0arcmin",
    #                    minoraxis='2.0arcmin',
                        spectrumtype="constant",
                        index=-1.0)
#    #Zenith
#    cl.addcomponent(dir='J2000 19h59m0.0s +89d54m01.5s',
#                        flux=2.17,            # For a gaussian, this is the integrated area.
#                        fluxunit='Jy',
#                        freq='LSRK 3.0GHz',
#                        shape='point',       ## Point source
#    #                    shape='gaussian',   ## Gaussian
#    #                    majoraxis="5.0arcmin",
#    #                    minoraxis='2.0arcmin',
#                        spectrumtype="constant",
#                        index=-1.0)
    

#    cl.addcomponent(dir='J2000 19h59m0.0s +40d51m01.5s',
#                        flux=3.145,            # For a gaussian, this is the integrated area.
#                        fluxunit='Jy',
#                        freq='LSRK 3.0GHz',
#                        shape='point',       ## Point source
#    #                    shape='gaussian',   ## Gaussian
#    #                    majoraxis="5.0arcmin",
#    #                    minoraxis='2.0arcmin',
#                        spectrumtype="constant",
#                        index=-1.0)
                        
#    cl.addcomponent(dir='J2000 19h59m55.5s +40d27m01.5s',
#                        flux=7.56,            # For a gaussian, this is the integrated area.
#                        fluxunit='Jy',
#                        freq='LSRK 3.0GHz',
#                        shape='point',       ## Point source
#    #                    shape='gaussian',   ## Gaussian
#    #                    majoraxis="5.0arcmin",
#    #                    minoraxis='2.0arcmin',
#                        spectrumtype="constant",
#                        index=-1.0)
                                  
    cl.rename(filename=clname)
    cl.done()
    
    #sm.setlimits(shadowlimit=100, elevationlimit='-180.0deg')
   

    sm.openfromms(ms_name)
    
    if dovp == False:
        sm.setvp(dovp=dovp)
    else:
        sm.setvp(dovp=dovp,usedefaultvp=True,dosquint=False)
      
    sm.predict(complist = clname ,incremental=False)

    sm.close()

    flagdata(vis=ms_name,mode='unflag')
    
    
    os.system('rm -rf '+ im_name + '.*')
    tclean(vis=ms_name,imagename=im_name,imsize=[200,400],cell=[20.0,20.0],specmode='cube',niter=0,pblimit=0.0,gridder=gridder,stokes='RR')
    
    convert_ms(ms_name)
    convert_image(im_name)

dovp = True
gridder = 'standard'
#gridder = 'mosaic'
#ms_name = 'zenith_point_source_sim_dovp_' + str(dovp) + '.ms'
#im_name = 'zenith_point_source_sim_img/point_source_sim_dovp_' + str(dovp) + '_gridder_' + gridder

ms_name = 'point_source_sim_vis/point_source_sim_dovp_' + str(dovp) + '.ms'
im_name = 'point_source_sim_img/point_source_sim_dovp_' + str(dovp) + '_gridder_' + gridder

conf_file = 'tel_config/vla_small.d.cfg'
#time_interval = ['-5.0h','5.0h']
#integration_time = '600.0s'
time_interval = ['-5.0h','5.0h']
integration_time = '3600.0s'
ps_sim(ms_name,im_name,conf_file,time_interval,dovp,gridder)
        





