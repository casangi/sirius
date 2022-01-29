#  CASA Next Generation Infrastructure
#  Copyright (C) 2021 AUI, Inc. Washington DC, USA
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from numba import jit
import numba
import numpy as np

import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# Import required tools/tasks
from casatools import simulator, image, table, coordsys, measures, componentlist, quanta, ctsys, ms
from casatasks import tclean, imstat, visstat
from casatasks.private import simutil
from IPython.display import Markdown as md

# Instantiate all the required tools
sm = simulator()
ia = image()
tb = table()
cs = coordsys()
me = measures()
qa = quanta()
cl = componentlist()
mysu = simutil.simutil()
myms = ms()


from sirius._sirius_utils._beam_utils import _pol_code_to_index
from sirius_data._constants import arcsec_to_rad, arcmin_to_rad, deg_to_rad, map_mueler_to_pol

import pylab as pl
import cngi.dio as dio
from cngi.conversion import convert_ms

def display_image(imname='sim.image', pbname='', resname='',source_peak=1.0,chan=0,ylim=[0.4,1.1]):
    #plt.close('all')
    ia.open(imname)
    shp = ia.shape()
    csys = ia.coordsys()
    impix = ia.getchunk()
    ia.close()
    if pbname != '':
        ia.open(pbname)
        impb = ia.getchunk()
        ia.close()

    rad_to_deg =  180/np.pi
    w = WCS(naxis=2)
    w.wcs.crpix = csys.referencepixel()['numeric'][0:2]
    w.wcs.cdelt = csys.increment()['numeric'][0:2]*rad_to_deg
    w.wcs.crval = csys.referencevalue()['numeric'][0:2]*rad_to_deg
    w.wcs.ctype = ['RA---SIN','DEC--SIN']
    #w.wcs.ctype = ['RA','DEC']

    #pl.figure(figsize=(12,5))
    pl.figure(figsize=(12,5))
    pl.clf()
    #pl.subplot(121)
    pl.subplot(121,projection=w)

    p1 = shp[0]#int(shp[0]*0.25)
    p2 = shp[1]#int(shp[0]*0.75)

    if pbname != '':
        impix[impb < 0.1] = 0.0
    
    pl.imshow(impix[:,:,0,chan].transpose(), origin='lower')
    if pbname != '':
        pl.contour(impb[:,:,0,chan].transpose(),[0.2],colors=['magenta'], origin='lower')
    pl.title('Image from channel 0')
    pl.xlabel('Right Ascension')
    pl.ylabel('Declination')
    
    
    pk = 0.0
    if shp[3]>1:
        pl.subplot(122)
        ploc = np.where( impix == impix.max() )
        pl.plot(impix[ploc[0][0], ploc[1][0],0,:]/source_peak,'b*-',label='Im', markersize=18)
        if pbname != '':
            pl.plot(impb[ploc[0][0], ploc[1][0],0,:],'ro-',label='PB')
        pl.title('Spectrum at source peak')
        pl.xlabel('Channel')
        #pl.ylim((0.4,1.1))
        pl.legend()
        pk = impix[ploc[0][0], ploc[1][0],0,0]
        print('Peak Intensity (chan0) : %3.7f'%(pk))
        if pbname != '':
            pbk = impb[ploc[0][0], ploc[1][0],0,0]
            print('PB at location of Intensity peak (chan0) : %3.7f'%(pbk))
        pl.ylim(ylim)

    else:
        ploc = np.where( impix == impix.max() )
        print("Image Peak : %3.4f"%(impix[ploc[0][0], ploc[1][0],0,0]))
        if pbname != '':
            print("PB Value : %3.4f"%(impb[ploc[0][0], ploc[1][0],0,0]))
        pk = impix[ploc[0][0], ploc[1][0],0,0]
        
    print("max pixel location ", ploc)

    if resname !='':
        istat = imstat(resname)  ### Make this calc within the PB.
        rres = istat['rms'][0]
        print('Residual RMS : %3.7f'%(rres))
    else:
        rres = None
    
    pl.show()
 
    return pk, rres   # Return peak intensity from channnel 0 and rms


def x_plot(vis='sim_data_ALMA.ms',ptype='amp-time',forceconvert=False,tel_name='ALMA'):
    """
    Make a few types of plots
    Supported types : amp-time, amp-freq, uvcov, plotants
    forceconvert=True/False : Convert the input MS to a Zarr dataset and read it into an XArray for plotting. 
                                               If set to False, it will skip the conversion step (and all the output messages the conversion produces). 
    """
    zvis = vis+'.zarr'
    if not os.path.exists(zvis) or forceconvert==True:
        convert_ms(vis, vis+'.zarr')
        
    xds = dio.read_vis(zvis)
    xdat = xds.xds0
    #print(xds)
    #print(xdat)
    #print(xds.ANTENNA)
    gxdat = xds.ANTENNA
    
    ant_names = gxdat.NAME.values
    xdat['field'] = xdat.FIELD_ID[:,0]
    
    
    xdat['DMAG'] = ((xdat['DATA'].real ** 2 + xdat['DATA'].imag ** 2) ** 0.5).mean(axis=3)
    xdat['U'] = xdat['UVW'][:,:,0]
    xdat['V'] = xdat['UVW'][:,:,1]
    xdat['-U'] = -xdat['UVW'][:,:,0]
    xdat['-V'] = -xdat['UVW'][:,:,1]
    
    ant_dias = np.unique(gxdat.DISH_DIAMETER)

    xAA = xdat.where( (gxdat.DISH_DIAMETER[xdat.ANTENNA1] == ant_dias[0])  &  (gxdat.DISH_DIAMETER[xdat.ANTENNA2] == ant_dias[0])  )
    xBB = xdat.where(  (gxdat.DISH_DIAMETER[xdat.ANTENNA1] == ant_dias[1])  &  (gxdat.DISH_DIAMETER[xdat.ANTENNA2] == ant_dias[1])  )
    xAB = xdat.where( ( (gxdat.DISH_DIAMETER[xdat.ANTENNA1] == ant_dias[0])  &  (gxdat.DISH_DIAMETER[xdat.ANTENNA2] == ant_dias[1]) | (gxdat.DISH_DIAMETER[xdat.ANTENNA1] == ant_dias[1])  &  (gxdat.DISH_DIAMETER[xdat.ANTENNA2] == ant_dias[0]) ) )


    if ptype == 'amp-time':
        fig, axes = pl.subplots(ncols=1,figsize=(9,3))
        for fld in np.unique(xdat.field):
            xAA.where(xAA.field==fld).plot.scatter(x='time',y='DMAG',  marker='.', color='r',alpha=0.1,label='A-A')
            xAB.where(xAB.field==fld).plot.scatter(x='time',y='DMAG',  marker='.', color='m',alpha=0.1,label='A-B')
            xBB.where(xBB.field==fld).plot.scatter(x='time',y='DMAG',  marker='.', color='b',alpha=0.1,label='B-B')
            
        pl.title('Visibility ampllitude : Red (A-A), Blue (B-B), Purple (A-B)');

    if ptype == 'amp-freq':
        fig, axes = pl.subplots(ncols=2,figsize=(9,3))
        ax=0
        for fld in np.unique(xdat.field):
            xAA.where(xAA.field==fld).plot.scatter(x='chan',y='DMAG',  marker='.', color='r',alpha=0.1,ax=axes[ax])
            xAB.where(xAB.field==fld).plot.scatter(x='chan',y='DMAG',  marker='.', color='m',alpha=0.1,ax=axes[ax])
            xBB.where(xBB.field==fld).plot.scatter(x='chan',y='DMAG',  marker='.', color='b',alpha=0.1,ax=axes[ax])
            axes[ax].set_title('Visibility Spectrum for field : '+ str(fld)) #+ '\nRed (A-A), Blue (B-B), Purple (A-B)')
            ax = ax+1
            
    if ptype == 'uvcov':
        fig, axes = pl.subplots(ncols=2,figsize=(9,4))
        ant_dias = np.unique(gxdat.DISH_DIAMETER)
        ax=0
        for fld in np.unique(xdat.field):
            xAB.where(xAB.field==fld).plot.scatter(x='U',y='V',  marker='.', color='m',ax=axes[ax])
            xAB.where(xAB.field==fld).plot.scatter(x='-U',y='-V',  marker='.', color='m',ax=axes[ax])
            xBB.where(xBB.field==fld).plot.scatter(x='U',y='V',  marker='.', color='b', ax=axes[ax])
            xBB.where(xBB.field==fld).plot.scatter(x='-U',y='-V',  marker='.', color='b', ax=axes[ax])
            xAA.where(xAA.field==fld).plot.scatter(x='U',y='V',  marker='.', color='r', ax=axes[ax])
            xAA.where(xAA.field==fld).plot.scatter(x='-U',y='-V',  marker='.', color='r', ax=axes[ax])
            axes[ax].set_title('UV coverage for field : '+str(fld))
            ax=ax+1
 

    if ptype == 'plotants':
        if tel_name=='ALMA':
            typeA = 'A'
        else:
            typeA = 'm'
        fig, axes = pl.subplots(ncols=1,figsize=(6,5))
        gxdat['ANT_XPOS'] = gxdat['POSITION'][:,0] - gxdat['POSITION'][:,0].mean()
        gxdat['ANT_YPOS'] = gxdat['POSITION'][:,1] - gxdat['POSITION'][:,1].mean()
        gxdat.plot.scatter(x='ANT_XPOS', y='ANT_YPOS',color='k',marker="1",s=200,linewidth=3.0)
        
        for i, txt in enumerate(ant_names):
            col = ('b' if (txt.count(typeA)>0) else 'r')
            pl.annotate('   '+txt, (gxdat['ANT_XPOS'].values[i], gxdat['ANT_YPOS'].values[i]),fontsize=12,color=col)   
        pl.title('Antenna Positions')
        
    pl.tight_layout()
    pl.show()
        
def listobs_jupyter(vis='sim_data_ALMA.ms'):
    """
    Print out the contents of listobs.
    TODO : Convert the contents to a df (if possible) for a pretty display
    """
    from casatasks import listobs
    listobs(vis=vis, listfile='obslist.txt', verbose=False, overwrite=True)
    ## print(os.popen('obslist.txt').read()) # ?permission denied?
    fp = open('obslist.txt')
    for aline in fp.readlines():
        print(aline.replace('\n',''))
    fp.close()

    tb.open(vis+'/ANTENNA')
    print("Dish diameter : " + str(tb.getcol('DISH_DIAMETER')))
    print("Antenna name : " + str(tb.getcol('NAME')))
    tb.close()
        
        
        
def image_ants(vis='sim_data_ALMA.ms',imname='try_ALMA', field='0', antsel='',tel='ALMA',vptable=''):
    """
    Run imager.....
    TODO  set pblimit=-1 ... or set the image stats method to look inside the PB region only.
    """
    antsels =  get_baseline_types(tel=tel)

    if antsel not in antsels.keys():
        print('Pick an antenna selection from '+ str(antsels))
        return
   
    
    if tel=='ALMA':
        cell='0.3arcsec'
        imsize=1024
        phasecenter='J2000 +19h59m28.5s -40d44m21.5s'
    if (tel=='NGVLA'):
        cell='1.0arcsec'
        imsize=1024
        phasecenter='J2000 +19h59m28.5s +40d44m21.5s'

   
    if field=='0' or field=='1':
        ftype = 'single'
    else:
        ftype = 'mosaic'
    
    imname1 = imname+'_'+antsel+'_'+ftype
    
    os.system('rm -rf '+imname1+'.*')
    
    print('antsels[antsel]2 ',antsels[antsel])
    
    # Run tclean
    tclean(vis=vis,
       antenna=antsels[antsel],
       field=field,
       imagename=imname1,
       imsize=imsize,
       cell=cell,
       phasecenter=phasecenter,
       specmode='cube',
       interpolation='nearest',
       nchan=-1,
       gridder='mosaic',
       vptable=vptable,
       normtype='flatnoise',
       wbawp=True,
       #pblimit=0.05,
       pblimit=0.2,
       pbmask=0.2,
       conjbeams=False,
       niter=1000,
       nsigma=3.0,
       datacolumn='data',
       weighting='natural')
        
        
        
def get_baseline_types(msname='',tel='ALMA'):
    """
    Define MSSelection strings for all 3 types of baselines, and all.
    Check them if an MS name is supplied.
    """

    if tel=='ALMA':
        antsels = {'A':'A*&',
                   'B':'J*,N*&',
                   'cross':'A* && J*,N*',
                   'all':'*'}
    if (tel=='NGVLA' or tel=='KSA'):
        antsels = {'A':'m*&',
                   'B':'s*&',
                   'cross':'m* && s*',
                   'all':'*'}


    if msname != "":
        for atype in antsels.keys():
            asel = myms.msseltoindex(vis=msname, baseline=antsels[atype])
            print(antsels[atype])
            print(asel['antenna1'])
            print(asel['antenna2'])
            print(asel['baselines'].transpose())
   
    return antsels
        


def check_vals(vis='',field='',spw='',pb_A='', pb_B='', antsels={}, meas_peaks={}, meas_rms={},tel='ALMA'):
    """
    For the given (selected) MS and PB images, pick out the PB values at the source location for antenna types A and B
    Calculate expected Intensity for the A-A, B-B, A-B and All baseline imaging cases.
    Using sigmas (or weights) and input vis-rms values, calculate expected rms for all 4 imaging cases
    Compare expected intensity, PB and RMS values with measured values.
    
    ALMA : A = 12m.   B = 7m
    ngVLA : A = 18m.  B = 6m
   
    """

    if tel=='ALMA':
        D_A = 12.0
        D_B = 7.0
        label_A = '12m'
        label_B = '7m'
        freq = 90.0 # GHz
    if (tel=='NGVLA'):
        D_A = 18.0
        D_B = 6.0
        label_A = '18m'
        label_B = '6m'
        freq = 40.0 # GHz
    if (tel=='KSA'):
        D_A = 20.0
        D_B = 10.0
        label_A = '20m'
        label_B = '10m'
        freq = 40.0 # GHz

    antsels3 = get_baseline_types(tel=tel)
    antsels3.pop('all')
    vis_summ = _get_vis_stat(vis=vis,field=field,spw=spw, antsels=antsels3)
    
    calc_peak_A = vis_summ['A']['mean']
    calc_peak_B = vis_summ['B']['mean']
    calc_peak_cross = vis_summ['cross']['mean']
    
    
    calc_peak_all = ( vis_summ['A']['mean']* vis_summ['A']['meanwt'] *  vis_summ['A']['npts'] +                \
                              vis_summ['B']['mean']* vis_summ['B']['meanwt']* vis_summ['B']['npts'] +                        \
                              vis_summ['cross']['mean']* vis_summ['cross']['meanwt']* vis_summ['cross']['npts'] ) /               \
                                 (vis_summ['A']['npts']* vis_summ['A']['meanwt'] +                                                       \
                                  vis_summ['B']['npts']* vis_summ['B']['meanwt'] +                                                           \
                                  vis_summ['cross']['npts']* vis_summ['cross']['meanwt'])
    
    calc_rms_A = (vis_summ['A']['stddev']/np.sqrt(vis_summ['A']['npts']) )
    calc_rms_B = vis_summ['B']['stddev']/np.sqrt(vis_summ['B']['npts'])
    calc_rms_cross = vis_summ['cross']['stddev']/np.sqrt(vis_summ['cross']['npts'])
    
    sumwt = (vis_summ['A']['npts']* vis_summ['A']['meanwt'] +                                                       \
                    vis_summ['B']['npts']* vis_summ['B']['meanwt'] +                                                           \
                    vis_summ['cross']['npts']* vis_summ['cross']['meanwt'])

    calc_rms_all = np.sqrt(( (calc_rms_A * vis_summ['A']['meanwt'] *  vis_summ['A']['npts']/sumwt)**2 +              \
                                         (calc_rms_B * vis_summ['B']['meanwt']* vis_summ['B']['npts']/sumwt)**2 +                      \
                                         (calc_rms_cross * vis_summ['cross']['meanwt']* vis_summ['cross']['npts']/sumwt)**2 )  )
  


    normn_A = vis_summ['A']['stddev']/vis_summ['A']['stddev']
    normn_B = vis_summ['B']['stddev']/vis_summ['A']['stddev']
    normn_cross = vis_summ['cross']['stddev']/vis_summ['A']['stddev']

    import pandas as pd
    from IPython.display import display, HTML
    #pd.set_option("display.precision", 3)
    results = { 'Baseline\nTypes' : [label_A+'-'+label_A,label_B+'-'+label_B , label_A+'-'+label_B, 'All'],
                'Vis Mean \n(V=I=P)' : [vis_summ['A']['mean'],vis_summ['B']['mean'], vis_summ['cross']['mean'], "-NA-" ],
           'Vis Std\n(sim)' : [vis_summ['A']['stddev'],vis_summ['B']['stddev'], vis_summ['cross']['stddev'], "-NA-" ],
                'Vis Std\n(norm sim)' : [normn_A, normn_B, normn_cross, "-NA-" ],
                'Vis Std\n(calc)' : [ 1.0, (D_A**2)/(D_B**2), (D_A**2)/(D_A*D_B)  , "-NA-" ],
           'Number of \n Data Pts' : [vis_summ['A']['npts'],vis_summ['B']['npts'],vis_summ['cross']['npts'],"-NA"],
           'Weight\n(calc)' :  [1.0, ((D_B**2)/(D_A**2))**2, ((D_A*D_B)/(D_A**2))**2 , "-NA-" ],
                'Weight\n(sim)' :[vis_summ['A']['meanwt'],vis_summ['B']['meanwt'], vis_summ['cross']['meanwt'], "-NA-" ],
           'Int Jy/bm\n(calc)' : [calc_peak_A, calc_peak_B, np.sqrt(calc_peak_A * calc_peak_B)  , calc_peak_all],
           'Int Jy/bm\n(meas)' : [meas_peaks['A'],meas_peaks['B'],meas_peaks['cross'],meas_peaks['all']],
           'RMS mJy\n(calc)' : [calc_rms_A*1e+3, calc_rms_B*1e+3, calc_rms_cross*1e+3, calc_rms_all*1e+3],
           'RMS mJy\n(meas)' : [meas_rms['A']*1e+3,meas_rms['B']*1e+3,meas_rms['cross']*1e+3,meas_rms['all']*1e+3]   }
    df = pd.DataFrame(results)
    with pd.option_context('display.float_format', '{:0.4f}'.format):
        display(HTML(df.to_html()))
        #print(pd.DataFrame(df).to_markdown())
    #print(df)
    

    print("Calculated PB size for type A (dia=%2.2f) : %3.5f arcmin"%(D_A, _calc_ang(freq,D_A)))
    print("Calculated PB size for type B (dia=%2.2f) : %3.5f arcmin"%(D_B, _calc_ang(freq,D_B)))
    

    return
    


def _get_vis_stat(vis='sim_data_ALMA.ms',field='',spw='', antsels={}):
    """
    ### This cell examines the dataset and checks noise level per baseline type.
    ## Use visstat with antenna selection ?
    """

    vis_summ = {}
    
    for btype in antsels.keys():
        vrec = visstat(vis=vis,field=field,spw=spw,antenna=antsels[btype],axis='amp',datacolumn='data')['DATA_DESC_ID=0']
        wrec = visstat(vis=vis,field=field,spw=spw,antenna=antsels[btype],axis='weight_spectrum',datacolumn='data')['DATA_DESC_ID=0']
        #print(vrec)
        vis_summ[btype] = {'mean':vrec['mean'], 'stddev':vrec['stddev'], 'sumwt':vrec['sumOfWeights'], 'npts':vrec['npts'], 'meanwt':wrec['mean']}
    
 #   df = pd.DataFrame(vis_summ)
 #   print("Summary of stats from the input MS")
 #   with pd.option_context('display.float_format', '{:0.8f}'.format):
 #       print(df)
 #   print("\n")
    return vis_summ

def _calc_ang(freq, dia):
    return  ( (3e+8/(freq*1e+9)) / dia ) * (180.0/np.pi) * 60.0



def display_J(J_xds, pa, chan, val_type='abs', units='rad'):
    '''
    val_type: ['abs','phase','real','imag']
    '''
    pol_indx = _pol_code_to_index(J_xds.pol.values)
    J_img_ = J_xds.J.isel(pa=pa,chan=chan,drop=True)
    
    if val_type == 'abs':
        J_img = np.abs(J_img_)
    elif val_type == 'phase':
        J_img = np.angle(J_img_)
    elif val_type == 'real':
        J_img = np.real(J_img_)
    elif val_type == 'imag':
        J_img = np.imag(J_img_)
    #print(J_img)
    
    if units == 'arcsec':
        l = J_xds.l.values/arcsec_to_rad
        m = J_xds.m.values/arcsec_to_rad
        pa_dis = J_xds.pa[pa].values/arcsec_to_rad
        
    if units == 'arcmin':
        l = J_xds.l.values/arcmin_to_rad
        m = J_xds.m.values/arcmin_to_rad
        pa_dis = J_xds.pa[pa].values/arcmin_to_rad
    
    if units == 'deg':
        l = J_xds.l.values/deg_to_rad
        m = J_xds.m.values/deg_to_rad
        pa_dis = J_xds.pa[pa].values/deg_to_rad
    
    if units == 'rad':
        l = J_xds.l.values
        m = J_xds.m.values
        pa_dis = J_xds.pa[pa].values
        
    group_1 = np.array([0,3]) #diagonal
    group_2 = np.array([1,2]) #the rest

    min_max_g1 = np.array([np.nan, np.nan])
    min_max_g2 = np.array([np.nan, np.nan])

    #print(J_img)
    #vmin=0, vmax=10
    for i in pol_indx:
        #print(np.where(pol_indx==i)[0][0])
        #img = J_img.isel(pol=np.where(pol_indx==i)[0][0]) fails with angle (angle somehow drops xarray labeling).
        img = J_img[np.where(pol_indx==i)[0][0],:,:]
        if i in group_1:
            min_val = np.min(img)
            if (min_max_g1[0] > min_val) or (np.isnan(min_max_g1[0])):
                min_max_g1[0] = min_val
            max_val = np.max(img)
            if (min_max_g1[1] < max_val) or (np.isnan(min_max_g1[1])):
                min_max_g1[1] = max_val
        if i in group_2:
            min_val = np.min(img)
            if (min_max_g2[0] > min_val) or (np.isnan(min_max_g2[0])):
                min_max_g2[0] = min_val
            max_val = np.max(img)
            if (min_max_g2[1] < max_val) or (np.isnan(min_max_g2[1])):
                min_max_g2[1] = max_val

    #print(min_max_g1,min_max_g2)
    print('Parallactic Angle',pa_dis, 'rad')
    print('Frequency', J_xds.chan[chan].values/10**9, 'GHz')
    
    plt.close('all')

    extent=(np.min(m),np.max(m),np.min(l),np.max(l))

    fig = plt.figure(figsize=(7., 5.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 2),  # creates 2x2 grid of axes
                 axes_pad=0.0,  # pad between axes in inch.
                 share_all=True)
                 
    img_list = [0,0,0,0]
    for i,ax in enumerate(grid):
        # Iterating over the grid returns the Axes.
        if i in pol_indx:
            #img = J_xds.J[pa,chan,np.where(pol_indx==i)[0][0],:,:]
            #img = J_img.isel(pol=np.where(pol_indx==i)[0][0])
            img = J_img[np.where(pol_indx==i)[0][0],:,:]
            img_list[i] = ax.imshow(img,extent=extent)
            
            if i in group_1:
                min_max = min_max_g1
                img_list[i] = ax.imshow(np.abs(img),extent=extent,vmin=min_max[0], vmax=min_max[1],cmap='viridis')
            if i in group_2:
                min_max = min_max_g2
                img_list[i] = ax.imshow(np.abs(img),extent=extent,vmin=min_max[0], vmax=min_max[1],cmap='inferno')
    '''
    if (0 in pol_indx):
        fig.subplots_adjust(right=0.7)
        cbar_ax = fig.add_axes([0.75, 0.15, 0.02, 0.7])
        fig.colorbar(img_list[0], cax=cbar_ax)
    elif (1 in pol_indx):
        fig.subplots_adjust(right=0.7)
        cbar_ax = fig.add_axes([0.75, 0.15, 0.02, 0.7])
        fig.colorbar(img_list[1], cax=cbar_ax)

    if (1 in pol_indx):
        fig.subplots_adjust(right=0.7)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.colorbar(img_list[1], cax=cbar_ax)
    elif (2 in pol_indx):
        fig.subplots_adjust(right=0.7)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.colorbar(img_list[2], cax=cbar_ax)
    '''
        

    for i,ax in enumerate(grid):
        if i in pol_indx:
            if i in group_1:
                fig.subplots_adjust(right=0.7)
                cbar_ax = fig.add_axes([0.75, 0.15, 0.02, 0.7])
                fig.colorbar(img_list[0], cax=cbar_ax)
            if i in group_2:
                fig.subplots_adjust(right=0.7)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
                fig.colorbar(img_list[2], cax=cbar_ax)
     
        
    ####################################################################
    # Code from https://stackoverflow.com/questions/15466893/how-can-i-create-single-axes-labels-for-an-entire-figure-created-with-imagegrid
    #get the extent of the largest box containing all the axes/subplots
    extents = np.array([a.get_position().extents for a in grid])  #all axes extents
    bigextents = np.empty(4)
    bigextents[:2] = extents[:,:2].min(axis=0)
    bigextents[2:] = extents[:,2:].max(axis=0)

    #text to mimic the x and y label. The text is positioned in the middle
    labelpad=0.1  #distance between the external axis and the text
    xlab_t = fig.text((bigextents[2]+bigextents[0])/2-0.1, bigextents[1]-labelpad, 'm (' + units + ')', horizontalalignment='center', verticalalignment = 'bottom',fontsize='large')
    ylab_t = fig.text( bigextents[0]-labelpad, (bigextents[3]+bigextents[1])/2, 'l (' + units + ')', rotation='vertical', horizontalalignment = 'left', verticalalignment = 'center',fontsize='large')
    
    xlab_t = fig.text((bigextents[2]+bigextents[0])/2, bigextents[2]+0.05, 'Antenna Sky Jones Matrix', horizontalalignment='center', verticalalignment = 'top',fontsize='x-large')
    
    plt.show()



def display_M(M_xds, pa, chan, val_type='abs', units='rad'):
    '''
    val_type: ['abs','phase','real','imag']
    '''
    if units == 'arcsec':
        l = M_xds.l.values/arcsec_to_rad
        m = M_xds.m.values/arcsec_to_rad
        pa_dis = M_xds.pa[pa].values/arcsec_to_rad
    
    if units == 'arcmin':
        l = M_xds.l.values/arcmin_to_rad
        m = M_xds.m.values/arcmin_to_rad
        pa_dis = M_xds.pa[pa].values/arcmin_to_rad
    
    if units == 'deg':
        l = M_xds.l.values/deg_to_rad
        m = M_xds.m.values/deg_to_rad
        pa_dis = M_xds.pa[pa].values/deg_to_rad
    
    if units == 'rad':
        l = M_xds.l.values
        m = M_xds.m.values
        pa_dis = M_xds.pa[pa].values
    
    print('Parallactic Angle',pa_dis, 'rad')
    print('Frequency', M_xds.chan[chan].values/10**9, 'GHz')
    
    m_sel = M_xds.m_sel.values
    
    plt.close('all')
    
    #M_img = M_xds.M.sel(pa=pa,chan=chan,drop=True)
    M_img = M_xds.M.isel(pa=pa,chan=chan,drop=True)
    if val_type == 'abs':
        M_img = np.abs(M_img)
    elif val_type == 'phase':
        M_img = np.angle(M_img)
    elif val_type == 'real':
        M_img = np.real(M_img)
    elif val_type == 'imag':
        M_img = np.imag(M_img)


    extent=(np.min(m),np.max(m),np.min(l),np.max(l))
    
    group_1 = np.array([0,5,10,15]) #diagonal
    group_2 = np.array([1,2,4,7,8,11,13,14]) #the rest
    group_3 = np.array([3,6,9,12]) #anti-diagonal
    
    min_max_g1 = np.array([np.nan, np.nan])
    min_max_g2 = np.array([np.nan, np.nan])
    min_max_g3 = np.array([np.nan, np.nan])
        
    
    #vmin=0, vmax=10
    for i in m_sel:
        img = M_img.sel(m_sel=i)
        if i in group_1:
            min_val = np.min(img)
            if (min_max_g1[0] > min_val) or (np.isnan(min_max_g1[0])):
                min_max_g1[0] = min_val
            max_val = np.max(img)
            if (min_max_g1[1] < max_val) or (np.isnan(min_max_g1[1])):
                min_max_g1[1] = max_val
        if i in group_2:
            min_val = np.min(img)
            if (min_max_g2[0] > min_val) or (np.isnan(min_max_g2[0])):
                min_max_g2[0] = min_val
            max_val = np.max(img)
            if (min_max_g2[1] < max_val) or (np.isnan(min_max_g2[1])):
                min_max_g2[1] = max_val
        if i in group_3:
            min_val = np.min(img)
            if (min_max_g3[0] > min_val) or (np.isnan(min_max_g3[0])):
                min_max_g3[0] = min_val
            max_val = np.max(img)
            if (min_max_g3[1] < max_val) or (np.isnan(min_max_g3[1])):
                min_max_g3[1] = max_val
            
    
    print(min_max_g1,min_max_g2,min_max_g3)

    #fig = plt.figure(figsize=(12., 6.5))
    fig = plt.figure(figsize=(10., 6.7))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 4),  # creates 2x2 grid of axes
                 axes_pad=0.0,  # pad between axes in inch.
                 share_all=True)
                 
    img_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i,ax in enumerate(grid):
        # Iterating over the grid returns the Axes.
        
        if i in m_sel:
            img = M_img.sel(m_sel=i)
            if i in group_1:
                min_max = min_max_g1
                img_list[i] = ax.imshow(np.abs(img),extent=extent,vmin=min_max[0], vmax=min_max[1],cmap='viridis')
            if i in group_2:
                min_max = min_max_g2
                img_list[i] = ax.imshow(np.abs(img),extent=extent,vmin=min_max[0], vmax=min_max[1],cmap='inferno')
            if i in group_3:
                min_max = min_max_g3
                img_list[i] = ax.imshow(np.abs(img),extent=extent,vmin=min_max[0], vmax=min_max[1],cmap='cividis')

        
    for i,ax in enumerate(grid):
        if i in m_sel:
            if i in group_1:
                fig.subplots_adjust(right=0.7)
                cbar_ax = fig.add_axes([0.70, 0.15, 0.02, 0.7])
                fig.colorbar(img_list[i], cax=cbar_ax)
            if i in group_2:
                fig.subplots_adjust(right=0.7)
                cbar_ax = fig.add_axes([0.80, 0.15, 0.02, 0.7])
                fig.colorbar(img_list[i], cax=cbar_ax)
            if i in group_3:
                fig.subplots_adjust(right=0.7)
                cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
                fig.colorbar(img_list[i], cax=cbar_ax)
    '''
    for i,ax in enumerate(grid):
        if i in m_sel:
            if i in group_1:
                fig.subplots_adjust(right=0.7)
                cbar_ax = fig.add_axes([0.65, 0.15, 0.02, 0.7])
                fig.colorbar(img_list[i], cax=cbar_ax)
            if i in group_2:
                fig.subplots_adjust(right=0.7)
                cbar_ax = fig.add_axes([0.75, 0.15, 0.02, 0.7])
                fig.colorbar(img_list[i], cax=cbar_ax)
            if i in group_3:
                fig.subplots_adjust(right=0.7)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
                fig.colorbar(img_list[i], cax=cbar_ax)
    '''
    # Code from https://stackoverflow.com/questions/15466893/how-can-i-create-single-axes-labels-for-an-entire-figure-created-with-imagegrid
    #get the extent of the largest box containing all the axes/subplots
    extents = np.array([a.get_position().extents for a in grid])  #all axes extents
    bigextents = np.empty(4)
    bigextents[:2] = extents[:,:2].min(axis=0)
    bigextents[2:] = extents[:,2:].max(axis=0)

    #text to mimic the x and y label. The text is positioned in the middle
    labelpad=0.1  #distance between the external axis and the text
    xlab_t = fig.text((bigextents[2]+bigextents[0])/2-0.1, bigextents[1]-labelpad/1.5, 'm (' + units + ')', horizontalalignment='center', verticalalignment = 'bottom',fontsize='large')
    ylab_t = fig.text( bigextents[0]-labelpad/2, (bigextents[3]+bigextents[1])/2, 'l (' + units + ')', rotation='vertical', horizontalalignment = 'left', verticalalignment = 'center',fontsize='large')
    
    xlab_t = fig.text((bigextents[2]+bigextents[0])/2, bigextents[2]+0.05, 'Antenna Sky Mueller Matrix', horizontalalignment='center', verticalalignment = 'top',fontsize='x-large')

    plt.show()

