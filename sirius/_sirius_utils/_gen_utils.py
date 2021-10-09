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
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# Import required tools/tasks
from casatools import simulator, image, table, coordsys, measures, componentlist, quanta, ctsys, ms
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

import pylab as pl
import cngi.dio as dio
from cngi.conversion import convert_ms

def _display_image(imname='sim.image', pbname='', resname='',source_peak=1.0,chan=0):
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

    else:
        ploc = np.where( impix == impix.max() )
        print("Image Peak : %3.4f"%(impix[ploc[0][0], ploc[1][0],0,0]))
        if pbname != '':
            print("PB Value : %3.4f"%(impb[ploc[0][0], ploc[1][0],0,0]))
        pk = impix[ploc[0][0], ploc[1][0],0,0]

    if resname !='':
        istat = imstat(resname)  ### Make this calc within the PB.
        rres = istat['rms'][0]
        print('Residual RMS : %3.7f'%(rres))
    else:
        rres = None
    
 
    return pk, rres   # Return peak intensity from channnel 0 and rms


def _x_plot(vis='sim_data_ALMA.ms',ptype='amp-time',forceconvert=False,tel_name='ALMA'):
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
            axes[ax].set_title('Visibility Spectrum for field : '+fld) #+ '\nRed (A-A), Blue (B-B), Purple (A-B)')
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
        
def _listobs_jupyter(vis='sim_data_ALMA.ms'):
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
        
        
        
        
        
        
        
        
        