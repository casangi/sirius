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
        pl.plot(impix[ploc[0][0], ploc[1][0],0,:]/source_peak,'bo-',label='Im', marker="*", markersize=18)
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


'''
def _display_image(imname='',chan=0):
    ia.open(imname)
    pix = ia.getchunk()[:,:,0,chan]
    csys = ia.coordsys()
    ia.close()
    shp = pix.shape

    rad_to_deg =  180/np.pi
    w = WCS(naxis=2)
    w.wcs.crpix = csys.referencepixel()['numeric'][0:2]
    w.wcs.cdelt = csys.increment()['numeric'][0:2]*rad_to_deg
    w.wcs.crval = csys.referencevalue()['numeric'][0:2]*rad_to_deg
    w.wcs.ctype = ['RA---SIN', 'DEC--SIN']

    #plt.subplot(projection=w)

    p1 = int(shp[0]*0.25)
    p2 = int(shp[0]*0.75)
    
    #print(p1,p2)
    
    plt.figure()
    plt.imshow(pix.transpose(), origin='lower',  cmap=plt.cm.viridis)
    plt.xlabel('Right Ascension')
    plt.ylabel('Declination')
    plt.show()
'''
