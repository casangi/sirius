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



### Gaussian Beam Source Model Functions ###
# To understand this math see memo https://drive.google.com/file/d/1ngPiUsGDazhfno8RxV4MXuPI6SnbLgO0/view

##import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift

#Performs convolution using fft
def _fft_conv(x,y,pix):
    return pix*pix*np.real(fftshift(ifft2((fft2(x))*(fft2(y)))))

#Convert Gaussian beam paramters from d1, d2, t (semi-major, semi-minor, theta) to a, b, g (alpha, beta, gamma)
def _convert_to_abg(parms_ddt):
    d1 = parms_ddt[0]
    d2 = parms_ddt[1]
    t = parms_ddt[2]
    m = (4*np.log(2))/(d1**2)
    n = (4*np.log(2))/(d2**2)
    a = m*(np.cos(t)**2) + n*(np.sin(t)**2)
    b = 2*(m-n)*np.sin(t)*np.cos(t)
    g = m*(np.sin(t)**2) + n*(np.cos(t)**2)
    return np.array([a, b, g])

#Convert Gaussian beam paramters from a, b, g (alpha, beta, gamma) to d1, d2, t (semi-major, semi-minor, theta)
def _convert_to_ddt(parms_abg):
    a = parms_abg[0]
    b = parms_abg[1]
    g = parms_abg[2]
    d1 = np.sqrt((8*np.log(2))/(a + g - np.sqrt(a**2 - 2*a*g + g**2 + b**2)))
    d2 = np.sqrt((8*np.log(2))/(a + g + np.sqrt(a**2 - 2*a*g + g**2 + b**2)))
    t = 0.5*np.arctan2(-b,g-a)
    return np.array([d1, d2, t])

#Calculates the abg parameters for a Fourier Transformed Gaussian
def _fft_parms_abg(parms_abg):
    a = parms_abg[0]
    b = parms_abg[1]
    g = parms_abg[2]
    a_ft = (4*a*g*np.pi**2)/(4*(a**2)*g-a*(b**2))
    print(a_ft)
    b_ft = (-4*a*b*np.pi**2)/(4*(a**2)*g-a*(b**2))
    g_ft = (4*(a**2)*np.pi**2)/(4*(a**2)*g-a*(b**2))
    return np.array([a_ft, b_ft, g_ft])

#Find the amplitude and ddt parameters for the correcting beam
def _gaussian_deconvolve(new_beam_parms_ddt,old_beam_parms_ddt,A_new_beam,A_old_beam):
    nb_abg = convert_to_abg(new_beam_parms_ddt)
    ob_abg = convert_to_abg(old_beam_parms_ddt)
    ft_nb_abg = fft_parms_abg(nb_abg)
    ft_ob_abg = fft_parms_abg(ob_abg)
    corrected_parms_abg = fft_parms_abg(ft_nb_abg-ft_ob_abg)
    corrected_parms_ddt = convert_to_ddt(corrected_parms_abg)
    
    A_nb_ft = ft_amp(new_beam_parms_ddt, A_new_beam)
    A_ob_ft = ft_amp(old_beam_parms_ddt, A_old_beam)
    A_cor = ift_amp(corrected_parms_ddt, A_nb_ft/A_ob_ft)
    
    return  A_cor, corrected_parms_ddt

#Fourier transformed amplitude of beam
def _ft_amp(beam_parms_ddt, A):
    return A*(np.pi*beam_parms_ddt[0]*beam_parms_ddt[1])/np.log(16)

#Inverse Fourier transformed amplitude of beam
def _ift_amp(beam_parms_ddt, A):
    return A*np.log(16)/(np.pi*beam_parms_ddt[0]*beam_parms_ddt[1])
    
#Generating beams from parameters
def _gauss_beam_ddt(amp,parms_ddt,x,y):
    '''
    parms_ddt:
    
    '''
    d1 = parms_ddt[0] #semi major axis
    d2 = parms_ddt[1] #semi minor axis
    t = parms_dtt[2] # positionangle t, is the anti-clockwise angle from the x axis to the line that lies along the greatest width
    x_r = np.cos(t)*x + np.sin(t)*y
    y_r = -np.sin(t)*x + np.cos(t)*y
    r = 4*np.log(2)*(x_r**2)/(d1**2) + 4*np.log(2)*(y_r**2)/(d2**2)
    
    B = amp*np.exp(-r)
    #B[r > 20] = 0.0
    return B

def _gauss_beam_abg(amp,parms_abg,x,y):
    a = parms_abg[0]
    b = parms_abg[1]
    g = parms_abg[2]
    r = a*(x**2) + b*x*y + g*y**2
    B = amp*np.exp(-r)
    #B[r > 20] = 0
    return B

