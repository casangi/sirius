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
import numpy as np
import numba
from ._math_utils import _powl

#@jit(nopython=False, cache=True, nogil=True)
def _generate_zernike_surface(c, x, y):

    '''
    Use polynomial approximations to generate Zernike surface
    '''

    #Setting coefficients array
    #Z = np.zeros(67,dtype=np.complex128)
    #Z = np.zeros(67,dtype=numba.c16)
    
    '''
    if len(coefficients) < len(Z):
        c = Z.copy()
        c[:len(coefficients)] += coefficients
    else:
        c = Z.copy()
        c[:len(coefficients)] += coefficients
    '''
    
    #Setting the equations for the Zernike polynomials
    #r = np.sqrt(_powl(x,2) + _powl(y,2))
    Z1  =  c[0]  * 1 # m = 0    n = 0
    Z2  =  c[1]  * x # m = -1   n = 1
    Z3  =  c[2]  * y # m = 1    n = 1
    Z4  =  c[3]  * 2*x*y # m = -2   n = 2
    Z5  =  c[4]  * (2*_powl(x,2) + 2*_powl(y,2) -1)# m = 0  n = 2
    Z6  =  c[5]  * (-1*_powl(x,2) + _powl(y,2))# m = 2  n = 2
    Z7  =  c[6]  * (-1*_powl(x,3) + 3*x*_powl(y,2)) #m = -3     n = 3
    Z8  =  c[7]  * (-2*x + 3*(_powl(x,3)) + 3*x*(_powl(y,2)))# m = -1   n = 3
    Z9  =  c[8]  * (-2*y + 3*_powl(y,3) + 3*(_powl(x,2)) * y)# m = 1    n = 3
    Z10 =  c[9]  * (_powl(y,3)-3*(_powl(x,2))*y)# m = 3 n =3
    Z11 =  c[10] * (-4*(_powl(x,3))*y + 4*x*(_powl(y,3)))#m = -4    n = 4
    Z12 =  c[11] * (-6*x*y + 8*(_powl(x,3))*y + 8*x*(_powl(y,3)))# m = -2   n = 4
    Z13 =  c[12] * (1-6*_powl(x,2) - 6*_powl(y,2) + 6*_powl(x,4) + 12*(_powl(x,2))*(_powl(y,2)) + 6*_powl(y,4))# m = 0  n = 4
    Z14 =  c[13] * (3*_powl(x,2) - 3*_powl(y,2) - 4*_powl(x,4) + 4*_powl(y,4))#m = 2    n = 4
    Z15 =  c[14] * (_powl(x,4) - 6*(_powl(x,2))*(_powl(y,2)) + _powl(y,4))# m = 4   n = 4
    Z16 =  c[15] * (_powl(x,5)-10*(_powl(x,3))*_powl(y,2) + 5*x*(_powl(y,4)))# m = -5   n = 5
    Z17 =  c[16] * (4*_powl(x,3) - 12*x*(_powl(y,2)) -5*_powl(x,5) + 10*(_powl(x,3))*(_powl(y,2)) + 15*x*_powl(y,4))# m =-3     n = 5
    Z18 =  c[17] * (3*x - 12*_powl(x,3) - 12*x*(_powl(y,2)) + 10*_powl(x,5) + 20*(_powl(x,3))*(_powl(y,2)) + 10*x*(_powl(y,4)))# m= -1  n = 5
    Z19 =  c[18] * (3*y - 12*_powl(y,3) - 12*y*(_powl(x,2)) + 10*_powl(y,5) + 20*(_powl(y,3))*(_powl(x,2)) + 10*y*(_powl(x,4)))# m = 1  n = 5
    Z20 =  c[19] * (-4*_powl(y,3) + 12*y*(_powl(x,2)) + 5*_powl(y,5) - 10*(_powl(y,3))*(_powl(x,2)) - 15*y*_powl(x,4))# m = 3   n = 5
    Z21 =  c[20] * (_powl(y,5)-10*(_powl(y,3))*_powl(x,2) + 5*y*(_powl(x,4)))#m = 5 n = 5
    Z22 =  c[21] * (6*(_powl(x,5))*y - 20*(_powl(x,3))*(_powl(y,3)) + 6*x*(_powl(y,5)))# m = -6 n = 6
    Z23 =  c[22] * (20*(_powl(x,3))*y - 20*x*(_powl(y,3)) - 24*(_powl(x,5))*y + 24*x*(_powl(y,5)))#m = -4   n = 6
    Z24 =  c[23] * (12*x*y + 40*(_powl(x,3))*y - 40*x*(_powl(y,3)) + 30*(_powl(x,5))*y + 60*(_powl(x,3))*(_powl(y,3)) - 30*x*(_powl(y,5)))#m = -2   n = 6
    Z25 =  c[24] * (-1 + 12*(_powl(x,2)) + 12*(_powl(y,2)) - 30*(_powl(x,4)) - 60*(_powl(x,2))*(_powl(y,2)) - 30*(_powl(y,4)) + 20*(_powl(x,6)) + 60*(_powl(x,4))*_powl(y,2) + 60 *(_powl(x,2))*(_powl(y,4)) + 20*(_powl(y,6)))#m = 0   n = 6
    Z26 =  c[25] * (-6*(_powl(x,2)) + 6*(_powl(y,2)) + 20*(_powl(x,4)) - 20*(_powl(y,4)) - 15*(_powl(x,6)) - 15*(_powl(x,4))*(_powl(y,2)) + 15*(_powl(x,2))*(_powl(y,4)) + 15*(_powl(y,6)))#m = 2   n = 6
    Z27 =  c[26] * (-5*(_powl(x,4)) + 30*(_powl(x,2))*(_powl(y,2)) - 5*(_powl(y,4)) + 6*(_powl(x,6)) - 30*(_powl(x,4))*_powl(y,2) - 30*(_powl(x,2))*(_powl(y,4)) + 6*(_powl(y,6)))#m = 4    n = 6
    Z28 =  c[27] * (-1*(_powl(x,6)) + 15*(_powl(x,4))*(_powl(y,2)) - 15*(_powl(x,2))*(_powl(y,4)) + _powl(y,6))#m = 6   n = 6
    Z29 =  c[28] * (-1*(_powl(x,7)) + 21*(_powl(x,5))*(_powl(y,2)) - 35*(_powl(x,3))*(_powl(y,4)) + 7*x*(_powl(y,6)))#m = -7    n = 7
    Z30 =  c[29] * (-6*(_powl(x,5)) + 60*(_powl(x,3))*(_powl(y,2)) - 30*x*(_powl(y,4)) + 7*_powl(x,7) - 63*(_powl(x,5))*(_powl(y,2)) - 35*(_powl(x,3))*(_powl(y,4)) + 35*x*(_powl(y,6))) #m = -5    n = 7
    Z31 =  c[30] * (-10*(_powl(x,3)) + 30*x*(_powl(y,2)) + 30*_powl(x,5) - 60*(_powl(x,3))*(_powl(y,2)) - 90*x*(_powl(y,4)) - 21*_powl(x,7) + 21*(_powl(x,5))*(_powl(y,2)) + 105*(_powl(x,3))*(_powl(y,4)) + 63*x*(_powl(y,6)))#m =-3       n = 7
    Z32 =  c[31] * (-4*x + 30*_powl(x,3) + 30*x*(_powl(y,2)) - 60*(_powl(x,5)) - 120*(_powl(x,3))*(_powl(y,2)) - 60*x*(_powl(y,4)) + 35*_powl(x,7) + 105*(_powl(x,5))*(_powl(y,2)) + 105*(_powl(x,3))*(_powl(y,4)) + 35*x*(_powl(y,6)))#m = -1  n = 7
    Z33 =  c[32] * (-4*y + 30*_powl(y,3) + 30*y*(_powl(x,2)) - 60*(_powl(y,5)) - 120*(_powl(y,3))*(_powl(x,2)) - 60*y*(_powl(x,4)) + 35*_powl(y,7) + 105*(_powl(y,5))*(_powl(x,2)) + 105*(_powl(y,3))*(_powl(x,4)) + 35*y*(_powl(x,6)))#m = 1   n = 7
    Z34 =  c[33] * (10*(_powl(y,3)) - 30*y*(_powl(x,2)) - 30*_powl(y,5) + 60*(_powl(y,3))*(_powl(x,2)) + 90*y*(_powl(x,4)) + 21*_powl(y,7) - 21*(_powl(y,5))*(_powl(x,2)) - 105*(_powl(y,3))*(_powl(x,4)) - 63*y*(_powl(x,6)))#m =3     n = 7
    Z35 =  c[34] * (-6*(_powl(y,5)) + 60*(_powl(y,3))*(_powl(x,2)) - 30*y*(_powl(x,4)) + 7*_powl(y,7) - 63*(_powl(y,5))*(_powl(x,2)) - 35*(_powl(y,3))*(_powl(x,4)) + 35*y*(_powl(x,6)))#m = 5  n = 7
    Z36 =  c[35] * (_powl(y,7) - 21*(_powl(y,5))*(_powl(x,2)) + 35*(_powl(y,3))*(_powl(x,4)) - 7*y*(_powl(x,6)))#m = 7  n = 7
    Z37 =  c[36] * (-8*(_powl(x,7))*y + 56*(_powl(x,5))*(_powl(y,3)) - 56*(_powl(x,3))*(_powl(y,5)) + 8*x*(_powl(y,7)))#m = -8  n = 8
    Z38 =  c[37] * (-42*(_powl(x,5))*y + 140*(_powl(x,3))*(_powl(y,3)) - 42*x*(_powl(y,5)) + 48*(_powl(x,7))*y - 112*(_powl(x,5))*(_powl(y,3)) - 112*(_powl(x,3))*(_powl(y,5)) + 48*x*(_powl(y,7)))#m = -6  n = 8
    Z39 =  c[38] * (-60*(_powl(x,3))*y + 60*x*(_powl(y,3)) + 168*(_powl(x,5))*y -168*x*(_powl(y,5)) - 112*(_powl(x,7))*y - 112*(_powl(x,5))*(_powl(y,3)) + 112*(_powl(x,3))*(_powl(y,5)) + 112*x*(_powl(y,7)))#m = -4   n = 8
    Z40 =  c[39] * (-20*x*y + 120*(_powl(x,3))*y + 120*x*(_powl(y,3)) - 210*(_powl(x,5))*y - 420*(_powl(x,3))*(_powl(y,3)) - 210*x*(_powl(y,5)) - 112*(_powl(x,7))*y + 336*(_powl(x,5))*(_powl(y,3)) + 336*(_powl(x,3))*(_powl(y,5)) + 112*x*(_powl(y,7)))#m = -2   n = 8
    Z41 =  c[40] * (1 - 20*_powl(x,2) - 20*_powl(y,2) + 90*_powl(x,4) + 180*(_powl(x,2))*(_powl(y,2)) + 90*_powl(y,4) - 140*_powl(x,6) - 420*(_powl(x,4))*(_powl(y,2)) - 420*(_powl(x,2))*(_powl(y,4)) - 140*(_powl(y,6)) + 70*_powl(x,8) + 280*(_powl(x,6))*(_powl(y,2)) + 420*(_powl(x,4))*(_powl(y,4)) + 280*(_powl(x,2))*(_powl(y,6)) + 70*_powl(y,8))#m = 0    n = 8
    Z42 =  c[41] * (10*_powl(x,2) - 10*_powl(y,2) - 60*_powl(x,4) + 105*(_powl(x,4))*(_powl(y,2)) - 105*(_powl(x,2))*(_powl(y,4)) + 60*_powl(y,4) + 105*_powl(x,6) - 105*_powl(y,6) - 56*_powl(x,8) - 112*(_powl(x,6))*(_powl(y,2)) + 112*(_powl(x,2))*(_powl(y,6)) + 56*_powl(y,8))#m = 2  n = 8
    Z43 =  c[42] * (15*_powl(x,4) - 90*(_powl(x,2))*(_powl(y,2)) + 15*_powl(y,4) - 42*_powl(x,6) + 210*(_powl(x,4))*(_powl(y,2)) + 210*(_powl(x,2))*(_powl(y,4)) - 42*_powl(y,6) + 28*_powl(x,8) - 112*(_powl(x,6))*(_powl(y,2)) - 280*(_powl(x,4))*(_powl(y,4)) - 112*(_powl(x,2))*(_powl(y,6)) + 28*_powl(y,8))#m = 4     n = 8
    Z44 =  c[43] * (7*_powl(x,6) - 105*(_powl(x,4))*(_powl(y,2)) + 105*(_powl(x,2))*(_powl(y,4)) - 7*_powl(y,6) - 8*_powl(x,8) + 112*(_powl(x,6))*(_powl(y,2)) - 112*(_powl(x,2))*(_powl(y,6)) + 8*_powl(y,8))#m = 6    n = 8
    Z45 =  c[44] * (_powl(x,8) - 28*(_powl(x,6))*(_powl(y,2)) + 70*(_powl(x,4))*(_powl(y,4)) - 28*(_powl(x,2))*(_powl(y,6)) + _powl(y,8))#m = 8     n = 9
    Z46 =  c[45] * (_powl(x,9) - 36*(_powl(x,7))*(_powl(y,2)) + 126*(_powl(x,5))*(_powl(y,4)) - 84*(_powl(x,3))*(_powl(y,6)) + 9*x*(_powl(y,8)))#m = -9     n = 9
    Z47 =  c[46] * (8*_powl(x,7) - 168*(_powl(x,5))*(_powl(y,2)) + 280*(_powl(x,3))*(_powl(y,4)) - 56 *x*(_powl(y,6)) - 9*_powl(x,9) + 180*(_powl(x,7))*(_powl(y,2)) - 126*(_powl(x,5))*(_powl(y,4)) - 252*(_powl(x,3))*(_powl(y,6)) + 63*x*(_powl(y,8)))#m = -7    n = 9
    Z48 =  c[47] * (21*_powl(x,5) - 210*(_powl(x,3))*(_powl(y,2)) + 105*x*(_powl(y,4)) - 56*_powl(x,7) + 504*(_powl(x,5))*(_powl(y,2)) + 280*(_powl(x,3))*(_powl(y,4)) - 280*x*(_powl(y,6)) + 36*_powl(x,9) - 288*(_powl(x,7))*(_powl(y,2)) - 504*(_powl(x,5))*(_powl(y,4)) + 180*x*(_powl(y,8)))#m = -5    n = 9
    Z49 =  c[48] * (20*_powl(x,3) - 60*x*(_powl(y,2)) - 105*_powl(x,5) + 210*(_powl(x,3))*(_powl(y,2)) + 315*x*(_powl(y,4)) + 168*_powl(x,7) - 168*(_powl(x,5))*(_powl(y,2)) - 840*(_powl(x,3))*(_powl(y,4)) - 504*x*(_powl(y,6)) - 84*_powl(x,9) + 504*(_powl(x,5))*(_powl(y,4)) + 672*(_powl(x,3))*(_powl(y,6)) + 252*x*(_powl(y,8)))#m = -3  n = 9
    Z50 =  c[49] * (5*x - 60*_powl(x,3) - 60*x*(_powl(y,2)) + 210*_powl(x,5) + 420*(_powl(x,3))*(_powl(y,2)) + 210*x*(_powl(y,4)) - 280*_powl(x,7) - 840*(_powl(x,5))*(_powl(y,2)) - 840*(_powl(x,3))*(_powl(y,4)) - 280*x*(_powl(y,6)) + 126*_powl(x,9) + 504*(_powl(x,7))*(_powl(y,2)) + 756*(_powl(x,5))*(_powl(y,4)) + 504*(_powl(x,3))*(_powl(y,6)) + 126*x*(_powl(y,8)))#m = -1   n = 9
    Z51 =  c[50] * (5*y - 60*_powl(y,3) - 60*y*(_powl(x,2)) + 210*_powl(y,5) + 420*(_powl(y,3))*(_powl(x,2)) + 210*y*(_powl(x,4)) - 280*_powl(y,7) - 840*(_powl(y,5))*(_powl(x,2)) - 840*(_powl(y,3))*(_powl(x,4)) - 280*y*(_powl(x,6)) + 126*_powl(y,9) + 504*(_powl(y,7))*(_powl(x,2)) + 756*(_powl(y,5))*(_powl(x,4)) + 504*(_powl(y,3))*(_powl(x,6)) + 126*y*(_powl(x,8)))#m = -1   n = 9
    Z52 =  c[51] * (-20*_powl(y,3) + 60*y*(_powl(x,2)) + 105*_powl(y,5) - 210*(_powl(y,3))*(_powl(x,2)) - 315*y*(_powl(x,4)) - 168*_powl(y,7) + 168*(_powl(y,5))*(_powl(x,2)) + 840*(_powl(y,3))*(_powl(x,4)) + 504*y*(_powl(x,6)) + 84*_powl(y,9) - 504*(_powl(y,5))*(_powl(x,4)) - 672*(_powl(y,3))*(_powl(x,6)) - 252*y*(_powl(x,8)))#m = 3  n = 9
    Z53 =  c[52] * (21*_powl(y,5) - 210*(_powl(y,3))*(_powl(x,2)) + 105*y*(_powl(x,4)) - 56*_powl(y,7) + 504*(_powl(y,5))*(_powl(x,2)) + 280*(_powl(y,3))*(_powl(x,4)) - 280*y*(_powl(x,6)) + 36*_powl(y,9) - 288*(_powl(y,7))*(_powl(x,2)) - 504*(_powl(y,5))*(_powl(x,4)) + 180*y*(_powl(x,8)))#m = 5     n = 9
    Z54 =  c[53] *(-8*_powl(y,7) + 168*(_powl(y,5))*(_powl(x,2)) - 280*(_powl(y,3))*(_powl(x,4)) + 56 *y*(_powl(x,6)) + 9*_powl(y,9) - 180*(_powl(y,7))*(_powl(x,2)) + 126*(_powl(y,5))*(_powl(x,4)) - 252*(_powl(y,3))*(_powl(x,6)) - 63*y*(_powl(x,8)))#m = 7     n = 9
    Z55 =  c[54] *(_powl(y,9) - 36*(_powl(y,7))*(_powl(x,2)) + 126*(_powl(y,5))*(_powl(x,4)) - 84*(_powl(y,3))*(_powl(x,6)) + 9*y*(_powl(x,8)))#m = 9       n = 9
    Z56 =  c[55] *(10*(_powl(x,9))*y - 120*(_powl(x,7))*(_powl(y,3)) + 252*(_powl(x,5))*(_powl(y,5)) - 120*(_powl(x,3))*(_powl(y,7)) + 10*x*(_powl(y,9)))#m = -10   n = 10
    Z57 =  c[56] *(72*(_powl(x,7))*y - 504*(_powl(x,5))*(_powl(y,3)) + 504*(_powl(x,3))*(_powl(y,5)) - 72*x*(_powl(y,7)) - 80*(_powl(x,9))*y + 480*(_powl(x,7))*(_powl(y,3)) - 480*(_powl(x,3))*(_powl(y,7)) + 80*x*(_powl(y,9)))#m = -8    n = 10
    Z58 =  c[57] *(270*(_powl(x,9))*y - 360*(_powl(x,7))*(_powl(y,3)) - 1260*(_powl(x,5))*(_powl(y,5)) - 360*(_powl(x,3))*(_powl(y,7)) + 270*x*(_powl(y,9)) - 432*(_powl(x,7))*y + 1008*(_powl(x,5))*(_powl(y,3)) + 1008*(_powl(x,3))*(_powl(y,5)) - 432*x*(_powl(y,7)) + 168*(_powl(x,5))*y - 560*(_powl(x,3))*(_powl(y,3)) + 168*x*(_powl(y,5)))#m = -6   n = 10
    Z59 =  c[58] *(140*(_powl(x,3))*y - 140*x*(_powl(y,3)) - 672*(_powl(x,5))*y + 672*x*(_powl(y,5)) + 1008*(_powl(x,7))*y + 1008*(_powl(x,5))*(_powl(y,3)) - 1008*(_powl(x,3))*(_powl(y,5)) - 1008*x*(_powl(y,7)) - 480*(_powl(x,9))*y - 960*(_powl(x,7))*(_powl(y,3)) + 960*(_powl(x,3))*(_powl(y,7)) + 480 *x*(_powl(y,9)))#m = -4   n = 10
    Z60 =  c[59] *(30*x*y - 280*(_powl(x,3))*y - 280*x*(_powl(y,3)) + 840*(_powl(x,5))*y + 1680*(_powl(x,3))*(_powl(y,3)) +840*x*(_powl(y,5)) - 1008*(_powl(x,7))*y - 3024*(_powl(x,5))*(_powl(y,3)) - 3024*(_powl(x,3))*(_powl(y,5)) - 1008*x*(_powl(y,7)) + 420*(_powl(x,9))*y + 1680*(_powl(x,7))*(_powl(y,3)) + 2520*(_powl(x,5))*(_powl(y,5)) + 1680*(_powl(x,3))*(_powl(y,7)) + 420*x*(_powl(y,9)) )#m = -2   n = 10
    Z61 =  c[60] * (-1 + 30*_powl(x,2) + 30*_powl(y,2) - 210*_powl(x,4) - 420*(_powl(x,2))*(_powl(y,2)) - 210*_powl(y,4) + 560*_powl(x,6) + 1680*(_powl(x,4))*(_powl(y,2)) + 1680*(_powl(x,2))*(_powl(y,4)) + 560*_powl(y,6) - 630*_powl(x,8) - 2520*(_powl(x,6))*(_powl(y,2)) - 3780*(_powl(x,4))*(_powl(y,4)) - 2520*(_powl(x,2))*(_powl(y,6)) - 630*_powl(y,8) + 252*_powl(x,10) + 1260*(_powl(x,8))*(_powl(y,2)) + 2520*(_powl(x,6))*(_powl(y,4)) + 2520*(_powl(x,4))*(_powl(y,6)) + 1260*(_powl(x,2))*(_powl(y,8)) + 252*_powl(y,10))#m = 0    n = 10
    Z62 =  c[61] * (-15*_powl(x,2) + 15*_powl(y,2) + 140*_powl(x,4) - 140*_powl(y,4) - 420*_powl(x,6) - 420*(_powl(x,4))*(_powl(y,2)) + 420*(_powl(x,2))*(_powl(y,4)) + 420*_powl(y,6) + 504*_powl(x,8) + 1008*(_powl(x,6))*(_powl(y,2)) - 1008*(_powl(x,2))*(_powl(y,6)) - 504*_powl(y,8) - 210*_powl(x,10) - 630*(_powl(x,8))*(_powl(y,2)) - 420*(_powl(x,6))*(_powl(y,4)) + 420*(_powl(x,4))*(_powl(y,6)) + 630*(_powl(x,2))*(_powl(y,8)) + 210*_powl(y,10))# m = 2  n = 10
    Z63 =  c[62] *(-35*_powl(x,4) + 210*(_powl(x,2))*(_powl(y,2)) - 35*_powl(y,4) + 168*_powl(x,6) - 840*(_powl(x,4))*(_powl(y,2)) - 840*(_powl(x,2))*(_powl(y,4)) + 168*_powl(y,6) - 252*_powl(x,8) + 1008*(_powl(x,6))*(_powl(y,2)) + 2520*(_powl(x,4))*(_powl(y,4)) + 1008*(_powl(x,2))*(_powl(y,6)) - 252*(_powl(y,8)) + 120*_powl(x,10) - 360*(_powl(x,8))*(_powl(y,2)) - 1680*(_powl(x,6))*(_powl(y,4)) - 1680*(_powl(x,4))*(_powl(y,6)) - 360*(_powl(x,2))*(_powl(y,8)) + 120*_powl(y,10))#m = 4     n = 10
    Z64 =  c[63] *(-28*_powl(x,6) + 420*(_powl(x,4))*(_powl(y,2)) - 420*(_powl(x,2))*(_powl(y,4)) + 28*_powl(y,6) + 72*_powl(x,8) - 1008*(_powl(x,6))*(_powl(y,2)) + 1008*(_powl(x,2))*(_powl(y,6)) - 72*_powl(y,8) - 45*_powl(x,10) + 585*(_powl(x,8))*(_powl(y,2)) + 630*(_powl(x,6))*(_powl(y,4)) - 630*(_powl(x,4))*(_powl(y,6)) - 585*(_powl(x,2))*(_powl(y,8)) + 45*_powl(y,10))#m = 6    n = 10
    Z65 =  c[64] *(-9*_powl(x,8) + 252*(_powl(x,6))*(_powl(y,2)) - 630*(_powl(x,4))*(_powl(y,4)) + 252*(_powl(x,2))*(_powl(y,6)) - 9*_powl(y,8) + 10*_powl(x,10) - 270*(_powl(x,8))*(_powl(y,2)) + 420*(_powl(x,6))*(_powl(y,4)) + 420*(_powl(x,4))*(_powl(y,6)) - 270*(_powl(x,2))*(_powl(y,8)) + 10*_powl(y,10))#m = 8    n = 10
    Z66 =  c[65] *(-1*_powl(x,10) + 45*(_powl(x,8))*(_powl(y,2)) - 210*(_powl(x,6))*(_powl(y,4)) + 210*(_powl(x,4))*(_powl(y,6)) - 45*(_powl(x,2))*(_powl(y,8)) + _powl(y,10))#m = 10   n = 10

    ZW =    Z1 + Z2 +  Z3+  Z4+  Z5+  Z6+  Z7+  Z8+  Z9+  Z10+ Z11+ Z12+ Z13+ Z14+ Z15+ Z16+ Z17+ Z18+ Z19+ Z20+ Z21+ Z22+ Z23+ Z24+ Z25+ Z26+ Z27+ Z28+ Z29+ Z30+ Z31+ Z32+ Z33+ Z34+ Z35+ Z36+ Z37+ Z38+ Z39+ Z40+ Z41+ Z42+ Z43+ Z44+ Z45+ Z46+ Z47+ Z48+ Z49+Z50+ Z51+ Z52+ Z53+ Z54+ Z55+ Z56+ Z57+ Z58+ Z59+Z60+ Z61+ Z62+ Z63+ Z64+ Z65+ Z66
    return ZW
