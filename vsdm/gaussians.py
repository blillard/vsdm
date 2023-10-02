"""Analytic results for gaussian functions.

Functions:
    normG_nli_integrand: analytic radial integrand for '\mathcal G'
    GaussianF: a class for functions that are defined as sums of gaussians
    GaussianFnlm: a Basis class with additional methods for GaussianF functions
"""

__all__ = ['GaussianF', 'GaussianFnlm', 'normG_nli_integrand']

import math
import numpy as np
import scipy.special as spf
import vegas # numeric integration
import gvar # gaussian variables; for vegas
# import time
# import quaternionic # For rotations
# import spherical #For Wigner D matrix
# import csv # file IO for projectFnlm
# import os.path
# import h5py # database format for mathcalI arrays

# from .units import *
from .utilities import *
from .basis import Basis


def normG_nli_integrand(radR_nlu, u_i, sigma_i, n, ell, u):
    """For mathcalG_nl(u_i, sigma_i) gaussian and radR_nlu radial function.

    Using spf.ive(z) = spf.exp(-z) * spf.iv(z) to control exponential factors.
    """
    z = (2*u_i*u)/sigma_i**2
    if z==0:
        if ell==0:
            ivefact = 4/math.sqrt(math.pi)
        else:
            return 0
    else:
        ivefact = math.sqrt(8/z) * spf.ive(ell+0.5, z)
    measure = u**2/sigma_i**3
    expfact = math.exp(-(u-u_i)**2/sigma_i**2)
    return measure * expfact * ivefact * radR_nlu(n, ell, u)


class GaussianF():
    """Format for describing a function as a sum of Gaussians.

    Format:
        g = sum_i c_i * g_i
        g_i: normalized 3d gaussian, units of 1/sigma_i**3
            g_i = exp(-(u-uSph_i)**2/sigma_i**2) / (pi**1.5 * sigma_i**3)
            int(d^3u g_i) = 1.
            int(d^3u g) = sum_i c_i
    Input:
        gvec_list: a list of gvec = (c_i, uSph_i, sigma_i).
            c_i: amplitude. Has units of (sigma_i**3 g_i)
            uSph_i: Spherical vector indicating center of gaussian
                uSph_i = (u_i, theta_i, phi_i)
            sigma_i: dispersion, in units of [u]
        -> if None or []
    Methods:
        gU: returns g(u) = sum_i c_i gaussian_i
        rescaleGaussianF: returns gFactor*fU as gveclist
    """
    def __init__(self, gvec_list):
        if gvec_list is None:
            gvec_list = []
        self.gvec_list = gvec_list
        self.N_gaussians = len(gvec_list)
        self.is_gaussian = (self.N_gaussians > 0)

    def gU(self, uSph):
        "The function g(u) in spherical coordinates."
        # Has units of c_i/sigma_i**3
        sum_i = 0.
        # (u, theta, phi) = uSph
        (ux, uy, uz) = sph_to_cart(uSph)
        for gvec in self.gvec_list:
            (c_i, uSph_i, sigma_i) = gvec
            (ux_i, uy_i, uz_i) = sph_to_cart(uSph_i)
            du2 = (ux - ux_i)**2 + (uy - uy_i)**2 + (uz - uz_i)**2
            gaussnorm = math.pi**(-1.5) * sigma_i**(-3) * math.exp(-du2/sigma_i**2)
            sum_i += c_i * gaussnorm
        return sum_i

    def rescaleGaussianF(self, gFactor):
        "gvec_list for the function gFactor*fU(u)."
        return [(gvec[0]*gFactor, gvec[1], gvec[2]) for gvec in self.gvec_list]


class GaussianFnlm(Basis, GaussianF):
    """Tools for projecting GaussianF functions onto an |nlm> basis.

    Input:
        bdict: describes the Basis
        gvec_list: specifies the GaussianF function.
            If None or [], this is a 'null gaussian'
    Methods:
        Gnl_i: evaluates G_nli for (u_i, sigma_i) and self.radRn(n, l)
            can save result to self.G_nli_dict
        g_nlm: returns <g|nlm> for this function
            <g|nlm> = sum_i (c_i Y_lm(u_i) G_nli)
            Reads value from G_nli_dict if possible, otherwise runs Gnl_i
        G_nli_array: formats G_nli_dict into 3d numpy array
    """
    def __init__(self, bdict, gvec_list):
        Basis.__init__(self, bdict)
        GaussianF.__init__(self, gvec_list) # can be None or []
        self.G_nli_dict = {} # format: G_nli_array[n,l,i] = G_nli



    def Gnl_i(self, n, ell, i, vegas_params, saveGnli=True):
        "Integrates Gnl_i for radRn(n,l) function."
        gvec = self.gvec_list[i]
        (c_i, uSph_i, sigma_i) = gvec
        (u_i, theta_i, phi_i) = uSph_i
        header = "\tGnl_i for (n,l,i): {}".format((n, ell, i))
        headerA = "\tGnl_i(A) for (n,l,i): {}".format((n, ell, i))
        headerB = "\tGnl_i(B) for (n,l,i): {}".format((n, ell, i))
        def integrand_Gnl(u1d):
            [u] = u1d
            return normG_nli_integrand(self.radRn, u_i, sigma_i, n, ell, u)
        if self.basis['type']=='wavelet' and n!=0:
            # split integrand into two regions...
            [umin, umid, umax] = self._baseOfSupport_n(n, getMidpoint=True)
            # Volume here in terms of u, not u/u0!
            volume_A = [[umin, umid]] # 1d
            volume_B = [[umid, umax]] # 1d
            mGnl_A = self.doVegas(integrand_Gnl, volume_A, vegas_params,
                                  printheader=headerA)
            mGnl_B = self.doVegas(integrand_Gnl, volume_B, vegas_params,
                                  printheader=headerB)
            mathGnl = mGnl_A + mGnl_B
        else:
            [umin, umax] = self._baseOfSupport_n(n, getMidpoint=False)
            volume_u = [[umin, umax]] # 1d
            mathGnl = self.doVegas(integrand_Gnl, volume_u, vegas_params,
                                   printheader=header)
        if saveGnli:
            self.G_nli_dict[(n,ell,i)] = mathGnl
        return mathGnl

    def getGnlm(self, nlm, vegas_params, saveGnli=True):
        "The result <g|nlm>. Reads from G_nli_dict when possible."
        (n, ell, m) = nlm
        # Note: c_i does not affect value of Gnl_i. It appears here in cY_i
        sum_g = 0.0
        for i,gvec in enumerate(self.gvec_list):
            (c_i, uSph_i, sigma_i) = gvec
            (u_i, theta_i, phi_i) = uSph_i
            if (n,ell,i) in self.G_nli_dict.keys():
                gnli = self.G_nli_dict[(n,ell,i)]
            else:
                gnli = self.Gnl_i(n, ell, i, vegas_params, saveGnli=saveGnli)
            cY_i = c_i * ylm_real(ell, m, theta_i, phi_i)
            sum_g += cY_i * gnli
        return sum_g / self.u0**3

    def updateGnlm(self, nlm, vegas_params, saveGnli=True):
        "The result <g|nlm>. Forces numeric evaluation."
        (n, ell, m) = nlm
        # Note: c_i does not affect value of Gnl_i. It appears here in cY_i
        sum_g = 0.0
        for i,gvec in enumerate(self.gvec_list):
            (c_i, uSph_i, sigma_i) = gvec
            (u_i, theta_i, phi_i) = uSph_i
            gnli = self.Gnl_i(n, ell, i, vegas_params, saveGnli=saveGnli)
            cY_i = c_i * ylm_real(ell, m, theta_i, phi_i)
            sum_g += cY_i * gnli
        return sum_g  / self.u0**3

    def G_nli_array(self, nMax, ellMax):
        "Creates np.array from G_nli_dict"
        garray = gvar.gvar(1., 0)*np.zeros([nMax+1, ellMax+1, self.N_gaussians],
                                           dtype='object')
        for nli,value in self.G_nli_dict.items():
            (n,l,i) = nli
            if n <= nMax and l <= ellMax:
                garray[nli] = value
        return garray

    def distEnergyG(self):
        "Total 'energy' int(d^3u g^2)"
        sumE = 0.0
        for i in range(self.N_gaussians):
            gvec_i = self.gvec_list[i]
            (c_i, uSph_i, sigma_i) = gvec_i
            sigma2_i = sigma_i**2
            ui = np.array(sph_to_cart(uSph_i))
            u2i = ui.dot(ui)
            for j in range(self.N_gaussians):
                gvec_j = self.gvec_list[j]
                (c_j, uSph_j, sigma_j) = gvec_j
                sigma2_j = sigma_j**2
                uj = np.array(sph_to_cart(uSph_j))
                u2j = uj.dot(uj)
                sigma2_ij = sigma2_i*sigma2_j/(sigma2_i + sigma2_j)
                u_ij = (sigma2_j*ui + sigma2_i*uj)/(sigma2_i + sigma2_j)
                u2_ij = u_ij.dot(u_ij)
                exp_ij = np.exp(u2_ij/sigma2_ij - u2i/sigma2_i - u2j/sigma2_j)
                sumE += c_i*c_j*exp_ij / (np.pi*1.5 * (sigma2_i+sigma2_j)**1.5)
        return sumE / self.u0**3




















#
