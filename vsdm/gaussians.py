"""Analytic results for gaussian functions.

Functions:
    normG_nli_integrand: analytic radial integrand for '\mathcal G'
    GaussianF: a class for functions that are defined as sums of gaussians
    Gnli: a Basis class with additional methods for GaussianF functions
"""

__all__ = ['GaussianF', 'Gnli', 'normG_nli_integrand']

import math
import numpy as np
import scipy.special as spf
import gvar # gaussian variables; for vegas

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
    return measure * expfact * ivefact * radR_nlu(n, u, l=ell)


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

    def __call__(self, uSph):
        return self.gU(uSph)

    def rescaleGaussianF(self, gFactor):
        "gvec_list for the function gFactor*fU(u)."
        return [(gvec[0]*gFactor, gvec[1], gvec[2]) for gvec in self.gvec_list]


class Gnli(Basis, GaussianF):
    """Tools for projecting GaussianF functions onto an |nlm> basis.

    Input:
        bdict: describes the Basis
        gvec_list: specifies the GaussianF function.
            If None or [], this is a 'null gaussian'
    Methods:
        Gnl_i: evaluates G_nli for (u_i, sigma_i) and self._r_n_x(n, l)
            can save result to self.G_nli
        g_nlm: returns <g|nlm> for this function
            <g|nlm> = sum_i (c_i Y_lm(u_i) G_nli)
            Reads value from G_nli if possible, otherwise runs Gnl_i
        G_nli_array: formats G_nli into 3d numpy array
    """
    def __init__(self, bdict, gvec_list):
        Basis.__init__(self, bdict)
        GaussianF.__init__(self, gvec_list) # can be None or []
        self.G_nli = {} # format: G_nli_array[n,l,i] = G_nli

    def Gnl_i(self, n, ell, i, integ_params, saveGnli=True):
        "Integrates Gnl_i for _r_n_x(n,l) function."
        gvec = self.gvec_list[i]
        (c_i, uSph_i, sigma_i) = gvec
        (u_i, theta_i, phi_i) = uSph_i
        x_i = u_i/self.u0
        xsigma_i = sigma_i/self.u0
        header = "\tGnl_i for (n,l,i): {}".format((n, ell, i))
        headerA = "\tGnl_i(A) for (n,l,i): {}".format((n, ell, i))
        headerB = "\tGnl_i(B) for (n,l,i): {}".format((n, ell, i))
        def integrand_Gnl(x1d):
            [x] = x1d
            return normG_nli_integrand(self._r_n_x, x_i, xsigma_i, n, ell, x)
        if self.basis['type']=='wavelet' and n!=0:
            # split integrand into two regions...
            [xmin, xmid, xmax] = self._x_baseOfSupport(n, getMidpoint=True)
            # Volume here in terms of u, not u/u0!
            volume_A = [[xmin, xmid]] # 1d
            volume_B = [[xmid, xmax]] # 1d
            mGnl_A = NIntegrate(integrand_Gnl, volume_A, integ_params,
                                printheader=headerA)
            mGnl_B = NIntegrate(integrand_Gnl, volume_B, integ_params,
                                printheader=headerB)
            mathGnl = mGnl_A + mGnl_B
        else:
            [xmin, xmax] = self._x_baseOfSupport(n, getMidpoint=False)
            volume_x = [[xmin, xmax]] # 1d
            mathGnl = NIntegrate(integrand_Gnl, volume_x, integ_params,
                                 printheader=header)
        if saveGnli:
            self.G_nli[(n,ell,i)] = mathGnl
        return mathGnl

    def getGnlm(self, nlm, integ_params, saveGnli=True):
        "The result <g|nlm>. Reads from G_nli when possible."
        (n, ell, m) = nlm
        # Note: c_i does not affect value of Gnl_i. It appears here in cY_i
        sum_g = 0.0
        for i,gvec in enumerate(self.gvec_list):
            (c_i, uSph_i, sigma_i) = gvec
            (u_i, theta_i, phi_i) = uSph_i
            if (n,ell,i) in self.G_nli.keys():
                gnli = self.G_nli[(n,ell,i)]
            else:
                gnli = self.Gnl_i(n, ell, i, integ_params, saveGnli=saveGnli)
            cY_i = c_i * ylm_real(ell, m, theta_i, phi_i)
            sum_g += cY_i * gnli
        return sum_g / self.u0**3

    def _doGnlm(self, nlm, integ_params, saveGnli=True):
        "The result <g|nlm>. Forces numeric evaluation."
        (n, ell, m) = nlm
        # Note: c_i does not affect value of Gnl_i. It appears here in cY_i
        sum_g = 0.0
        for i,gvec in enumerate(self.gvec_list):
            (c_i, uSph_i, sigma_i) = gvec
            (u_i, theta_i, phi_i) = uSph_i
            gnli = self.Gnl_i(n, ell, i, integ_params, saveGnli=saveGnli)
            cY_i = c_i * ylm_real(ell, m, theta_i, phi_i)
            sum_g += cY_i * gnli
        return sum_g  / self.u0**3

    def G_nli_array(self, nMax, ellMax, use_gvar=False):
        "Creates np.array from G_nli"
        if use_gvar:
            arraySize = [nMax+1, ellMax+1, self.N_gaussians, 2]
        else:
            arraySize = [nMax+1, ellMax+1, self.N_gaussians]
        g_array = np.zeros(arraySize)
        for nli,value in self.G_nli.items():
            (n,l,i) = nli
            if n <= nMax and l <= ellMax:
                if use_gvar:
                    if type(value) is gvar._gvarcore.GVar:
                        garray[nli] = [value.mean, value.sdev]
                    else:
                        garray[nli][0] = value.mean
                elif type(value) is gvar._gvarcore.GVar:
                    garray[nli] = value.mean
                else:
                    garray[nli] = value
        return garray

    def norm_energy(self):
        "Total 'energy' int(d^3x g**2) = int(d^3u/u0**3 g**2)."
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
                sumE += c_i*c_j*exp_ij / (np.pi**1.5 * (sigma2_i+sigma2_j)**1.5)
        return sumE / self.u0**3




#
