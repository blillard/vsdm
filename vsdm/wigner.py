"""Implementation of WignerG rotation matrices for real spherical harmonics.

"""

__all__ = ['WignerG', 'testD_lm', 'testG_lm']

# import math
import numpy as np
# import scipy.special as spf
# import vegas # numeric integration
# import gvar # gaussian variables; for vegas
# import time
import quaternionic # For rotations
import spherical #For Wigner D matrix
# import csv # file IO for projectFnlm
# import os.path
# import h5py # database format for mathcalI arrays

from .utilities import *


def _applyR_thetaphi(R, theta, phi):
    x, y, z = sph_to_cart([1, theta, phi])
    v = quaternionic.array(0, x, y, z)
    vp = R * v / R
    vpx, vpy, vpz = vp.imag
    r, th, ph = cart_to_sph([vpx, vpy, vpz])
    return (th, ph)

def testD_lm(l, m, printout=False):
    wigD = spherical.Wigner(l)
    R = quaternionic.array([2, 5, 3, 7]).normalized
    theta = 0.4*np.pi
    phi = 1.3*np.pi
    th_p, ph_p = _applyR_thetaphi(1/R, theta, phi)
    Ylm_R_direct = ylm_cx(l, m, th_p, ph_p)
    # WignerD matrix
    D = wigD.D(R)
    Ylm_R = 0.
    Ylm_R_T = 0.
    Ylm_R_star = 0.
    Ylm_R_dag = 0.
    for mp in range(-l, l+1):
        D_mp_m = D[wigD.Dindex(l, mp, m)]
        Ylm_R += D_mp_m * ylm_cx(l, mp, theta, phi)
        Ylm_R_star += np.conjugate(D_mp_m) * ylm_cx(l, mp, theta, phi)
        D_m_mp = D[wigD.Dindex(l, m, mp)]
        Ylm_R_T += D_m_mp * ylm_cx(l, mp, theta, phi)
        Ylm_R_dag += np.conjugate(D_m_mp) * ylm_cx(l, mp, theta, phi)
    lbls = ['D', 'D_T', 'D_star', 'D_dagger']
    vals = [Ylm_R, Ylm_R_T, Ylm_R_star, Ylm_R_dag]
    diffs = [Ylm_R_direct - y for y in vals]
    eps = 1e-12
    spherical_D_is = []
    for j in range(len(vals)):
        diff = diffs[j]
        lbl = lbls[j]
        if np.abs(diff) < eps:
            spherical_D_is += [lbl]
    if printout:
        print('Ylm(R^(-1) * x): {}'.format(Ylm_R_direct))
        print('D_(k,m)*Ylk(x): {}'.format(Ylm_R))
        print('D_(m,k)*Ylk(x): {}'.format(Ylm_R_T))
        print('D*_(k,m)*Ylk(x): {}'.format(Ylm_R_star))
        print('D*_(m,k)*Ylk(x): {}'.format(Ylm_R_dag))
        print('differences:')
        for j in range(len(vals)):
            diff = diffs[j]
            lbl = lbls[j]
            print('\t{}: {}'.format(lbl, diff))
        print('version of Wigner D(R) provided by spherical.D(R):',
              spherical_D_is)
    return spherical_D_is


def testG_lm(l, m, printout=True):
    wigG = WignerG(l)
    R = quaternionic.array([2, 5, 3, 7]).normalized
    theta = 0.4*np.pi
    phi = 1.3*np.pi
    th_p, ph_p = _applyR_thetaphi(1/R, theta, phi)
    Ylm_R_direct = ylm_real(l, m, th_p, ph_p)
    # WignerD matrix
    gL = wigG.G_l(R)
    mxGl = gL[l]
    Ylm_R = 0.
    for mp in range(-l, l+1):
        G_mp_m = mxGl[l+mp, l+m]
        Ylm_R += G_mp_m * ylm_real(l, mp, theta, phi)
    diff = Ylm_R_direct - Ylm_R
    eps = 1e-12
    if printout:
        print('Ylm(R^(-1) * x): {}'.format(Ylm_R_direct))
        print('G_(k,m)*Ylk(x): {}'.format(Ylm_R))
        print('difference: {}'.format(diff))
    return diff

class WignerG():
    """Assembles the real form of the Wigner D matrix.

    Arguments:
        ellMax: largest ell for which G(R) is needed
        center_Z2: if True, then only need even values of ell

    Returns:
        self.Glist, self.rotations: dictionaries for saving G_ell matrices
            indexed 0,1,... by order in which G_ell(R) is evaluated

    Method:
        G_ell(R, save=True): calculates G for specified rotation
            if save: add rotation to self.rotations, G_ell to self.Glist
            returns dictionary gL, gL[ell] = G(ell), gL['R'] = R

    NOTE: Convention for Wigner D matrix:
    In terms of the Wigner "little d" matrix, d^(ell), my D^(ell) matrix is:

        D^(ell)_{m',m} = exp(-i m' alpha) * d^(ell)_{m'm}(beta) * exp(-i m gamma)

    for an active rotation with z-y-z Euler angles (alpha, beta, gamma).
    This is intended to match the convention in the documentation of 'spherical'.
    However, spherical v1.0.14 Wigner.D returns the complex conjugate of D,
        rather than D(l,mp,m)(R) = <l,mp| R |l,m> for active rotation R.

    The function testD_lm(l,m) tests whether it is D(l,mp,m) or its complex
        conjugate that is returned by Wigner.D(R), and adjusts the calculation
        of WignerG accordingly.
    """
    def __init__(self, ellMax, center_Z2=False):
        self.wigD = spherical.Wigner(ellMax)
        self.ellMax = ellMax
        # evaluate once per rotation: mxD = wigD.D(rotation)
        #     this "matrix" is saved as 1d array of coefficients...
        # get coefficient for index (ell, mprime, m):
        #     D^ell_{mp,m} = mxD[wigD.Dindex(ell, mp, m)]
        self.rotations = {} # save list of rotations
        self.Glist = {} # dict of G_ell matrices for each rotation
        self.rIndex = -1
        self.center_Z2 = center_Z2
        if ellMax > 0:
            self.conj_D = ('D_star' in testD_lm(ellMax, 1))
        else:
            self.conj_D = False

    def G_l(self, R, save=False):
        """Calculates G(ell) for all ell=0...ellMax.

        Arguments:
        * R: an SO(3) element, in quaternion representation
        * save: if True, adds R to self.rotations, and G_l(R) to self.Glist

        Output:
        * gL, a dict of G(ell) matrices, gL[ell] = G(ell).
            Includes gL['R'] entry, the value of the quaternion R

        """
        gL = {}
        gL['R'] = R
        # R is a quaternion: doesn't need to be a unit quaternion
        # mxD = self.wigD.D(R)
        if self.conj_D:
            # to match the definition of D(R) in spherical v1.0:
            mxD = np.conjugate(self.wigD.D(R))
        else:
            mxD = self.wigD.D(R)
        for ell in range(self.ellMax+1):
            gL[ell] = np.zeros([2*ell+1, 2*ell+1])
            if self.center_Z2 and ell%2!=0: continue
            # mp=m=0:
            ix00 = (ell, ell)
            d_00 = mxD[self.wigD.Dindex(ell, 0, 0)]
            # mp=0, m>0
            if ell==0:
                continue
            # row index j, column index k
            d_0p_k = np.array([[(-1)**k*mxD[self.wigD.Dindex(ell, 0, k)]
                                for k in range(1, ell+1)]])
            d_p0_j = np.array([[(-1)**j*mxD[self.wigD.Dindex(ell, j, 0)]]
                               for j in range(1, ell+1)])
            d_mp_jk = np.array([[(-1)**k*mxD[self.wigD.Dindex(ell, -j, k)]
                                for k in range(1, ell+1)] for j in range(1, ell+1)])
            d_pp_jk = np.array([[(-1)**(j+k)*mxD[self.wigD.Dindex(ell, j, k)]
                                for k in range(1, ell+1)] for j in range(1, ell+1)])
            G_mm_jk = np.real(d_pp_jk) - np.real(d_mp_jk)
            G_mp_jk = -np.imag(d_pp_jk) + np.imag(d_mp_jk)
            G_pm_jk = np.imag(d_pp_jk) + np.imag(d_mp_jk)
            G_pp_jk = np.real(d_pp_jk) + np.real(d_mp_jk)
            G_m0_j = - np.sqrt(2) * np.imag(d_p0_j)
            G_p0_j = np.sqrt(2) * np.real(d_p0_j)
            G_0m_k = np.sqrt(2) * np.imag(d_0p_k)
            G_0p_k = np.sqrt(2) * np.real(d_0p_k)
            G_00 = np.real(d_00)
            G_mm = np.flip(np.flip(G_mm_jk, axis=1), axis=0)
            G_mp = np.flip(G_mp_jk, axis=0)
            G_pm = np.flip(G_pm_jk, axis=1)
            G_m0 = np.flip(G_m0_j, axis=0)
            G_0m = np.flip(G_0m_k, axis=1)
            # concatenate first along the k axis:
            _G_m = np.concatenate((G_mm, G_m0, G_mp), axis=1)
            _G_0 = np.concatenate((G_0m, [[G_00]], G_0p_k), axis=1)
            _G_p = np.concatenate((G_pm, G_p0_j, G_pp_jk), axis=1)
            # concatenate results along the j axis:
            mxG = np.concatenate((_G_m, _G_0, _G_p), axis=0)
            gL[ell] = mxG
        if save:
            self.rIndex += 1 # new index for new entry
            self.rotations[self.rIndex] = R
            self.Glist[self.rIndex] = gL
        return gL

    def G(self, l, m, mp, R):
        if l%2==1 and self.center_Z2==True:
            return 0.
        gL = self.G_l(R, save=False)
        return gL[l][l-m, l-mp]
