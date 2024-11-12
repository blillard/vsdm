"""Implementation of WignerG rotation matrices for real spherical harmonics.

"""

__all__ = ['WignerG', 'Gindex', 'testD_lm', 'testG_lm']

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

def Gindex(l, m, k, lmod=1):
    if lmod==2:
        if l%2!=0:
            return None
        # only including even l
        return int(l*(4*l**2 - 6*l - 1)/6) + (2*l+1)*(l+m) + (l+k)
    # else:
    return int(l*(4*l**2-1)/3) + (2*l+1)*(l+m) + (l+k)

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
    # WignerG, matrix form
    gL = wigG.G_l(R)
    mxGl = gL[l]
    Ylm_M = 0.
    for mp in range(-l, l+1):
        G_mp_m = mxGl[l+mp, l+m]
        Ylm_M += G_mp_m * ylm_real(l, mp, theta, phi)
    diffM = Ylm_R_direct - Ylm_M
    # WignerG, vector form
    gv = wigG.G(R)
    Ylm_V = 0.
    for mp in range(-l, l+1):
        G_mp_m = gv[wigG.Gindex(l, mp, m)]
        Ylm_V += G_mp_m * ylm_real(l, mp, theta, phi)
    diffV = Ylm_R_direct - Ylm_V
    if printout:
        print('Ylm(R^(-1) * x): {}'.format(Ylm_R_direct))
        print('(M) G_(k,m)*Ylk(x): {}'.format(Ylm_M))
        print('(M) difference: {}'.format(diffM))
        print('(V) G_(k,m)*Ylk(x): {}'.format(Ylm_V))
        print('(V) difference: {}'.format(diffV))
    return diffM, diffV


class WignerG():
    """Assembles the real form of the Wigner D matrix.

    Arguments:
        ellMax: largest ell for which G(R) is needed
        lmod: if ==2, then only use even values of ell

    Returns:
        self.G_array: saved values of G_ell(R) matrices. All G(l,m,mp) values
            are concatenated into a single 1d array, 'gvec', indexed by
            Gindex(l,m,mp). Each row of G_array corresponds to a rotation R,
            ordered according to the calls to self.G(R, save=True).
         self.rotations: saved values of quaternions R, for each row in G_array.

    Method:
        G(R, save=True): calculates G for specified rotation (quaternion) R
            if save: add rotation to self.rotations, G_vec to self.G_array.
            returns a 1d array 'gvec' of G(l,m,mp), indexed by Gindex(l,m,mp).
        Gindex(l, m, mp): returns the index in gvec that matches these values
            of (l, m, mp).
        G_lmk(l,m,mp,R): calculates gvec(R), and returns its (l,m,mp) component.
            Multiple calls to G_lmk are inefficient: it is generally faster to
            find G(l,m,mp) from gvec[Gindex(l, m, mp)].
        G_l(R): returns a dictionary of 2d arrays, one for each ell (gL[ell]).
            No longer used in ratecalc.

    Convention for Wigner D matrix:
    In terms of the Wigner "little d" matrix, d^(ell), my D^(ell) matrix is:

        D^(ell)_{m',m} = exp(-i m' alpha) * d^(ell)_{m'm}(beta) * exp(-i m gamma)

    for an active rotation with z-y-z Euler angles (alpha, beta, gamma).
    This is intended to match the convention in the documentation of 'spherical'.

    Note: spherical v1.0.14 Wigner.D returns the complex conjugate of this D(R).
    The function testD_lm(l,m) tests whether it is D(l,mp,m) or its complex
        conjugate that is returned by Wigner.D(R), and adjusts the calculation
        of WignerG accordingly.
    """
    def __init__(self, ellMax, rotations=[], lmod=1, run_D_test=False):
        self.wigD = spherical.Wigner(ellMax)
        self.lmod = lmod
        if lmod==2 and ellMax%2==1:
                ellMax += -1
        self.ellMax = ellMax
        # evaluate once per rotation: mxD = wigD.D(rotation)
        #     this "matrix" is saved as 1d array of coefficients...
        # get coefficient for index (ell, mprime, m):
        #     D^ell_{mp,m} = mxD[wigD.Dindex(ell, mp, m)]
        self.rotations = [] # list of rotations to evaluate at init
        self.G_array = [] # array of G_ell 1d arrays for each rotation

        # current version of spherical returns D^* rather than D.
        self.conj_D = True
        if run_D_test: # check whether conj_D should be updated:
            if ellMax > 0:
                # correct for different definition of D from spherical
                self.conj_D = ('D_star' in testD_lm(ellMax, 1))

        # evaluate G(l) for all rotations in the list:
        if len(rotations) > 0:
            # initialize self.Glist
            for R in rotations:
                self.G(R, save=True)

    def G(self, R, save=False):
        """Calculates G(l, m, mp) as 1d array.

        Arguments:
        * R: an SO(3) element, in quaternion representation
        * save: if True, adds R to self.rotations, and G_l(R) to self.Glist

        Output:
        * gvec, a 1d array. G(l, m, mp) = gvec[Gindex(l, m, mp, lmod=lmod)]
        """
        if self.conj_D:
            # to match the definition of D(R) in spherical v1.0:
            mxD = np.conjugate(self.wigD.D(R))
        else:
            mxD = self.wigD.D(R)
        # begin with l=0:
        d_00 = mxD[self.wigD.Dindex(0, 0, 0)]
        gvec = [np.real(d_00)]
        # continue with l=lmod (1 or 2):
        for ell in range(self.lmod, self.ellMax+1, self.lmod):
            # mp=m=0:
            d_00 = mxD[self.wigD.Dindex(ell, 0, 0)]
            # mp=0, m>0
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
            gvec_l = [glmk for glmk in mxG.reshape((2*ell+1)**2)]
            gvec += gvec_l
        if save:
            self.rotations += [R]
            self.G_array += [gvec]
        return np.array(gvec)

    def G_l_dict(self, lmod=None, ellMax=None):
        """Converts self.G_array into a dict format, indexed by ell.

        Each entry [l] is a human-readable matrix, sized (2*l+1) by (2*l+1)
        Optional arguments:
        * lmod: can set lmod=2 even if self.lmod=1.
        * ellMax: can truncate G_l at a smaller value of ellMax < self.ellMax
        """
        if lmod is None:
            lmod = self.lmod
        elif lmod==1 and self.lmod==2: # can't generate odd l if self.lmod==2.
            lmod = 2
        if ellMax is None or ellMax > self.ellMax:
            ellMax = self.ellMax
        ells = [l for l in range(0, ellMax+1, lmod)]
        gld = {}
        for l in ells:
            start, end = self.Gindex(l, -l, -l), self.Gindex(l, l, l)
            gld[l] = self.G_array[:, start:end+1]
        return gld

    def G_l(self, R):
        """Calculates G(ell) matrices for all ell=0...ellMax.

        Arguments:
        * R: an SO(3) element, in quaternion representation
        * save: if True, adds R to self.rotations, and G_l(R) to self.Glist

        Output:
        * gL, a dictionary of G(ell) matrices, G(l,m,k) = gL[l][l+m, l+k]

        ! This method has been superceded by 1d array self.G(R) in v.0.3.3.
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
            if ell%self.lmod!=0: continue
            # mp=m=0:
            d_00 = mxD[self.wigD.Dindex(ell, 0, 0)]
            # mp=0, m>0
            if ell==0:
                gL[ell] = np.array([[np.real(d_00)]])
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
        return gL

    def G_lmk(self, l, m, k, R):
        """Returns a single element of G(l,m,k).

        Note: this method calculates gvec = self.G(R) from scratch. For multiple
            values of (l, m, k), it is faster to evaluate gvec once, and to
            find G_lmk from gvec[self.Gindex(l, m, k)].
        """
        if l%self.lmod!=0:
            return 0.
        gv = self.G(R, save=False)
        return gv[self.Gindex(l, m, k, lmod=self.lmod)]

    def Gindex(self, l, m, k):
        return Gindex(l, m, k, lmod=self.lmod)







#
