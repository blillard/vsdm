"""VSDM: Rate calculation implemented as matrix multiplication on (V,Q) basis.

    _vecK: combines gX, fgs2, and mI objects and returns mcalK matrices
        as 1d array, indexing (l, mv, mq) using wigner.Gindex

Note: RateCalc returns a scaled event rate mu = R/g_k0, rather than the rate R.
    For a specific sigma0 and exposure, multiply by k0 at the end:

    R[ell] = mu[ell] * g_k0(exp_kgyr, sigma0_cm2, rhoX_GeVcm3, ...)

"""

__all__ = ['RateCalc', '_vecK']

import math
import numpy as np
# import scipy.special as spf
# import vegas # numeric integration
import gvar # gaussian variables; for vegas
import time
# import quaternionic # For rotations
# import spherical #For Wigner D matrix
# import csv # file IO for projectFnlm
# import os.path
# import h5py # database format for mathcalI arrays

from .wigner import Gindex
from .utilities import *


def _vecK(gV, fsQ, mI, ellMax=None, lmod=1, use_gvar=False, sparse=False):
    """Combines <V|I|Q>, <gX|V>, and <Q|fgs2> into vector K(ell,mv,mq).

    Arguments:
        gV, fsQ: Fnlm instances
        mI: MakeMcalI instance
        ellMax: can manually set maximum value of ell.
            will still truncate ell at the minimum value from (gV, fsQ, mI).
        lmod: if 2, skips all odd values of ell
        use_gvar=True for gvar-valued matrix coefficients
        sparse: if True, then evaluates K directly from gV.f_nlm and fsQ.f_nlm.

    Returns:
        vecK, 1d array saving K(l,mv,mq), using index order from wigner.Gindex.

    Formats for physics inputs:
    * mI.mcalI[l, nv, nq], a 3d array.
        I(ell) = mI.mcalI[ell]
    * (gV)(fsQ).f_lm_n[ix, n], 2d arrays indexed by
        ix = lm_index.index((ell, m))
    * mI.mI_shape = (ellMax+1, nvMax+1, nqMax+1)
    """
    ellMaxGFI = np.min([gV.ellMax, fsQ.ellMax, mI.mI_shape[0]-1])
    if ellMax is None:
        ellMax = ellMaxGFI
    else:
        ellMax = np.min([ellMax, ellMaxGFI])
    nvMax = np.min([gV.nMax, mI.mI_shape[1]-1])
    nqMax = np.min([fsQ.nMax, mI.mI_shape[2]-1])
    # initialize...
    lenK = Gindex(ellMax, ellMax, ellMax, lmod=lmod) + 1
    vecK = np.zeros(lenK)

    # fill in values:
    if sparse:
        for v_nlm,gV_nlm in gV.f_nlm.items():
            if gV_nlm==0: continue
            (nv, ell, mv) = v_nlm
            if ell%lmod!=0: continue
            for q_nlm,fsQ_nlm in fsQ.f_nlm.items():
                (nq, ellq, mq) = q_nlm
                if ellq != ell: continue
                if fsQ_nlm==0: continue
                ix_K = Gindex(ell, mv, mq, lmod=lmod)
                Ilvq = mI.getI_lvq_analytic((ell, nv, nq))
                vecK[ix_K] += gV_nlm * Ilvq * fsQ_nlm
        return vecK
    ### ELSE: (not sparse)
    # this _makeFarray does use_gvar only if gV.use_gvar:
    fLMn_gV = gV._makeFarray(use_gvar=use_gvar)
    # this _makeFarray does use_gvar only if fsQ.use_gvar
    fLMn_fsQ = fsQ._makeFarray(use_gvar=use_gvar)
    # trim the widths of the arrays to the minimum (ellMax, nvMax, nqMax)
    fLMn_gV = gV.f_lm_n[:, 0:nvMax+1]
    fLMn_fsQ = fsQ.f_lm_n[:, 0:nqMax+1]
    if not mI.evaluated:
        mI.update_mcalI((ellMax, nvMax, nqMax), {}, analytic=True)
    if use_gvar:
        mcI = mI.mcalI_gvar[0:ellMax+1, 0:nvMax+1, 0:nqMax+1]
    else:
        mcI = mI.mcalI[0:ellMax+1, 0:nvMax+1, 0:nqMax+1]
    for ell in range(0, ellMax+1, lmod):
        for mv in range(-ell, ell+1):
            # get index for f_lm_n[lm -> x] vectors
            if (ell,mv) in gV.lm_index:
                xlm_v = gV.lm_index.index((ell,mv))
            else:
                continue
            for mq in range(-ell, ell+1):
                # map (mv, mq) to the index ix_K for the matrix K(ell):
                ix_K = Gindex(ell, mv, mq, lmod=lmod)
                # get index for f_lm_n[lm -> x] vectors
                if (ell,mq) in fsQ.lm_index:
                    xlm_q = fsQ.lm_index.index((ell,mq))
                else:
                    continue
                # combine vectors with mcalI(ell) matrix.
                # may have one or both of gV and fsQ with use_gvar==True.
                if use_gvar and gV.use_gvar:
                    gvecM = np.array([gvar.gvar(flmn[0], flmn[1])
                                      for flmn in fLMn_gV[xlm_v]])
                else:
                    gvecM = fLMn_gV[xlm_v]
                if use_gvar and fsQ.use_gvar:
                    fvecM = np.array([gvar.gvar(flmn[0], flmn[1])
                                      for flmn in fLMn_fsQ[xlm_q]])
                else:
                    fvecM = fLMn_fsQ[xlm_q]
                mxI = mcI[ell]
                vecK[ix_K] = gvecM @ mxI @ fvecM
    return vecK


class RateCalc():
    """Evaluates rate from <nlm|gX> and <nlm|fgs2>, including rotations.

    This is the dimensionless version of the rate, mu_R = R/g_k0,
        where g_k0 contains the dependence on the exposure mass*time,
        cross section normalization, local DM mass density, v0 and q0.

    Input:
        gV, fsQ: Fnlm instances in V and Q spaces.
        mI: an McalI instance with modelDMSM including DM mass and form factor,
            in the same V and Q bases as gV and fsQ.
        ellMax,lmod: can impose new restrictions on the values of ell
    """
    # To evaluate with long list of rotations, evaluate mathcalK matrix,
    #     then get mu(ell) = Tr(G^T * K) for each G(rotation)
    def __init__(self, gV, fsQ, mI, ellMax=None, lmod=None,
                 use_gvar=False, sparse=False):
        # gV is a projectFnlm instance for gtilde
        # fsQ is a projectFnlm instance for fgs
        # mI is a mathcalI instance
        # Any of these can be drawn from data
        # Does not include exposure-dependent prefactor g_k0
        self.gV = gV
        self.fsQ = fsQ
        self.mI = mI
        self.use_gvar = use_gvar
        if lmod is None:
            if gV.center_Z2 or fsQ.center_Z2 or mI.center_Z2:
                # only need one to be true to set ell==even:
                self.lmod = 2
            else:
                self.lmod = 1
        else:
            self.lmod = lmod
        self.ellMax = np.min([gV.ellMax, fsQ.ellMax, mI.mI_shape[0]-1])
        if ellMax is not None and ellMax < self.ellMax:
            self.ellMax = ellMax
        self.nvMax = np.min([gV.nMax, mI.mI_shape[1]-1])
        self.nqMax = np.min([fsQ.nMax, mI.mI_shape[2]-1])
        #module for rotations:
        t0 = time.time()
        self.vecK = _vecK(gV, fsQ, mI, lmod=self.lmod,
                          use_gvar=use_gvar, sparse=sparse)
        self.t_eval = time.time() - t0

    def tr_K_l(self):
        """List of Tr(K[l]) values (for rate without rotation)."""
        trKl = []
        for ell in range(0, self.ellMax, self.lmod):
            Klmm = 0.
            for m in range(-ell, ell+1):
                Klmm += self.vecK[Gindex(ell, m, m, lmod=self.lmod)]
            trKl += [Klmm]
        return np.array(trKl)


    def mu_garray(self, gvec_list, lmod_g=None):
        """Partial rate mu[ell] = R[ell]/g_k0 for a list of WignerG gvec vectors.

        Arguments:
        * gvec_array: a list of WignerG 'gvec' to evaluate
            gvec: a 1d WignerG vector of type WignerG.G(R) for rotation 'R'.
        * lmod_g: if lmod for gvec does not match self.lmod, set lmod_g here.

        Output:
        * list of mu[ell] = R[ell]/g_k0 vectors for ell = 0,...ellMax
            mu_l is the same axis=0 length as the 2d gvec_array
        """
        if lmod_g is None: # assume that indexing of gvec matches K
            lmod_g = self.lmod
        if len(np.shape(gvec_list))==1:
            gvec_array = np.array([gvec_list])
        else:
            gvec_array = np.array(gvec_list)
        n_rotations, len_G = np.shape(gvec_array)
        # ensure that gvec includes all ell up to ellMax:
        ellMax = self.ellMax
        if len_G < Gindex(ellMax, ellMax, ellMax, lmod=lmod_g):
            ellMax = 0
            for l in range(lmod_g, self.ellMax+1, lmod_g):
                if Gindex(l, l, l, lmod=lmod_g) <= len_G:
                    ellMax = l
        # for partial rate R(l), split G and K into sections of constant ell:
        lmod = 1
        if lmod_g==2 or self.lmod==2:
            lmod = 2
        # perform the calculation using only the local 'lmod'
        ells = [l for l in range(0, ellMax+1, lmod)]
        # dict version:
        gR_l = {}
        k_l = {}
        for l in ells:
            start_k = Gindex(l, -l, -l, lmod=self.lmod)
            end_k = Gindex(l, l, l, lmod=self.lmod)
            k_l[l] = self.vecK[start_k:end_k+1]
            start_g = Gindex(l, -l, -l, lmod=lmod_g)
            end_g = Gindex(l, l, l, lmod=lmod_g)
            gR_l[l] = gvec_array[:, start_g:end_g+1]
        mu_l = np.zeros((n_rotations, len(ells)))
        for ix_l,l in enumerate(ells):
            mu_l[:, ix_l] = gR_l[l] @ k_l[l]
        return mu_l

    def mu_R_l(self, wG):
        """Partial rate R[ell]/g_k0 for a WignerG instance wG."""
        return self.mu_garray(wG.G_array, lmod_g=wG.lmod)

    def mu_R(self, wG):
        """Total rate sum_ell R[ell]/g_k0 for a WignerG instance wG."""
        return self.mu_garray(wG.G_array, lmod_g=wG.lmod).sum(axis=1)

#
