"""VSDM: Rate calculation implemented as matrix multiplication on (V,Q) basis.

    _mcalK: combines gX, fgs2, and mI objects and returns mcalK matrices

Note: RateCalc provides mu/g_k0, rather than mu.
    For specific sigma0 and exposure, multiply by k0 at the end.

Suggested order of calculations for complex problems:
* For every pair of (V,Q) basis choices:
    run mathcalI with a large list of (omegaS, mX, FDM_n), save to hdf5 file
    Use one hdf5 file for every pair of (V,Q) bases
    Make sure omegaS list includes every value relevant to fgs
    This process may be time-consuming, but is easily parallelized
* save projectFnlm objects for <gX|nlm> and <fgs|nlm> to CSV files
    Use different file for each basis choice and for each new gX or fgs function
* When RateCalc is needed, import gV, fsQ, mI from data, and apply rotation

NOTE: for large rotation list, it is wasteful to use RateCalc directly.
Define a vector version of RateCalc, indexed by (omegaS,mX,FDM_n):
  - For each (omegaS,mX,FDM_n), get mathK matrices
  - Suggestion: save K(omegaS,mX,FDM_n) to same hdf5 file as mathcalI,
        in 'group' specific to (gX,fgs) combination.
        Avoids keeping large numbers of matrices in memory
  - For each rotation, calculate G(ell) once and apply it to all mathcalK(ell)
  - Save rotation,Tr(G @ K) list for each (omegaS,mX,FDM_n)
This can be done in parallel for each distinct (gX,fgs), basis(V,Q) choice
"""

__all__ = ['RateCalc', '_mcalK', 'tr_mcalK']

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

# from .wigner import WignerG
from .utilities import *


def _mcalK(gV, fsQ, mI, ellMax=None, use_gvar=False, sparse=False):
    """Combines <V|I|Q>, <gX|V>, and <Q|fgs2> into matrices K(ell).

    Arguments:
        gV, fsQ: Fnlm instances
        mI: MakeMcalI instance
        use_gvar=True for gvar-valued matrix coefficients
        sparse: if True, then evaluates
    Returns:
        mathKell, a dictionary of K_{m,m'} matrices for ell=0,1,...

    Formats for physics inputs:
    * mI.mcalI[l, nv, nq], a 3d array.
        I(ell) = mI.mcalI[ell]
    * (gV)(fsQ).f_lm_n[ix, n], 2d arrays indexed by
        ix = _LM_to_x(ell, m, phi_symmetric=bool)
    * mI.mI_shape = (ellMax+1, nvMax+1, nqMax+1)
    """
    ellMaxGFI = np.min([gV.ellMax, fsQ.ellMax, mI.mI_shape[0]-1])
    if ellMax is None:
        ellMax = ellMaxGFI
    else:
        ellMax = np.min([ellMax, ellMaxGFI])
    nvMax = np.min([gV.nMax, mI.mI_shape[1]-1])
    nqMax = np.min([fsQ.nMax, mI.mI_shape[2]-1])
    theta_Zn = 1
    if gV.center_Z2 or fsQ.center_Z2:
        theta_Zn = 2 # only even values of ell are relevant
    # initialize...
    mathKell = {} # mu_ell = g_k0 * Tr(K^T * G_ell )
    for ell in range(ellMax+1):
        if use_gvar:
            mathKell[ell] = gvar.gvar(1,0)*np.zeros([2*ell+1, 2*ell+1])
        else:
            mathKell[ell] = np.zeros([2*ell+1, 2*ell+1])
    # fill in matrix.
    if sparse:
        for v_nlm,gV_nlm in gV.f_nlm.items():
            if gV_nlm==0: continue
            (nv, ell, mv) = v_nlm
            if ell%theta_Zn!=0: continue
            for q_nlm,fsQ_nlm in fsQ.f_nlm.items():
                (nq, ellq, mq) = q_nlm
                if ellq != ell: continue
                if fsQ_nlm==0: continue
                ix_K = (ell+mv, ell+mq) # index for mcalK[ell]
                Ilvq = mI.getI_lvq_analytic((ell, nv, nq))
                mathKell[ell][ix_K] += gV_nlm * Ilvq * fsQ_nlm
        return mathKell
    ### ELSE:
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
    for ell in range(ellMax+1):
        if ell%theta_Zn!=0: continue
        for mv in range(-ell, ell+1):
            # get index for f_lm_n[lm -> x] vectors
            if (ell,mv) in gV.lm_index:
                xlm_v = gV.lm_index.index((ell,mv))
            else:
                continue
            for mq in range(-ell, ell+1):
                # map (mv, mq) to the index ix_K for the matrix K(ell):
                ix_K = (ell+mv, ell+mq)
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
                mathKell[ell][ix_K] = gvecM @ mxI @ fvecM
    return mathKell


def tr_mcalK(gV, fsQ, mI, use_gvar=False):
    """Combines <V|I|Q>, <gX|V>, and <Q|fgs2> into matrices K(ell).

    Arguments:
        gV, fsQ: Fnlm instances
        mI: MakeMcalI instance
        use_gvar=True for gvar-valued matrix coefficients
    Returns:
        tr_Kl, a list of traces: tr_Kl[ell] = Tr(K[ell])

    Formats for physics inputs:
    * mI.mcalI[l, nv, nq], a 3d array.
        I(ell) = mI.mcalI[ell]
    * (gV)(fsQ).f_lm_n[ix, n], 2d arrays indexed by
        ix = _LM_to_x(ell, m, phi_symmetric=bool)
    * mI.mI_shape = (ellMax+1, nvMax+1, nqMax+1)
    """
    ellMax = np.min([gV.ellMax, fsQ.ellMax, mI.mI_shape[0]-1])
    nvMax = np.min([gV.nMax, mI.mI_shape[1]-1])
    nqMax = np.min([fsQ.nMax, mI.mI_shape[2]-1])
    fLMn_gV = gV._makeFarray(use_gvar=use_gvar)
    fLMn_fsQ = fsQ._makeFarray(use_gvar=use_gvar)
    # trim the widths of the arrays to the minimum (ellMax, nvMax, nqMax)
    fLMn_gV = gV.f_lm_n[:, 0:nvMax+1]
    fLMn_fsQ = fsQ.f_lm_n[:, 0:nqMax+1]
    if use_gvar:
        mcI = mI.mcalI_gvar[0:ellMax+1, 0:nvMax+1, 0:nqMax+1]
    else:
        mcI = mI.mcalI[0:ellMax+1, 0:nvMax+1, 0:nqMax+1]
    # one or both of these arrays might have azimuthal symmetry, m=0 for all ell
    if use_gvar:
        tr_Kl = gvar.gvar(1,0)*np.zeros([ellMax+1], dtype='object')
    else:
        tr_Kl = np.zeros([ellMax+1])
    for ell in range(ellMax+1):
        trk = 0.0
        for m in range(-ell, ell+1):
            # map (mv, mq) to the index ix_K for the matrix K(ell):
            # get index for f_lm_n[lm -> x] vectors
            if (ell,mv) in gV.lm_index:
                xlm_v = gV.lm_index.index((ell,mv))
            else:
                continue
            if (ell,mq) in fsQ.lm_index:
                xlm_q = fsQ.lm_index.index((ell,mq))
            else:
                continue
            # combine vectors with mcalI(ell) matrix:
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
            trk += gvecM @ mxI @ fvecM
        tr_Kl[ell] = trk
    return tr_Kl


class RateCalc():
    """Evaluates rate from <nlm|gX> and <nlm|fgs2>, including rotations.

    Input:
        gV, fsQ: projectFnlm instances in V and Q spaces.
            Can be taken from CSV data
        mI: a mathcalI instance with (omegaS,mX,FDM_n).
            Can be taken from hdf5 data

    For detector rotations, need matrices K(ell). Otherwise, just need Tr(K).
        .ellMax, .nvMax, .nqMax: minimum common ellMax, nvMax, nqMax
        .mathKell: dict, saving K(ell)_{m'm} matrices
        .trKell: dict, saving trace(K(ell)).
        .times: dict, saving evaluation times for various matrix operations
    * mathcalKell(ell, save) calculates K(ell) from fLM_n_gvar
        If save, adds result to self.mathKell[ell], and trace to self.trKell[ell]
    * muEll(rotation, save): calculates rate component mu_ell from K(ell),
        given G_ell(rotation). If save, writes output to self.mu_ell
    """
    # To evaluate with long list of rotations, evaluate mathcalK matrix,
    #     then get mu(ell) = Tr(G^T * K) for each G(rotation)
    def __init__(self, gV, fsQ, mI, use_gvar=False, sparse=True):
        # gV is a projectFnlm instance for gtilde
        # fsQ is a projectFnlm instance for fgs
        # mI is a mathcalI instance
        # Any of these can be drawn from data
        # Does not include exposure-dependent prefactor g_k0
        self.gV = gV
        self.fsQ = fsQ
        self.mI = mI
        self.use_gvar = use_gvar
        self.center_Z2 = False
        if gV.center_Z2 or fsQ.center_Z2:
            # only need one to be true to set ell==even:
            self.center_Z2 = True
        self.ellMax = np.min([gV.ellMax, fsQ.ellMax, mI.mI_shape[0]-1])
        self.nvMax = np.min([gV.nMax, mI.mI_shape[1]-1])
        self.nqMax = np.min([fsQ.nMax, mI.mI_shape[2]-1])
        #module for rotations:
        t0 = time.time()
        self.mcalK = _mcalK(gV, fsQ, mI, use_gvar=use_gvar, sparse=sparse)
        self.t_eval = time.time() - t0


    def tr_K_l(self):
        """List of Tr(K[l]) values (for rate without rotation)."""
        return [self.mcalK[ell].trace() for ell in range(self.ellMax+1)]


    def mu_l(self, G_ell_R, sparse=False):
        """Provides list of mu[ell]/g_k0 for ell = 0,1...ellMax.

        Argument:
        * G_l_R, a WignerG.G_l object for the rotation 'R'.
            G_l_R[l] = G^{(l)}, a matrix of size (2l+1)*(2l+1)

        Output:
        * list of mu[ell]/g_k0 for ell = 0,1...ellMax
        """
        if self.use_gvar:
            mu_l = gvar.gvar(1,0)*np.zeros([self.ellMax+1], dtype='object')
        else:
            mu_l = np.zeros([self.ellMax+1])
        if not sparse:
            for ell in range(self.ellMax+1):
                if self.center_Z2 and ell%2!=0: continue
                gellT = np.transpose(G_ell_R[ell])
                mu_l[ell] = np.trace(gellT @ self.mcalK[ell])
            return mu_l
        #else: sparse version
        nlmVlist = self.gV.f_nlm.keys()
        nlmQlist = self.fsQ.f_nlm.keys()
        for nlmV in nlmVlist:
            [nv, ell, mv] = nlmV
            if self.center_Z2 and ell%2!=0: continue
            if ell > self.ellMax: continue
            for nlmQ in nlmQlist:
                [nq, ellQ, mq] = nlmQ
                if ellQ != ell:
                    continue
                if rotation is None and mv!=mq:
                    continue
                if self.use_gvar:
                    Iell_nvnq = self.mI.mcalI_gvar[ell, nv, nq]
                    gt = self.gV.f_nlm[nlmV]
                    fgs = self.fsQ.f_nlm[nlmQ]
                else:
                    Iell_nvnq = self.mI.mcalI[ell, nv, nq]
                    gt = self.gV.f_nlm[nlmV].mean
                    fgs = self.fsQ.f_nlm[nlmQ].mean
                if rotation is None:
                    mu_l[ell] += gt * Iell_nvnq * fgs
                else:
                    Gell_mvmq = G_ell_R[ell][ell+mv, ell+mq]
                    mu_l[ell] += gt * Iell_nvnq * Gell_mvmq * fgs
        return mu_l
