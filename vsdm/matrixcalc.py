"""VSDM: Calculates McalI matrix, for velocity (V) and momentum (Q) bases.

Contents:
    _mathI_vq: performs integration
    class McalI: assembles mcalI_gvar list of matrices
        includes methods for save/read/import with hdf5 files
"""

__all__ = ['McalI', 'MakeMcalI']

import math
import numpy as np
import vegas # numeric integration
import gvar # gaussian variables; for vegas
import time
import os.path
import h5py


from .utilities import *
from .portfolio import *
from .units import *
from .analytic import *
from .basis import haar_sph_value, tophat_value, Basis


#

class McalI():
    """Handles the I(ell) matrices for one choice of DM,SM particle models.

    Arguments:
        V, Q: Basis instances for velocity and momentum spaces
            Can be supplied as 'basis' dictionary, or as class instance
        modelDMSM: dict describing DM and SM particle physics and DeltaE
            DeltaE: energy transfer (for kinematics)
            mX: DM particle mass
            fdm_n: DM-SM scattering form factor index (e.g. n=0, n=2)
            mSM: SM target particle mass (e.g. electron, mElec)
        mI_shape: (optional) determines the initial size of the I_lnvq array
            mI_shape = (ellMax+1, nvMax+1, nqMax+1) = mcalI.shape
        f_type: labels 'type' for Portfolio. Default value TNAME_mcalI='mcalI'

    Primary variable:
        mcalI: a 3d array of I(l, nv, nq)
        mcalI_gvar: a 3d gvar array of I(l, nv, nq)

    """
    def __init__(self, V, Q, modelDMSM, mI_shape=(0,0,0), f_type=None,
                 use_gvar=False, do_mcalI=False, center_Z2=False):
        t0 = time.time()
        self.evaluated = False
        if type(V)==dict:
            V = Basis(V)
        if type(Q)==dict:
            Q = Basis(Q)
        self.modelDMSM = modelDMSM
        if f_type is None:
            self.f_type = TNAME_mcalI # default value: see portfolio.py
        else:
            self.f_type = f_type
        self.use_gvar = use_gvar
        # v0 and q0 for normalization:
        v0 = V.u0
        q0 = Q.u0
        self.center_Z2 = center_Z2
        # if center_Z2, then all odd ell are skipped in update_mcalI()
        # DM and SM particle model parameters:
        mX = modelDMSM['mX'] # DM mass
        fdm_n = modelDMSM['fdm_n'] # DM-SM scattering form factor index
        mSM = modelDMSM['mSM'] # SM particle mass (mElec)
        DeltaE = modelDMSM['DeltaE'] # DM -> SM energy transfer
        muSM = mX*mSM/(mX + mSM) # reduced mass
        qStar = math.sqrt(2*mX*DeltaE)
        vStar = qStar/mX
        # Mass-dependent prefactor kI scales with q0 and v0:
        self.kI = 0.5 * (q0/v0)/mX * ((q0/v0)/muSM)**2
        # initialize mcalI[(ell, nv, nq)]:
        self.mI_shape = mI_shape
        self.mcalI = np.zeros(mI_shape)
        if self.use_gvar:
            self.mcalI_gvar = gvar.gvar(1,0)*np.zeros(mI_shape, dtype='object')
        # Memorialize:
        self.V = V
        self.Q = Q
        self.v0 = v0
        self.q0 = q0
        self.modelDMSM['muSM'] = muSM
        self.modelDMSM['kI'] = self.kI
        self.modelDMSM['qStar'] = qStar
        self.modelDMSM['vStar'] = vStar
        # save mcalI parameters, including V.basis and Q.basis, in one place:
        self.attributes = self._mergeBasisVQ(V, Q)
        self.attributes.update(self.modelDMSM)
        self.attributes['v0'] = self.v0
        self.attributes['q0'] = self.q0
        # self.attributes is the dict that will be saved in hdf5 files
        if do_mcalI:
            lnvq_max = (mI_shape[0]-1, mI_shape[1]-1, mI_shape[2]-1)
            self.update_mcalI(lnvq_max, dict(verbose=False), analytic=True)
            self.evaluated = True
        self.t_eval = time.time() - t0


    @staticmethod
    def _mergeBasisVQ(V, Q):
        Vbasis = V.basis
        Qbasis = Q.basis
        out = {}
        for key,value in Vbasis.items():
            okey = 'V_' + str(key)
            out[okey] = value
        for key,value in Qbasis.items():
            okey = 'Q_' + str(key)
            out[okey] = value
        return out

    def _pad_mcalI(self, new_lnvq):
        """Pads mcalI with zeros to accommodate new lnvq value."""
        larger_shape = compare_index_to_shape(new_lnvq, self.mI_shape)
        if larger_shape==self.mI_shape:
            # Nothing to be done here. (_update_size not allowed to shrink I)
            return
        # Otherwise, pad mcalI with zeros to make it large enough:
        self.mcalI.resize(larger_shape)
        if self.use_gvar:
            self.mcalI_gvar.resize(larger_shape)
        self.mI_shape = larger_shape


    def getI_lvq(self, lnvq, integ_params):
        """Calculates I^(ell)_{vq} integrals for this (V,Q) basis.

        Arguments:
            lnvq = (ell,nv,nq): basis vector indices for <V| and |Q>
            integ_params: integration parameters neval, nitn, etc.
        Returns:
            mcalI^(ell)_{nv,nq}(modelDMSM)
        """
        (ell, nv, nq) = lnvq
        # v0 and q0 for normalization:
        v0 = self.v0
        q0 = self.q0
        # DM and SM particle model parameters
        mX = self.modelDMSM['mX'] # DM mass
        fdm_n = self.modelDMSM['fdm_n'] # DM-SM scattering form factor index
        mSM = self.modelDMSM['mSM'] # SM particle mass (mElec)
        DeltaE = self.modelDMSM['DeltaE'] # DM -> SM energy transfer
        nitn_init = integ_params['nitn_init'] # reduced mass
        nitn = integ_params['nitn']
        neval = integ_params['neval']
        verbose = integ_params['verbose']
        if 'neval_init' in integ_params:
            neval_init = integ_params['neval_init']
        else:
            neval_init = neval
        if verbose:
            print("Calculating <V|I|Q> for (l,nv,nq): ", lnvq)
        # Establish integration region: check whether the integrand vanishes
        #     if QVrange does not include any v > vMin(q), then mcalI = 0.
        Qrange = self.Q._u_baseOfSupport(nq, getMidpoint=False)
        Vrange = self.V._u_baseOfSupport(nv, getMidpoint=False)
        QVrange = [Qrange, Vrange] #Integration volume
        # if v_type is not 'laguerre' and q_type is not 'laguerre':
        [qA, qB] = Qrange
        q0vmin = math.sqrt(2*mX*DeltaE)  #where vmin(q) is minimized
        minVmin = math.sqrt(2*DeltaE/mX) #smallest possible value of vmin(q)
        # if q0vmin is not within Qrange, then the smallest vmin(q) in the
        #     region in integration is instead:
        if qA > q0vmin:
            minVmin = DeltaE/qA + qA/(2*mX)
        elif qB < q0vmin:
            minVmin = DeltaE/qB + qB/(2*mX)
        # If vB < minVmin, then I=0.
        [vA, vB] = Vrange
        if vB < minVmin:
            if verbose:
                print("\t v < vMin(q) for all (v,q) in integration range.")
                print("mathcalI = 0.\n")
            return gvar.gvar(0,0)
        # If vB > minVmin, then the integral can be nonzero:
        def integrand(qv):
            [q,v]=qv
            vMinq = DeltaE/q + q/(2*mX)
            if v < vMinq:
                return 0
            else:
                partQ = self.Q.r_n(nq, q, l=ell) * fdm2_n(q, fdm_n)
                partV = self.V.r_n(nv, v, l=ell) * plm_norm(ell,0,vMinq/v)
                return self.kI * q*v/(q0*v0)**2 * partQ * partV
            # Integrand units cancel against units from QV integration area
        integrator = vegas.Integrator(QVrange)
        integrator(integrand, nitn=nitn_init, neval=neval_init) #training
        I_lnvq = integrator(integrand, nitn=nitn, neval=neval) #result
        if verbose:
            print(I_lnvq.summary())
        return I_lnvq


    def getI_lvq_analytic(self, lnvq, verbose=False):
        """Analytic calculation for I(ell) matrix coefficients.

        Only available for 'tophat' and 'wavelet' bases (so far).

        Arguments:
            lnvq = (ell, nv, nq)
            verbose: whether to print output
        """
        (ell, nv, nq) = lnvq
        V = self.V
        Q = self.Q
        v0 = self.v0
        q0 = self.q0
        mX = self.modelDMSM['mX'] # DM mass
        fdm_n = self.modelDMSM['fdm_n'] # DM-SM scattering form factor index
        mSM = self.modelDMSM['mSM'] # SM particle mass (mElec)
        DeltaE = self.modelDMSM['DeltaE'] # DM -> SM energy transfer
        muSM = self.modelDMSM['muSM'] # reduced mass
        # Integrand is written in terms of dimensionless vStar and qStar:
        qStar = math.sqrt(2*mX*DeltaE)
        vStar = qStar/mX
        qStar = self.modelDMSM['qStar']
        vStar = self.modelDMSM['vStar']
        v_type = V.basis['type']
        q_type = Q.basis['type']
        if verbose:
            print("Calculating <V|I|Q> for (l,nv,nq): ", (ell,nv,nq))
        # scale_factor = DeltaE**2 / (q0**2 * v0**2)
        try:
            assert v_type=='tophat' or v_type=='wavelet', "Analytic method only available for 'tophat' and 'wavelet'"
            assert q_type=='tophat' or q_type=='wavelet', "Analytic method only available for 'tophat' and 'wavelet'"
        except AssertionError:
            # returns the error type, so one can try again with getI_lvq()
            return AssertionError
        commonFactor = ((q0/v0)**3/(2*mX*muSM**2) * (2*DeltaE/(q0*v0))**2
                        *(q0_fdm/qStar)**(2*fdm_n))
        n_regions = [1,1]
        if v_type=='tophat':
            [v1, v2] = V._u_baseOfSupport(nv, getMidpoint=False)
            A_v = tophat_value(v1/v0, v2/v0, dim=3) #normalize to 0<x<1
            n_regions[0] = 1
        elif v_type=='wavelet':
            [v1, v2, v3] = V._u_baseOfSupport(nv, getMidpoint=True)
            if nv==0:
                A_v = haar_sph_value(nv, dim=3)
                n_regions[0] = 1
            else:
                [A_v, B_v] = haar_sph_value(nv, dim=3) #already normalized
                n_regions[0] = 2
        if q_type=='tophat':
            [q1, q2] = Q._u_baseOfSupport(nq, getMidpoint=False)
            A_q = tophat_value(q1/q0, q2/q0, dim=3) #normalize to 0<x<1
            n_regions[1] = 1
        elif q_type=='wavelet':
            [q1, q2, q3] = Q._u_baseOfSupport(nq, getMidpoint=True)
            if nq==0:
                A_q = haar_sph_value(nq, dim=3)
                n_regions[1] = 1
            else:
                [A_q, B_q] = haar_sph_value(nq, dim=3) #already normalized
                n_regions[1] = 2
        # There is always an A_v A_q term:
        v12_star = [v1/vStar, v2/vStar]
        q12_star = [q1/qStar, q2/qStar]
        term_AA = A_v*A_q * mI_star(ell, fdm_n, v12_star, q12_star)
        # There are only B-type contributions if V or Q uses wavelets
        term_AB, term_BA, term_BB = 0, 0, 0
        if n_regions[0]==2:
            v23_star = [v2/vStar, v3/vStar]
            term_BA = B_v*A_q * mI_star(ell, fdm_n, v23_star, q12_star)
        if n_regions[1]==2:
            q23_star = [q2/qStar, q3/qStar]
            term_AB = A_v*B_q * mI_star(ell, fdm_n, v12_star, q23_star)
        if n_regions==[2,2]:
            term_BB = B_v*B_q * mI_star(ell, fdm_n, v23_star, q23_star)
        Ilvq = commonFactor * (term_AA + term_BA + term_AB + term_BB)
        if verbose:
            print("\t Ilvq = ", Ilvq)
        return Ilvq


    def updateIlvq(self, lnvq, integ_params, analytic=False):
        """Calculates Ilvq(l,nv,nq) using numeric or analytic method.

        Arguments:
            lnvq: index (l, nv, nq)
            integ_params: a dict of integ_params style if analytic==False,
                or a single entry {'verbose': bool} for analytic==True
                If empty, assume verbose=False.
            analytic: whether to try analytic method or not
        """
        if analytic:
            if 'verbose' in integ_params:
                verbose = integ_params['verbose']
            else:
                verbose = False
            # try getI_lvq_analytic:
            Ilvq = self.getI_lvq_analytic(lnvq, verbose=verbose)
            # if a basis without supported analytic methods is used,
            # then getI_lvq_analytic returns AssertionError
            if Ilvq!=AssertionError:
                self.mcalI[lnvq] = Ilvq
                if self.use_gvar:
                    self.mcalI_gvar[lnvq] = gvar.gvar(1., 0) * Ilvq
                return Ilvq
        #else: analytic method not possible
        analytic = False
        Ilvq = self.getI_lvq(lnvq, integ_params)
        if self.use_gvar:
            self.mcalI_gvar[lnvq] = Ilvq
        self.mcalI[lnvq] = Ilvq.mean
        return Ilvq

    def update_mcalI(self, lnvq_max, integ_params, analytic=True):
        """Calculates entire Ilvq array in series, up to lnvq_max.

        Arguments:
            lnvq_max: (l_max, nv_max, nq_max)
            integ_params: a dict of integ_params style if analytic==False,
                or a single entry {'verbose': bool} for analytic==True
                If empty, assume verbose=False.
            analytic: whether to try analytic method or not
        """
        (l_max, nv_max, nq_max) = lnvq_max
        for l in range(l_max + 1):
            if self.center_Z2 and l%2!=0: continue
            for nv in range(nv_max + 1):
                for nq in range(nq_max + 1):
                    lnvq = (l, nv, nq)
                    self.updateIlvq(lnvq, integ_params, analytic=analytic)
                    # this style sets analytic -> False if it is attempted on
                    # incompatible basis functions
        self.evaluated = True


    def writeMcalI(self, hdf5file, modelName, alt_type=None):
        """Saves Ilvq array to hdf5 under name 'modelName'.

        Recommend using DeltaE and DM parameters as the model label:
            e.g. modelName = (DeltaE)/(mX, fdm_n)
        """
        if alt_type is not None:
            typeName = alt_type
        else:
            typeName = self.f_type
        #saves I(ell) arrays in [ell, nv, nq] to HDF file
        dset_attrs = {}
        for lbl,value in self.attributes.items():
            if value is not None:
                dset_attrs[lbl] = value
        folio = Portfolio(hdf5file, extra_types=[typeName])
        dn_mean = DNAME_I + '_mean' # intended dbase name
        dn_sdev = DNAME_I + '_sdev' # intended dbase name
        if self.use_gvar:
            Ilvq_mean, Ilvq_sdev = splitGVARarray(self.mcalI_gvar)
        else:
            Ilvq_mean = self.mcalI
            Ilvq_sdev = np.zeros_like(Ilvq_mean)
        dname_mean = folio.add_folio(typeName, modelName, dn_mean,
                                     data=Ilvq_mean, attrs=dset_attrs)
        dname_sdev = folio.add_folio(typeName, modelName, dn_sdev,
                                     data=Ilvq_sdev, attrs=dset_attrs)
        return dname_mean, dname_sdev


    def write_update(self, hdf5file, modelName, d_pair, newdata,
                     alt_type=None):
        """Adds mcalI[l,nv,nq] values to existing hdf5 datasets.

        Arguments:
            hdf5file, modelName, d_pair: specify datasets to use
                hdf5file/type/modelName/
                    d_pair[0]: _mean ,   d_pair[1]: _sdev.
            newdata: a dict of I[l,nv,nq] coefficients, in style of mcalI.
        """
        if alt_type is not None:
            typeName = alt_type
        else:
            typeName = self.f_type
        folio = Portfolio(hdf5file, extra_types=[typeName])
        folio.update_gvar(typeName, modelName, d_pair, newdata=newdata)


    def importMcalI(self, hdf5file, modelName, d_pair=[], alt_type=None):
        """Imports mcalI from hdf5, adds to f_nlm.

        Arguments:
            d_pair: pair of _mean and _sdev files to merge;
                or just _mean, if len(dnames)==1.
            hdf5file, modelName, alt_type: sets fileName/typeName/modelName
                with typeName = self.f_type unless an alt_type is provided

        Returns:
            dataIlvq: the Ilvq gvar array from hdf5file
            attrs: contains basis parameters for V and Q
                * recommend checking that this matches self.attributes.

        Updates self.mcalI with all (l,nv,nq) in dataIlvq.
        """
        if alt_type is not None:
            typeName = alt_type
        else:
            typeName = self.f_type
        folio = Portfolio(hdf5file, extra_types=[typeName])
        dataIlvq, attrs = folio.read_gvar(typeName, modelName, d_pair=d_pair)
        basis_info = str_to_bdict(attrs)
        dshape = np.shape(dataIlvq)
        # if needed, pad self.mcalI to accommodate data:
        corner_lnvq = [d+1 for d in dshape]
        self._pad_mcalI(corner_lnvq)
        for l in range(dshape[0]):
            for nv in range(dshape[1]):
                for nq in range(dshape[2]):
                    self.mcalI[l, nv, nq] = dataIlvq[l, nv, nq].mean
                    if self.use_gvar:
                        self.mcalI_gvar[l, nv, nq] = dataIlvq[l, nv, nq]
        return dataIlvq, attrs

#


class MakeMcalI(McalI):
    """Facilitates the parallel evaulation of I(l,nv,nq).

    Arguments:
        V, Q, modelDMSM, mI_shape: for McalI
        integ_params: dict for integ_params, or {'verbose': verbose}
            if analytic==True
        analytic: whether to attempt an analytic evaluation of mcalI
        lnvq_list: which (l,nv,nq) to evaluate during initialization
        hdf5file: for storing results in hdf5. If 'None', skips this step
        f_type: 'type' for hdf5 file, default 'mcalI'
        modelName: hdf5 dataset name.
            If already taken, dname_manager finds a related available name
    """
    # Reserving DeltaE, mX, FDM2 for parallelization
    # Note: for variable nMax(ell) or variable precision, run init() with ellMax=0, then do add_ell for ell=1..ellMax
    def __init__(self, V, Q, modelDMSM, mI_shape, integ_params={},
                 analytic=True, lnvq_list=None, f_type=TNAME_mcalI,
                 hdf5file=None, modelName=None):
        McalI.__init__(self, V, Q, modelDMSM, mI_shape=mI_shape, f_type=f_type)
        # adds mI_shape, f_type to self.
        if lnvq_list is not None:
            for lnvq in lnvq_list:
                self.updateIlvq(lnvq, integ_params, analytic=analytic)
            # save empty mcalI array to hdf5 if no lnvq_list provided
            if hdf5file is not None:
                dname_mean, dname_sdev = self.writeMcalI(hdf5file, modelName)
        elif hdf5file is not None: # but lnvq_list is None:
            dname_mean = DNAME_I + '_mean'
            # if hdf5file doesn't exist, make one
            if not os.path.isfile(hdf5file):
                dname_mean, dname_sdev = self.writeMcalI(hdf5file, modelName)
            # if modelName doesn't have dataset_mean, make one:
            if not _dset_exists(hdf5file, f_type, modelName, dname_mean):
                dname_mean, dname_sdev = self.writeMcalI(hdf5file, modelName)
            else: # don't overwrite an existing dataset with a blank one.
                dname_sdev = DNAME_I + '_sdev'
        # Now, the hdf5file exists.
        # If an lnvq_list was provided, mcalI contains these values.
        # If no lnvq_list, and there was no hdf5 model,
        #     then it has been initialized with gvar zeros.
        # If no lnvq_list, and a model entry exists in hdf5file,
        #     it has not been overwritten.
        # Going forwards, only use write_update to edit hdf5
        self.hdf5file = hdf5file
        self.modelName = modelName
        if hdf5file is not None:
            self.d_pair = [dname_mean, dname_sdev]
        self.analytic = analytic
        self.ellMax = mI_shape[0]
        self.nvMax = mI_shape[1]
        self.nqMax = mI_shape[2]

    def add_lnvqs(self, lnvq_list, integ_params={}):
        newdata = {}
        for lnvq in lnvq_list:
            Ilvq = self.updateIlvq(lnvq, integ_params,
                                   analytic=self.analytic)
            newdata[lnvq] = Ilvq
        # wait until the end to write the list to hdf5
        self.write_update(self.hdf5file, self.modelName, self.d_pair, newdata)










#
