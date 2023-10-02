"""VSDM: ProjectFnlm implements Basis integration for a specific function fSph.

"""

__all__ = ['ProjectFnlm']


import math
import numpy as np
import scipy.special as spf
import vegas # numeric integration
import gvar # gaussian variables; for vegas
import time
import csv # file IO for projectFnlm
import os.path
import h5py

from .basis import ylm_real
from .gaussians import GaussianFnlm
from .utilities import *
from .portfolio import *







class ProjectFnlm(GaussianFnlm):
    """Performs projection <fSph|nlm> for one 3d function fSph.

    Upon initialization, evaluates <f|nlm> for all nlm in nlmlist. Additional
      values of nlm can be added subsequently using updateFnlm method.
    Like Basis, ProjectFnlm does not require .units.

    Arguments:
        fSph: a function of 3d spherical uvec OR a GaussianF object
            Can have attributes: is_gaussian, phi_symmetric, etc
            These become attributes of the ProjectFnlm instance
        bdict: Basis parameters
        vegas_params: e.g. dict(neval=1e4, nitn=10, nitn_init=5, verbose=False)
        f_type: labels the type of function being projected, e.g. 'gX' or 'fs2'
            default label 'proj' defined in portfolio.py

    Returns:
        f_nlm: dictionary of <f|nlm>, indexed by (n,l,m), with gvar values
        f_lm_n: a single 2d array of coefficients,
            labelled by [x(ell,m), n], for x=0,1,2..., (ellMax+1)**2 - 1.
            See utilities.py.
            phi_symmetric: if so, only adds entries for m=0 coefficients.
                Length of array is always (ellMax+1)**2!
            NEW: need not be gvar valued.
    f_nlm is best for sparse (nlm) lists;
    f_lm_n is best for saving hdf5 files.

    Methods and variables:
        updateFnlm: adds new f[nlm], or overwrites old value
            e.g. with better precision
        _makeFarray: makes vectors of <f|nlm> with fixed (lm)
            self.phi_symmetric sets m=0.
            use_gvar=False for float-valued matrix output
        nlmFu, nlmFu_subset: returns f(uvec) from sums over <f|nlm>
        errorCalcs: integrates nlmFu(r) and original f(r), and (nlmFu-f)(r)
        errorTot, errorAbsTot: integrate (nlmFu-f)(r) and np.abs(nlmFu-f)(r)
        getVSOPbasis: returns a Basis instance of (basisType,u0,uMax)
    Methods for saving/reading/importing...
    * writeFnlm(hdf5file, nlmlist=None, modelName=''):
        saves f_nlm to readable CSV file, for all nlm, or nlm in nlmlist
        row format: ["modelName", "n", "l", "m", "f.mean", "f.sdev"]
        modelName is optional: can be used to save incompatible
            <f|nlm> to same file
    * writeFlmVecs(hdf5file,modelName=''):
        saves <f(lm)|n> vectors to CSV file
        row format: ["modelName", "l", "m", "fn.mean", "fn.sdev"],
            with fn as lists on [n=0,1,...nMax]
    * writeF_array(hdf5file,modelName):
        saves <f|(lm)n> array to HDF5 dataset in hdf5file/modelName
        row format: x = _LM_to_x(l,m) = 0,1,2,...l(l+2).
        RECOMMEND: modelName = 'gX/(index)' or 'fgs2/(index)' for model index
    * readFnlm(), readFlmVecs(hdf5file, modelName=None):
        opens hdf5file, returns data_fnlm dict using all entries
            (or only those with matching modelName)
    * importFnlm(), importFlmVecs(csvfile, modelName=None):
        writes data_fnlm from CSV into self.f_nlm
        Caution! import() will overwrite any existing f_nlm entries
    """
    def __init__(self, bdict, fSph, vegas_params, nlmlist=[], f_type=None,
                 csvsave_name=None, csvsave_model='', use_gvar=True):
        t0 = time.time()
        self.fSph = fSph # a function fSph(uvec), with uvec=(u,theta,phi)
        self.use_gvar = use_gvar
        self.vegas_params = vegas_params
        self.t_eval = 0.
        # Or, fSph may be a GaussianF instance:
        if hasattr(fSph, 'is_gaussian') and (fSph.is_gaussian):
            GaussianFnlm.__init__(self, bdict, fSph.gvec_list)
            # GaussianFnlm contains Basis
            # New ProjectFnlm instance does not inherit the G_nli_dict of fSph.
        else:
            # use GaussianFnlm to import Basis.__init__
            GaussianFnlm.__init__(self, bdict, None) # self.is_gaussian = False
        self.z_even = False
        self.phi_symmetric = False
        self.phi_cyclic = 1
        self.phi_even = False
        self.center_Z2 = False
        if hasattr(fSph, 'z_even') and fSph.z_even not in [0, None]:
            # if fSph is Z_2 symmetric in the z direction, only even ell are nonzero
            # preserved by rotation: NO
            self.z_even = self.fSph.z_even
        if hasattr(fSph, 'phi_symmetric') and fSph.phi_symmetric not in [0, None]:
            # complete azimuthal symmetry sets m=0
            # preserved by rotation: NO
            self.phi_symmetric = self.fSph.phi_symmetric
        if hasattr(fSph, 'phi_cyclic') and fSph.phi_cyclic not in [0, None]:
            # phi can have Zn symmetry (2,3,4,...), setting m%Zn = 0.
            # preserved by rotation: NO (unless rotation is about z axis)
            self.phi_cyclic = self.fSph.phi_cyclic
        if hasattr(fSph, 'phi_even') and fSph.phi_even not in [0, None]:
            # if f is even under phi -> -phi, then m >= 0 (real spherical Ylm).
            # preserved by rotation: NO
            self.phi_even = self.fSph.phi_even
        if hasattr(fSph, 'center_Z2') and fSph.center_Z2 not in [0, None]:
            # central inversion is theta,phi -> pi-theta,pi+phi.
            # some materials have this, even if they lack other symmetries
            # Consequence: only even ell are nonzero
            # preserved by rotation: YES
            self.center_Z2 = self.fSph.center_Z2
        if self.phi_symmetric:
            self.phi_even = True
        if self.z_even and self.phi_cyclic%2==0:
            self.center_Z2 = True
        if f_type is None:
            self.f_type = TNAME_generic # from portfolio
        else:
            self.f_type = f_type
        self.f_nlm = {} # Primary: <f|nlm> coefficients in gvar format
        # The following are to be updated by update_maxes():
        self.ellMax = 0
        self.nMax = 0
        # Unpack VEGAS integration parameters:
        nitn_init = vegas_params['nitn_init']
        nitn = vegas_params['nitn']
        neval = int(vegas_params['neval'])
        verbose = vegas_params['verbose']
        if 'neval_init' in vegas_params:
            neval_init = int(vegas_params['neval_init'])
        else:
            neval_init = neval
        # self.vegas_params = vegas_params
        if self.use_gvar:
            self.f2_energy = gvar.gvar(0,0)
        else:
            self.f2_energy = 0.0
        if nlmlist is not None:
            for nlm in nlmlist:
                self.update_maxes(nlm)
                if self.is_gaussian:
                    # use method from GaussianFnlm:
                    fnlm = self.getGnlm(nlm, vegas_params, saveGnli=True)
                    # save=True ensures that self.G_nli_dict is updated
                else:
                    # use method from Basis
                    fnlm = self.getFnlm(fSph, nlm, vegas_params)
                if self.use_gvar:
                    self.f_nlm[nlm] = fnlm
                else:
                    self.f_nlm[nlm] = fnlm.mean
                f2_power = fnlm**2
                self.f2_energy += f2_power
                if verbose:
                    print("Result: <f|{}> = {}".format(nlm, fnlm))
                    print("\ttotal energy: {}".format(self.f2_energy))
                if csvsave_name is not None:
                    # save as you go:
                    # append this f(nlm) to the CSV file 'csvsave_name',
                    self.writeFnlm_csv(csvsave_name, nlmlist=[nlm],
                                       modelName=csvsave_model)
        self.t_init = time.time() - t0
        self.t_eval = self.t_init
        # Add other metadata to basis dict
        for lbl in vegas_params:
            self.basis[lbl] = vegas_params[lbl]
        self.basis['is_gaussian'] = self.is_gaussian

    def getNLMlist(self):
        """Returns all (nlm) that have been added to <f|nlm>."""
        nlmlist = [nlm for nlm in self.f_nlm]
        return nlmlist

    def filter_fnlm(self, fnlm_min):
        """Returns a new f_nlm dict with all |fnlm| >= |fnlm_min|."""
        # compare to |fnlm_min| if it is negative:
        fnlm_min = math.fabs(fnlm_min)
        main_fnlm = {}
        for nlm,fnlm in self.f_nlm.items():
            # for gvar-type fnlm, '>=' uses value of fnlm.mean
            if math.fabs(fnlm) >= fnlm_min:
                main_fnlm[nlm] = fnlm
        return main_fnlm

    def f2nlm_energy(self):
        """Calculates 'sum(<f|nlm>**2)' from all coefficients."""
        if self.use_gvar:
            self.f2_energy = gvar.gvar(0,0)
        else:
            self.f2_energy = 0.0
        for f in self.f_nlm.values():
            self.f2_energy += f**2
        return self.f2_energy

    def getNLMpower(self, nMax=None, lMax=None):
        """Dist.power for all (nlm), sorted (descending)."""
        #assemble
        powerNLM = {}
        for nlm,fnlm in self.f_nlm.items():
            if ((nMax is not None and nlm[0]>nMax)
                or (lMax is not None and nlm[1]>lMax)):
                continue
            powerNLM[nlm] = fnlm**2
        #sort
        sortnlm = sorted(powerNLM.items(),
                        key=lambda z: z[1], reverse=True)
        powerNLM = {}
        for key,power in sortnlm:
            powerNLM[key] = power
        return powerNLM

    def getLMpower(self, lMax=None):
        """Dist.power in (lm), sorted (descending)."""
        #assemble
        powerLM = {}
        for nlm,fnlm in self.f_nlm.items():
            (n, l, m) = nlm
            if lMax is not None and l>lMax:
                continue
            if (l, m) in powerLM.keys():
                powerLM[(l, m)] += fnlm**2
            else:
                powerLM[(l, m)] = fnlm**2
        #sort
        sortlm = sorted(powerLM.items(),
                        key=lambda z: z[1], reverse=True)
        powerLM = {}
        for key,power in sortlm:
            powerLM[key] = power
        return powerLM

    def getLpower(self, lMax=None):
        """Dist.power in (l), sorted (descending)."""
        #assemble
        powerL = {}
        for nlm,fnlm in self.f_nlm.items():
            (n, l, m) = nlm
            if lMax is not None and l>lMax:
                continue
            if l in powerL.keys():
                powerL[l] += fnlm**2
            else:
                powerL[l] = fnlm**2
        #sort
        sortl = sorted(powerL.items(),
                       key=lambda z: z[1], reverse=True)
        powerL = {}
        for key,power in sortl:
            powerL[key] = power
        return powerL

    def getDistEnergy(self, vegas_params, integrateG=False):
        """Integral of f**2/self.u0**3 (the distributional energy, rescaled by u0).

        mSymmetry and lSymmetry restrict integration region to a subregion
            of solid angle.
        For l only expect lSymmetry=2, e.g. z <-> -z.
        m can take larger values than 2 (commonly 4), e.g. polygonal cylindrical symmetry

        Saves result to self.basis['distEnergy']
        """
        if self.is_gaussian and not integrateG:
            energyG = self.distEnergyG()
            energy = gvar.gvar(energyG, 0)
            return energy
        theta_Zn = 1
        if self.center_Z2:
            theta_Zn = 2
        theta_region = [0, math.pi/theta_Zn]
        if self.phi_symmetric:
            volume = [[0, self.uMax], theta_region]
            if integrateG:
                def power_u(u2d):
                    (r, theta) = u2d
                    uvec = (r, theta, 0.0)
                    return dV_sph(uvec) * self.fSph.gU(uvec)**2/self.u0**3
            else:
                def power_u(u2d):
                    (r, theta) = u2d
                    uvec = (r, theta, 0.0)
                    return dV_sph(uvec) * self.fSph(uvec)**2/self.u0**3
            # do VEGAS:
            energy = (self.doVegas(power_u, volume, vegas_params)
                      * 2*math.pi * theta_Zn)
            if vegas_params['verbose']==True:
                print('energy: {}'.format(energy))
            # include result in self.basis[]
            self.basis['distEnergy'] = energy
            return energy
        #else:
        phi_region = [0, 2*math.pi/self.phi_cyclic]
        volume = [[0, self.uMax], theta_region, phi_region]
        if integrateG:
            def power_u(uvec):
                # energy has units of uMax**3
                return dV_sph(uvec) * self.fSph.gU(uvec)**2/self.u0**3
        else:
            def power_u(uvec):
                # energy has units of uMax**3
                return dV_sph(uvec) * self.fSph(uvec)**2/self.u0**3
        energy = (self.doVegas(power_u, volume, vegas_params)
                  * self.phi_cyclic * theta_Zn)
        if vegas_params['verbose']==True:
            print('energy: {}'.format(energy))
        return energy

    def update_maxes(self, nlm):
        """Updates self.Max values to encompass 'nlm'."""
        (n,l,m) = nlm
        assert(int(math.fabs(m)) <= l), "Invalid (nlm): m out of range"
        assert(l >= 0), "Invalid (nlm): l out of range"
        if l > self.ellMax:
            self.ellMax = l
        if n > self.nMax:
            self.nMax = n

    def updateFnlm(self, nlm, vegas_params, csvsave_name=None, csvsave_model=''):
        """Calculates <f|nlm> for new (nlm), or overwrites existing <f|nlm>."""
        # Good for adding nlm to the list.
        # Or, recalculate f_nlm with better precision in vegas_params.
        t0 = time.time()
        self.update_maxes(nlm)
        fnlm = self.getFnlm(self.fSph, nlm, vegas_params, saveGnli=True)
        self.f_nlm[nlm] = fnlm
        t1 = time.time() - t0
        self.t_eval += t1
        self.basis['t_eval'] = self.t_eval
        if vegas_params['verbose']:
            print("Result: <f|{}> = {}".format(nlm, fnlm))
        if csvsave_name is not None:
            # save as you go:
            # append this f(nlm) to the CSV file 'csvsave_name',
            self.writeFnlm_csv(csvsave_name, nlmlist=[nlm],
                               modelName=csvsave_model)
        return fnlm #in case anyone wants it


    def nlmFu(self, uvec, nlmlist=None):
        """The VSOP reconstruction of f(r), from all <f|nlm>."""
        return self.nlmAssembleFu(self.f_nlm, uvec, nlmlist=nlmlist)


    def _makeFarray(self, use_gvar=False):
        """Creates 'secondary' formats for <f|nlm> from self.f_nlm.

        Makes self.f_lm_n, for hdf5 output.
            rows labeled by (lm) -> x = 0, 1, 2, ... l(l+2).
            if phi_symmetric: only make rows x = ell, all with m=0.
                See utilities._LM_to_x() for x(lm) function.
        """
        #make sure self.Max are up to date:
        for nlm in self.getNLMlist():
            self.update_maxes(nlm)
        if self.phi_symmetric:
            arraySize = [(self.ellMax+1), (self.nMax+1)]
        else:
            arraySize = [(self.ellMax+1)**2, (self.nMax+1)]
        if use_gvar:
            self.f_lm_n = gvar.gvar(1., 0)*np.zeros(arraySize, dtype='object')
        else:
            self.f_lm_n = np.zeros(arraySize)
        for nlm,f_gvar in self.f_nlm.items():
            (n,l,m) = nlm
            ix_y = n
            if self.center_Z2 and (l%2 != 0):
                continue
            if self.phi_symmetric and (m!=0):
                continue
            ix_x = _LM_to_x(l,m, phi_symmetric=self.phi_symmetric)
            if use_gvar:
                self.f_lm_n[ix_x, ix_y] = f_gvar
            else:
                self.f_lm_n[ix_x, ix_y] = f_gvar.mean
        return self.f_lm_n

    @staticmethod
    def addcombine_fnlms(fnlm_list):
        """Adds results from multiple f_nlm objects.

        Useful when f is split into f = f_1 + f_2 + ... + f_k,
            and <f_i|nlm> are calculated individually.
        """
        grand_fnlm = {}
        all_nlm = []
        # Step 1: find out which (nlm) need to be included.
        for fnlm in fnlm_list:
            # fnlm: a dict of type f_nlm
            for nlm in fnlm:
                if nlm not in all_nlm:
                    all_nlm += [nlm]
        # Step 2: fill grand_fnlm from all contributions fnlm[nlm]
        for nlm in all_nlm:
            gf = gvar.gvar(0.0, 0.0)
            for fnlm in fnlm_list:
                # skip fnlm if fnlm[nlm] DNE. Otherwise, add it to gf:
                if nlm in fnlm:
                    gf += fnlm[nlm]
            grand_fnlm[nlm] = gf
        return grand_fnlm



    def writeFnlm(self, hdf5file, modelName,
                  writeGnli=False, alt_type=None, use_gvar=False):
        """Adds hdf5 dataset to group 'modelName'.

        Arguments:
            hdf5file: the hdf5 file to open/write
            modelName: a group name for storing <f|nlm> datasets
        Saves to: hdf5file/typeName/modelName/dataset
            with typeName = self.f_type.
        Database names use the default portfolio.DNAME_F,
            which specifies that this object is an <f|nlm> dataset.
            Saves gvar .mean and .sdev arrays to _mean and _sdev datasets.

        Using utilities.dname_manager() to avoid name conflicts within
            the group 'modelName' (or 'typeName/modelName').
        Writing multiple versions to the same 'modelName' will save
            the later copies as 'modelName_2', 'modelName_3'...
        """
        if alt_type is not None:
            typeName = alt_type
        else:
            typeName = self.f_type
        self._makeFarray(use_gvar=use_gvar)
        # include the following information in .basis and the output:
        self.basis['nMax'] = self.nMax
        self.basis['ellMax'] = self.ellMax
        self.basis['phi_symmetric'] = self.phi_symmetric
        self.basis['t_eval'] = self.t_eval
        folio = Portfolio(hdf5file, extra_types=[typeName])
        dset_attrs = {} # save relevant basis info to hdf5:
        for lbl,value in self.basis.items():
            if value is not None:
                dset_attrs[lbl] = value
        dn_mean = DNAME_F + '_mean' # intended dbase name
        dn_sdev = DNAME_F + '_sdev' # intended dbase name
        if use_gvar:
            flm_n_mean, flm_n_sdev = splitGVARarray(self.f_lm_n)
        else:
            flm_n_mean = self.f_lm_n
            flm_n_sdev = np.zeros_like(flm_n_mean)
        dname_mean = folio.add_folio(typeName, modelName, dn_mean,
                                     data=flm_n_mean, attrs=dset_attrs)
        dname_sdev = folio.add_folio(typeName, modelName, dn_sdev,
                                     data=flm_n_sdev, attrs=dset_attrs)
        # add_folio returns the actual dname used
        if writeGnli and (self.is_gaussian): #also save mG_nli_array
            g_array = self.G_nli_array(self.nMax, self.ellMax)
            gn_mean = DNAME_G + '_mean'
            gn_sdev = DNAME_G + '_sdev'
            gnl_mean, gnl_sdev = splitGVARarray(g_array)
            gname_mean = folio.add_folio(typeName, modelName, gn_mean,
                                         data=gnl_mean, attrs=dset_attrs)
            gname_sdev = folio.add_folio(typeName, modelName, gn_sdev,
                                         data=gnl_sdev, attrs=dset_attrs)
        return dname_mean,dname_sdev

    def write_additional(self, hdf5file, modelName, d_pair, newdata={},
                         alt_type=None):
        """Adds <f|nlm> values to existing hdf5 datasets.

        Arguments:
            hdf5file, modelName, d_pair: specify datasets to use
                hdf5file/type/modelName/
                    d_pair[0]: _mean ,   d_pair[1]: _sdev.
            newdata: a dict of f[(nlm)] coefficients, in style of self.f_nlm.
        Note: _additional only works for f_nlm, not f_lm_n.
        """
        if alt_type is not None:
            typeName = alt_type
        else:
            typeName = self.f_type
        folio = Portfolio(hdf5file, extra_types=[typeName])
        data_out = {}
        for nlm,f in newdata.items():
            (n,l,m) = nlm
            ix_x = _LM_to_x(l,m, phi_symmetric=self.phi_symmetric)
            ix_y = n
            data_out[(ix_x, ix_y)] = f
        folio.update_gvar(typeName, modelName, d_pair, newdata=data_out)


    def importFnlm(self, hdf5file, modelName, d_pair, alt_type=None):
        """Imports <f|nlm> from hdf5, adds to f_nlm.

        Arguments:
            d_pair: pair of _mean and _sdev files to merge;
                or just _mean, if len(dnames)==1.
            hdf5file, modelName, alt_type: sets hdf5file/typeName/modelName
                with typeName = self.f_type unless an alt_type is provided

        Returns:
            dataFnlm: the f_lm_n gvar array from hdf5file
            bdict_info: contains basis parameters incl. 'phi_symmetric'
                * recommend checking that this matches self.basis.

        Adds or overwrites f_nlm for all (n,l,m) included in the file.
        Does not create/overwrite self.f_lm_n.
        """
        t0 = time.time()
        if alt_type is not None:
            typeName = alt_type
        else:
            typeName = self.f_type
        folio = Portfolio(hdf5file, extra_types=[typeName])
        dataFnlm, attrs = folio.read_gvar(typeName, modelName, d_pair)
        bdict_info = str_to_bdict(attrs)
        # Without 'phi_symmetric', can't make sense of f_lm_n array:
        assert('phi_symmetric' in bdict_info), "Unspecified mandatory item 'phi_symmetric' for hdf5 dataset."
        for x,row in enumerate(dataFnlm):
            ell,m = _x_to_LM(x, phi_symmetric=self.basis['phi_symmetric'])
            for n in len(row):
                self.f_nlm[(n,l,m)] = row[n]
        self.t_eval += time.time() - t0
        return dataFnlm, bdict_info

    def writeFnlm_csv(self, csvfile, nlmlist=None, modelName=''):
        """Saves <f|nlm> dictionary to CSV file.

        Arguments:
            csvfile: CSV file to create or append to
            nlmlist: specific coefficients to write out.
                If None, then write all entries of self.f_nlm
            modelName: optional marker saved to first column of csv
                used in read() to pick out matching coefficients

        Recommended use:
            For slow calculations, save each new <f|nlm> coefficient to
            a CSV file, as writeFnlm_csv(csvname, nlmlist=[nlm])
        Also a human-readable output, good for inspection of coefficients.
        """
        # t0 = time.time()
        #'a': Always append, never overwrite. If file does not exist, open(...,'a') creates it.
        # modelName is saved as 0th column, in case data of multiple types needs to be saved to the same file (avoid this)
        # Default: writeFnlm for all nlm that have been evaluated.
        #    Alternative: only the nlm in nlmlist.
        #    Can use this option to save-as-you-go, with nlmlist=[(nlm)].
        # If this is the first time we write to csvfile, generate a header row:
        makeHeader = not os.path.exists(csvfile)
        self.basis['nMax'] = self.nMax
        self.basis['ellMax'] = self.ellMax
        self.basis['t_eval'] = self.t_eval
        with open(csvfile, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quoting=csv.QUOTE_MINIMAL)
            if makeHeader:
                header = [r'#', 'n', 'l', 'm', 'f.mean', 'f.sdev']
                writer.writerow(header)
                bparams = [r'#'] + [str(lbl) + ': ' + str(prm)
                                     for lbl,prm in self.basis.items()]
                writer.writerow(bparams)
            if nlmlist is None:
                nlmlist = self.getNLMlist()
            for nlm in nlmlist:
                f = self.f_nlm[nlm]
                mean, std = f.mean, f.sdev
                newline = [modelName, nlm[0], nlm[1], nlm[2], mean, std]
                writer.writerow(newline)

    @staticmethod
    def readFnlm_csv(csvfile, modelName=None):
        """Reads <f|nlm> coefficients from CSV file."""
        # modelName: only import f(nlm) for entries with modelName==modelName
        # File format (each row): modelName, n,l,m, f.mean, f.sdev
        # all rows that are not commented out with '#,' should follow this format
        data_fnlm = {}
        with open(csvfile, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',',
                                quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                if row[0]=='#':
                    # skip commented lines and the header
                    continue
                mName, str_n, str_l, str_m, f_mean, f_std = row
                nlm = (int(str_n),int(str_l),int(str_m))
                # isheader = (mName=="#")
                # iscomment = False
                # if len(mName) > 0:
                #     iscomment = (mName[0]=="#")
                if (modelName is None or mName==modelName):
                    data_fnlm[nlm] = gvar.gvar(float(f_mean), float(f_std))
        #Doesn't create any self.data object, just returns a fnlm_gvar dict.
        # If there are repeated instances of (nlm), data_fnlm is overwritten with the most recent version.
        return data_fnlm

    def importFnlm_csv(self, csvfile, modelName=None):
        """Imports <f|nlm> coefficients from CSV file, add to f_nlm.

        Note: can be run repeatedly for different files, to add or overwrite
            previous f_nlm values.

        * Caution: Basis.basis dictionary items are saved in commented
            header line in CSV file, but not automatically imported.
        * User needs to ensure that Fnlm input 'bdict' matches CSV file.
        Items self.nMax, ellMax will be updated using self.update_maxes().

        * Caution: modelName input here is different from the hdf5 'model'
            This modelName is an optional string in the first CSV column
            Only rows with the matching modelName are read into Fnlm.
        """
        # reads the file, and overwrites f_nlm with the results
        t0 = time.time()
        data_fnlm = self.readFnlm_csv(csvfile, modelName=modelName)
        for nlm,fdata in data_fnlm.items():
            self.f_nlm[nlm] = fdata
            self.update_maxes(nlm)
        self.t_eval += time.time() - t0

    def importFromProjF(self, projF, nlmlist):
        """Fill in f_nlm from a different ProjectFnlm instance."""
        for nlm in nlmlist:
            self.update_maxes(nlm)
            fnlm = projF.f_nlm[nlm]
            if type(fnlm) is gvar._gvarcore.GVar and not self.use_gvar:
                self.f_nlm[nlm] = fnlm.mean
            else:
                self.f_nlm[nlm] = fnlm






#
