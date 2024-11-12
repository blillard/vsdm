"""An adaptive version of EvaluateFnlm using wavelet extrapolation.

HaarExtrapolate: Extrapolation for a 1d wavelet expansion. Identifies a
    pth-order Taylor series for 'blocks' of adjacent wavelets.

AdaptiveFn: runs the iterative <f|nlm> integration for fixed (lm).
    Step 1. Evaluates <f|bin,lm> for (2**power2) of orthogonal tophat functions.
    Step 2. Checks the precision of the numeric integration
    Step 3. Applies the discrete spherical wavelet transformation to convert
        <f|bin,lm> to the usual <f|nlm>.
    Step 4. Uses HaarExtrapolate to predict the values of the largest 'n'
        wavelets (the latest 'generation'), and compares each prediction
        to the numerical integral result for <f|nlm>. Any 'block' whose
        descendant wavelets are not all precisely predicted is added to the
        list self.blocks_to_refine for further evaluation.
    Step 5. Each block in blocks_to_refine is subdivided, and new generations
        of <f|nlm> are evaluated, until the predictions from HaarExtrapolate
        are sufficiently accurate. The class method refineCompletely() repeats
        this until all blocks have converged.

WaveletFnlm: a EvaluateFnlm object that runs AdaptiveFn for any number
    of (l,m) modes.
"""

__all__ = ['HaarExtrapolate', 'AdaptiveFn', 'WaveletFnlm']


import math
import numpy as np
import gvar # gaussian variables; for vegas
import time

from .basis import f_tophat_to_sphwave
from .projection import EvaluateFnlm
from .utilities import *
from .haar import *
from .portfolio import *



class HaarExtrapolate(Interpolator1d):
    """Applies wavelet extrapolation to a list of evaluated <f_lm|n> coefficients.

    Input:
    f_n_dict: a dict of <f|n> coefficients
    p_order: polynomial order for Taylor series extrapolation
        p_depth: number of wavelet generations needed to calculate Taylor
            series coefficients within each HaarBlock.
    dim: dimensionality of basis functions

    Tracks two HaarString objects:
    self.evaluated: for the tree of evaluated <f|n> coefficients
    self.p_regions: subdivides [0,1] into regions with different Taylor series
        p_regions is a list of HaarBlock headers

    Uses Interpolator1d to provide f(x) for dimensionless x on [0,1].

    Methods:
    get_f_p_list: the list of f(x) and its derivatives at the center of each block,
        f_p_list[i] = [f0, df_1, df_2, ...](x2_i)
    testBlockAccuracy: compares the wavelet extrapolation <f|n> predictions
        with the integrated inner products, self.f_n[n]
    refresh_all: Assuming f_n has been modified, re-calculates 'evaluated'
        and 'p_regions' from scratch.
    __call__(x), inherited from Interpolator1d. Returns interpolated f(x).
    """
    def __init__(self, f_n_dict, p_order=3, dim=3):
        self.f_n = f_n_dict
        self.dim = dim
        self.p_order = p_order
        self.p_depth = int(math.log2(p_order+1))
        # p_depth: number of generations needed for a polynomial Block
        n_list = [n for n in self.f_n.keys()]
        evaluated = hs_n_to_hstr(n_list, inclusive=False)
        self.evaluated = HaarString(hstr=evaluated)
        if self.p_depth==0:
            p_regions = self.evaluated.prevGen(level=0)
        else:
            p_regions = self.evaluated.prevGen(level=self.p_depth-1)
        self.p_regions = HaarString(hstr=p_regions)
        self.p_blocks = {}
        for n in self.p_regions:
            pblock = HaarBlock(n, p_order=self.p_order, dim=self.dim)
            self.p_blocks[n] = pblock
        # make the Interpolator1d:
        x_bounds = self.p_regions.regions_x()
        x0_vals = self.p_regions.midpoints_x()
        f_p_list = self.get_f_p_list()
        Interpolator1d.__init__(self, x_bounds, x0_vals, f_p_list)
        # Get the block-level power "spectrum"
        self.block_power = []
        for n in self.p_regions:
            dE_dn = self.block_power_n(n)
            self.block_power += [dE_dn]

    def extrapolate_n(self, nd):
        # use nearest p_block to estimate <nd|f>, if nd is a descendant of n.
        x2 = haar_x123(nd)[1] # midpoint of nd
        n = self.p_regions.x_to_n(x2) # map x2 to the block n
        if nd > n:
            return self.p_blocks[n].extrapolate_n(self.f_n, nd)
        else:
            print("Can't extrapolate backwards. Using <f|n> from self.f_n...")
            return self.f_n[nd]

    def block_inverse(self):
        """Coarse inverse wavelet transformation, constant over each block."""
        return self.p_regions.sparse_inverse(self.f_n, dim=self.dim, include_hstr=False)

    def f_p_n(self, n):
        """Calculate [f0, df_1, df_2...] vectors in block 'n'.

        'n' can also be an ancestor of any element of p_blocks. This case
            necessitates a new block_inverse() calculation, to get 'f0'.
        """
        if n in self.p_regions:
            ix = self.p_regions.index(n)
            block_inv = self.block_inverse()
            pblock = self.p_blocks[n]
        else:
            n_tree = hs_list_all_n(self.p_regions, include_self=False, assume_complete=False)
            if n not in n_tree:
                print("Error: 'n' not an ancestor of self.p_regions.")
                return
            # else:
            # replace the descendants of 'n' in self.p_regions with n:
            new_hstr = []
            added_n = False
            for nd in self.p_regions.hstr:
                in_block_n = _h_n_covers_nd(n, nd)
                if not _h_n_covers_nd(n, nd):
                    new_hstr += [nd]
                else:
                    if added_n:
                        continue
                    else:
                        new_hstr += [n]
                        added_n = True
            coarser_regions = HaarString(hstr=new_hstr)
            ix = coarser_regions.index(n)
            block_inv = coarser_regions.sparse_inverse(self.f_n, dim=self.dim, include_hstr=False)
            pblock = coarser_regions[n]
        df_p = pblock.df_p(self.f_n)
        f_0_coarse = block_inv[ix]
        f_p = [f_0_coarse+df_p[0]] + df_p[1:]
        return f_p

    def get_f_p_list(self):
        """Calculate [f0, df_1, df_2...] vectors for each block in p_regions."""
        block_inv = self.block_inverse()
        f_p_list = []
        for ix,n in enumerate(self.p_regions):
            pblock = self.p_blocks[n]
            df_p = pblock.df_p(self.f_n)
            f_0_coarse = block_inv[ix]
            f_p = [f_0_coarse+df_p[0]] + df_p[1:]
            f_p_list += [f_p]
        return f_p_list

    def get_fp_DeltaX_list(self):
        """Calculate [f0, df_1 DeltaX, df_2 DeltaX**2, ...].

        Here DeltaX is the width of the block p_blocks[n].
        The Taylor series extrapolation is only expected to converge
        once f0 > df_1 DeltaX > df_2 DeltaX**2 > ... > df_p DeltaX**p.
        """
        block_inv = self.block_inverse()
        f_dX_p_list = []
        for ix,n in enumerate(self.p_regions):
            pblock = self.p_blocks[n]
            DeltaX = haar_x123(n)[2] - haar_x123(n)[0]
            df_p = pblock.df_p(self.f_n)
            f_0_coarse = block_inv[ix]
            f_p = [f_0_coarse+df_p[0]] + df_p[1:]
            f_dX_p = [f_p[k]*DeltaX**k for k in range(len(f_p))]
            f_dX_p_list += [f_dX_p]
        return f_dX_p_list

    def refresh_all(self):
        "Recalculates 'p_regions', 'evaluated', 'p_blocks' with current f_n."
        n_list = [n for n in self.f_n.keys()]
        evaluated = hs_n_to_hstr(n_list, inclusive=False)
        self.evaluated = HaarString(hstr=evaluated)
        p_regions = self.evaluated.prevGen(level=self.p_depth-1)
        self.p_regions = HaarString(hstr=p_regions)
        self.block_power = []
        for n in self.p_regions:
            self.p_blocks[n] = HaarBlock(n, p_order=self.p_order, dim=self.dim)
            # Get the block-level power "spectrum"
            dE_dn = self.power_region(n)
            self.block_power += [dE_dn]
        # Update Interpolator1d objects:
        self.u_bounds = self.p_regions.regions_x()
        self.u0_vals = self.p_regions.midpoints_x()
        self.f_p_list = self.get_f_p_list()
        return

    def addTo_f_n(self, new_f_n):
        "Modify self.f_n and update the block lists and Interpolator1d."
        for n,fn in new_f_n.items():
            self.f_n[n] = fn
        self.refresh_all()
        return

    def testBlockAccuracy(self, n, level=1):
        """Compares the extrapolated and integrated values of <f|n>.

        Returns list of (n, f_integrated, f_extrapolated)."""
        polyblock = HaarBlock(n, p_order=self.p_order, dim=self.dim)
        fn_predicts = polyblock.extrapolate_newgen(self.f_n, level=level)
        all_evaluated = True
        for nd in fn_predicts.keys():
            if nd not in self.f_n.keys():
                all_evaluated = False
        assert all_evaluated, "Missing <f|n> coefficients in f_n: can't test extrapolation accuracy."
        out = []
        for nd,f_ex in fn_predicts.items():
            f_actual = self.f_n[nd]
            # f_diff = f_actual - f_ex
            out += [(nd, f_actual, f_ex)]
        return out

    def block_power_n(self, n):
        """Uses Taylor series to estimate the norm-energy in the n-th block.

        block label: n. Can be in p_regions or its ancestors.
        Integral: int(x**(d-1)dx * f**2), for x1(n)<x<x3(n)."""
        f_p = self.f_p_n(n)
        x1,x2,x3 = haar_x123(n)
        a = x1 - x2
        b = x3 - x2
        d = self.dim
        sum_alpha = 0.
        for alpha in range(d):
            # yes, terminate at alpha = d-1
            fact_a = math.factorial(d-1)/math.factorial(alpha)/math.factorial(d-1-alpha)
            sum_jk = 0.
            for j in range(self.p_order+1):
                # first, the k<j terms:
                for k in range(j):
                    jkda = j+k+d-alpha
                    jkfact = f_p[j]*f_p[k]/(math.factorial(j)*math.factorial(k))
                    sum_jk += 2 * jkfact * (b**jkda - a**jkda)/jkda
                # last, the j=k term:
                j2da = 2*j+d-alpha
                jfact = f_p[j]/math.factorial(j)
                sum_jk += jfact**2 * (b**j2da - a**j2da)/j2da
            # double sum complete. Add to sum over alpha.
            sum_alpha += fact_a * x2**alpha * sum_jk
        return sum_alpha

    def norm_fx2(self):
        """Uses Taylor series to estimate the norm-energy.

        Integral: int(x**(d-1)dx * f**2), for 0<x<1."""
        e_f2 = 0.
        for n in self.p_regions:
            e_f2 += self.block_power_n(n)
        return e_f2

    def f_x(self, x):
        """Inherited from Interpolator1d."""
        return self.fU(x)


class AdaptiveFn(HaarExtrapolate,EvaluateFnlm):
    """Adaptive version of EvaluateFnlm, for fixed (l,m).

    Input parameters:
    - lm: fixed (l,m) for this part of the calculation
    - power2: sets the initial radial grid to include 2**power2 segments.
    - p_order=1,3,7,...: order of the extrapolation.
    - epsilon: relative accuracy goal for the <f_lm|n> expansion.
        * for the largest f_lm harmonic modes, epsilon should match the
          global precision goal.
        * for smaller f_lm, epsilon can be less precise.
    - atol_f2norm: alternative maximum for epsilon, based on the
        total distributional energy (f^2 norm).
    - integ_params: integration parameters, including:
        * method='gquad' or 'vegas' (recommend 'gquad')
        * verbose=True to print intermediate results

    Objects:
    - self.hstr: the HaarString tracking the evaluated <f_lm|n> coefficients
    - self.f_n: the usual list of <f|nlm>, with lm given by self.lm.
        Note: self.f_nlm from EvaluateFnlm is redundant with self.f_n.

    Class inheritance:
    * both EvaluateFnlm and HaarExtrapolate have __call_ and fU() methods, which
        conflict. HaarExtrapolate is given priority: its fU(x) is a function of
        1d variable x = u/u0. It is used to define self.flm_u().
    * the inherited HaarString.subdivideAt() updates self.hstr.
        This method is only used in conjunction with EvaluateFnlm.updateFnlm(),
        which updates self.f_nlm.

    Methods:
    * coarsegridInitialization: evaluates all tophat integrals <f|n'>, then
        finds wavelet coefficients <f|n> from discrete wavelet transformation
    * diagnose_convergence: calculates f', f'', f''',... derivatives and
        tests whether f' > f'' > f''' > .... If not, then may need to re-run
        coarsegridInitialization with a finer grid.
    * check_extrap_accuracy: tests absolute accuracy of extrapolated <f|n>
        coefficients.
    * evaluateBatchNextGen: calculates new wavelet coefficients <f|n> in the
        next generation, if needed
    * refineCompletely: iterate evaluateBatchNextGen until complete convergence
    * flm_u(u): returns interpolated value of f_lm(u) for u = x*u0.

    Evaluates higher-n wavelet <f|nlm> coefficients for fixed (l,m)
    using the following process:
    (1) use integration to find first nMax ~ 2**lambdaStar coefficients
    (2) using the first nMax/2 coefficients, estimate values of n > nMax/2
        from cubic extrapolation
    (3) In intervals where cubic extrapolation does NOT accurately predict <f|nlm>:
        (a) evaluate the descendant wavelets, (n) --> (2n), (2n+1)
        (b) compare accuracy of next-gen cubic extrapolation
        (c) if not sufficiently accurate, subdivide again and repeat
    (4) End, once every branch meets one of two conditions, for all
        of the four descendant wavelets in generation lambda0+2:
        (a) the integrated and extrapolated versions of <f|nlm> match,
            within |integrated - extrapolated| < epsilon [absolute accuracy]
        (b) the integrated value of <f|nlm> is smaller than epsilon.
    The end result is a tree of evaluated wavelet coefficients, which can be
    extrapolated to fill out all the nonzero (|<f|nlm>| > epsilon) coefficients.

    PRECISION.
    Keeping three (3) types of global precision:
    (1) epsilon: relative tolerance for all numerical integrals
    (2) atol_f2norm: the threshold for dropping contributions to sum_nlm <f|nlm>**2
    (3) atol_fnlm: the threshold for dropping individual <f|nlm> terms.

    Relatedly, integ_params includes related tolerances for the numeric
        integration: respectively,
    (1) rtol_e and rtol_f (relative tolerance for f**2 and <f|nlm> integrals)
    (2) atol_e: absolute tolerance for integrals of (f(u))**2
    (3) atol_f: abs.tolerance for <f|nlm> inner products.
    In each case, 'tol' refers to the precision goal for the 1d integrals in
        gquad. The actual tolerance (for 2d, 3d) is expected to be larger.
    Current rule of thumb:

    Precision goal: recover f = sum_nlm <nlm|f>|nlm> up to one part in 1/epsilon
    Use distEnergy as a proxy for global error.
    - need absolute integration accuracy to be better than epsilon
    - also need to control truncation error
    Relative to benchmark f_0 = sqrt(distEnergy/u0**3)

    Truncation error:
    * must at least include all coefficients |<f|nlm>| > epsilon
    If there are N_edge coefficients with <f|nlm> <~ epsilon, then
        expect sum_nlm <f|nlm> error on the order of sqrt(N_edge) * epsilon.
        (Assuming fast-dropping <f|nlm> values at higher n.)
    For 3d (nlm) space, N_edge ~ (N_tot)**(2/3). For fixed (lm), N_edge~N/2
    --> suggest epsilon_trunc < 0.05*epsilon
        i.e. set <f|nlm> -> 0 if |<f|nlm>| < epsilon_trunc * f_0

    Integration error:
    * need absolute tolerance atol < epsilon_trunc (e.g. 0.01*epsilon)
    * atol applies to VEGAS integration and also the extrapolation method
    Expected total error in f (if uncorrelated): sqrt(N_tot) * atol.
        expected error in distEnergy: sqrt(N_primary) * 2*atol,
        from [large <f|nlm>]**2 * (1 +/- atol)**2
    * atol is relative to the expected total size of |f|:
        Delta(<f|nlm>) < atol * f_0

    Relative integration error:
    * to keep relative error (for individual coefficients) below some
    threshold, e.g. 1%, even for coefficients <f|nlm> ~ atol f_0


    """
    def __init__(self, basis, fSph, lm, integ_params, power2=5, p_order=3,
                 import_fn=None, epsilon=1e-4, atol_f2norm=None, atol_fnlm=None,
                 f_type=None, csvsave_name=None, use_gvar=True):
        t0 = time.time()
        self.lm = lm
        self.p_depth = int(math.log2(p_order+1))
        if self.p_depth > power2:
            power2 = self.p_depth # minimum N of evaluated wavelet coefficients
        # self.p_order = p_order
        self.verbose = True
        # self.use_gvar = use_gvar
        if 'verbose' in integ_params:
            self.verbose = integ_params['verbose']
        # integ_params['method'] = 'gquad' # need 'atol' and 'rtol' from scipy version

        """Integration parameters: tol_e for norm-energy, tol_f for <f|nlm> integrals.

        rtol_e and rtol_f are always defined: either directly from integ_params,
            or using epsilon.
        If atol_e or atol_f is not defined, then NIntegrate only uses rtol.

        Class parameters atol_f2norm, atol_fnlm determine when to terminate the
            expansion in <f|nlm>. Their counterparts in integ_params allow for
            higher precision during numerical integration.
        """
        if 'method' not in integ_params:
            integ_params['method'] = 'gquad'
        assert integ_params['method'] == 'gquad', "Vegas integration not currently supported in adaptive.py."
        if 'rtol_e' not in integ_params:
            integ_params['rtol_e'] = epsilon
        if 'rtol_f' not in integ_params:
            integ_params['rtol_f'] = epsilon
        if 'atol_e' not in integ_params:
            if atol_f2norm is not None:
                integ_params['atol_e'] = 0.05 * atol_f2norm
            else:
                integ_params['atol_e'] = None
        if 'atol_f' not in integ_params:
            if atol_fnlm is not None:
                integ_params['atol_f'] = 0.05 * atol_fnlm
            else:
                integ_params['atol_f'] = None

        self.epsilon = epsilon
        if atol_fnlm is not None:
            self.atol_fnlm = atol_fnlm
        else:
            self.atol_fnlm = integ_params['atol_f']
        if atol_f2norm is not None:
            self.atol_f2norm = atol_f2norm
        else:
            self.atol_f2norm = integ_params['atol_e']
        self.integ_f = integ_params
        self.integ_f['atol'] = integ_params['atol_f']
        self.integ_f['rtol'] = integ_params['rtol_f']
        self.integ_e = integ_params
        self.integ_e['atol'] = integ_params['atol_e']
        self.integ_e['rtol'] = integ_params['rtol_e']

        assert('u0' in basis), "Missing mandatory parameter: 'u0'."
        u0 = basis['u0']
        assert('type' in basis), "Missing mandatory parameter: 'type'."
        uType = basis['type']
        assert('uMax' in basis), "Missing mandatory parameter: 'uMax'."
        uMax = basis['uMax']

        EvaluateFnlm.__init__(self, basis, fSph, self.integ_f, nlmlist=[],
                              f_type=f_type, csvsave_name=csvsave_name,
                              use_gvar=use_gvar)

        """Step 0. If import_fn is not None, check whether it has enough entries
        to skip Steps 1-3.
        """
        nMax = 2**power2 - 1
        f_n = {}
        n_list = [n for n in range(nMax+1)]
        if all([n in import_fn.keys() for n in n_list]):
            for n,fn in import_fn.items():
                f_n[n] = fn
                self.f_nlm[(n,lm[0],lm[1])] = fn
        else:
            gridb = dict(u0=u0, uMax=uMax, type='tophat')
            uiList = [uMax * n/2**power2 for n in range(2**power2+1)]
            # Note: nth bin has base of support (uiList[n], uiList[n+1])
            gridb['uiList'] = uiList
            gridb['nMax'] = nMax
            """Steps 1-3: """
            f_n = self.coarsegridInitialization(gridb)


        # init HaarExtrapolate after EvaluateFnlm: dim is defined in basis.
        HaarExtrapolate.__init__(self, f_n, p_order=p_order, dim=self.dim)
        # Now self.f_n, self.p_regions, etc are defined.

        """Step 4. Check accuracy of cubic extrapolation method."""
        self.blocks_to_refine = self.check_extrap_accuracy(verbose=self.verbose)

        """At this stage, p_regions lists the narrowest possible wavelet blocks.
        None of the descendant <f|nlm> have been tested directly, but those
        descending from self.blocks_to_refine are assumed to be inaccurate.

        At the end of initialization, the angular power P_lm for this harmonic
        mode can be estimated from self.f2_norm().

        Step 5. Evaluate <f|nlm> for the descendants of blocks_to_refine.
        - Use evaluateBlockNextGen(n) on all n in blocks_to_refine.
            Repeat until blocks_to_refine is empty.
        """

    def coarsegridInitialization(self, gridb):
        """Step 1. Evaluate <f|bin,lm> on 2**power2 equal-width bins."""
        nMax = gridb['nMax']
        nlmlist = [(n, self.lm[0], self.lm[1]) for n in range(nMax+1)]
        if self.verbose:
            print("Calculating <f|nlm> for (l,m)={} on {} bins".format(self.lm,nMax+1))
            print("\t atol={}, rtol={}".format(self.integ_f['atol'],
                                               self.integ_f['rtol']))
        coarseGrid = EvaluateFnlm(gridb, self.fSph, self.integ_f, nlmlist=nlmlist,
                                  f_type=self.f_type,
                                  csvsave_name=None,
                                  use_gvar=self.use_gvar)
        self.tGrid = coarseGrid.t_eval
        if self.verbose:
            print("Integration time for coarse grid:", self.tGrid)

        """Step 2. Use estimate for |<f|nlm>|^2 norm to check precision goals."""
        # calculate the initial norm-energy for this set
        power_lm_init = coarseGrid.f2_norm()
        if self.verbose:
            print("integration time: {}".format(self.tGrid))
            print("Power_lm = {}".format(power_lm_init))
            if self.use_gvar:
                delta_Plm = power_lm_init.sdev
                rdelta_P = delta_Plm/power_lm_init.mean
                print("\tuncertainty: {}".format(delta_Plm))
                print("\trelative error: {}".format(rdelta_P))
                atol_e = self.atol_f2norm
                if atol_e is not None and atol_e>delta_Plm:
                    print("Warning: delta_Plm is larger than atol_e={}".format(atol_e))
                if rdelta_P>self.epsilon:
                    print("Warning: relative error is larger than epsilon={}".format(self.epsilon))

        """Step 3. Convert <f|bin> to wavelet coefficients."""
        grid_flmn = [coarseGrid.f_nlm[(n, self.lm[0], self.lm[1])]
                     for n in range(nMax+1)]
        wave_flmn = f_tophat_to_sphwave(grid_flmn) #ordered n=0,1,2,...nMax
        f_n = {} # 1d version of f_nlm, with lm fixed, for use with HaarBlock extrapolation
        for nlm in nlmlist:
            (n,l,m) = nlm
            self.f_nlm[(nlm)] = wave_flmn[n]
            f_n[n] = wave_flmn[n] #for HaarExtrapolate
        if self.csvsave_name is not None:
            self.writeFnlm_csv(self.csvsave_name, nlmlist=nlmlist)
        return f_n

    def evaluateBlockNextGen(self, n, overwrite=False):
        """Evaluate all <f|n> in the next generation under block 'n'.

        If overwrite==False and <f|nlm> is already in self.f_n, then the
        saved value of f_n is used. (No need to run updateFnlm.)
        This is most likely to happen when using import_fn to read in self.f_nlm
        values, either from another EvaluateFnlm instance or from a CSV file.

        Checks the extrapolation accuracy, and updates self.blocks_to_refine.
        Updates the self.p_regions and self.evaluated HaarStrings
        """
        pblock = HaarBlock(n, p_order=self.p_order, dim=self.dim)
        fn_predicts = pblock.extrapolate_newgen(self.f_n, level=1)
        all_good = True
        for nd,f_nd in fn_predicts.items():
            if self.verbose:
                print("Block {}:".format(n))
            t0 = time.time()
            nlm = (nd, self.lm[0], self.lm[1])
            if overwrite or nd not in self.f_n.keys():
                fnlm = self.updateFnlm(nlm, self.integ_f,
                                       csvsave_name=self.csvsave_name)
                self.f_n[nd] = fnlm
            else:
                fnlm = self.f_n[nd]
            f_diff = fnlm - f_nd
            if self.verbose:
                print("n={}\tf_actual: {} \tf_extp: {}".format(nd, fnlm, f_nd))
                print("\tf_diff: {}".format(f_diff))
                print("\tIntegration time: {}".format(time.time()-t0))
            if self.use_gvar:
                a_error = math.fabs(f_diff.mean)
            else:
                a_error = math.fabs(f_diff)
            if self.atol_fnlm is not None and a_error > self.atol_fnlm:
                all_good = False
                if self.use_gvar and a_error < f_diff.sdev:
                    all_good = True
                    print("Warning: f.sdev is larger than atol_f. Need to make integ_params['atol'] more precise.")
        self.blocks_to_refine.remove(n)
        if not all_good:
            self.blocks_to_refine += [2*n, 2*n+1]
        # Update the hstrings:
        self.p_regions.subdivideAt(n, level=1)
        # Update the list of evaluated <f|n>:
        for ne in pblock:
            self.evaluated.subdivideAt(ne, level=1) #
        return all_good

    def evaluateBatchNextGen(self, overwrite=False):
        """Evaluate all <f|n> descendants of blocks_to_refine, in series.

        Checks the extrapolation accuracy, and updates self.blocks_to_refine."""
        t0 = time.time()
        nList = [n for n in self.blocks_to_refine]
        nBlocks = len(nList)
        if self.verbose:
            print("Evaluating {} blocks...".format(nBlocks))
        all_blocks_good = True
        for n in nList:
            is_good = self.evaluateBlockNextGen(n, overwrite=overwrite)
            if not is_good:
                all_blocks_good = False
        if self.verbose:
            print("Average integration time per block: {}".format((time.time()-t0)/nBlocks))
            print("Blocks needing further refinement: {}".format(len(self.blocks_to_refine)))
        return all_blocks_good

    def refineCompletely(self, max_depth=5, overwrite=False):
        """Evaluates <f|n> until reaching convergence everywhere on [0,1].

        Terminates once self.blocks_to_refine = [] is empty,
        or after calculating <f|n> for 'max_depth' many new generations.
        """
        gen_counter = 0
        all_blocks_good = True
        while gen_counter < max_depth and len(self.blocks_to_refine) > 0:
            all_blocks_good = self.evaluateBatchNextGen(overwrite=overwrite)
            gen_counter += 1
        return all_blocks_good

    def diagnose_convergence(self, verbose=True):
        """Test f0 > df_1 DeltaX > df_2 DeltaX**2 > ... convergence.

        DeltaX is the full width of the block 'n'.
        Returns two lists of block_n values, 'converging' and 'diverging'.
        Both lists have entries of the form [(n, nGensExpected)], where:
            nGensExpected = log2(df_p DeltaX**p / f0) * 1/p.
        This is the expected number of new generations in <f|n> that must be
        evaluated before this block's dfDx_p will be ordered.
        For converging blocks it is negative, otherwise it is usually positive.
        Note: a block is added to 'diverging' UNLESS its dfDx_p is
            monotonically decreasing.
        """
        if self.p_order==0:
            print("This method requires p_order > 0.")
            return
        dfDx_p_list = self.get_fp_DeltaX_list()
        converging = []
        diverging = []
        for ix,dfDx_p_in in enumerate(dfDx_p_list):
            if type(dfDx_p_in[0]) is gvar._gvarcore.GVar:
                dfDx_p = [df.mean for df in dfDx_p_in]
            else:
                dfDx_p = [df for df in dfDx_p_in]
            n = self.p_regions[ix]
            if dfDx_p[0]==0 or dfDx_p[-1]==0:
                gensToGo = 0
            else:
                gensToGo = math.log2(math.fabs(dfDx_p[-1]/dfDx_p[0])) / self.p_order
            decreasing = (gensToGo < 0)
            if decreasing:
                converging += [(n, gensToGo)]
            else:
                diverging += [(n, gensToGo)]
        if verbose:
            print("Converging blocks:")
            print("  n\tGenerations to go until convergence:")
            for n,gensToGo in converging:
                print("  {}\t{}".format(n,gensToGo))
            print("Diverging blocks:")
            print("  n\tGenerations to go until convergence:")
            for n,gensToGo in diverging:
                print("  {}\t{}".format(n,gensToGo))
        return converging, diverging

    def check_extrap_accuracy(self, verbose=True):
        if self.atol_fnlm is not None and verbose:
            print("Accuracy goal: atol_fnlm =", self.atol_fnlm)
        blocks_to_refine = [] # tracks blocks with inaccurate wavelet extrapolation
        for n in self.p_regions.prevGen(level=1): # back up one level
            n_fint_fext = self.testBlockAccuracy(n, level=1)
            all_good = True
            for nd,f_actual,f_ex in n_fint_fext:
                f_diff = f_actual - f_ex
                if verbose:
                    print("n={}\tf_actual: {:.6g} \tf_est: {:.6g}".format(nd, f_actual, f_ex))
                    print("\tf_diff: {:.8g}".format(f_diff))
                if self.use_gvar:
                    a_error = math.fabs(f_diff.mean)
                else:
                    a_error = math.fabs(f_diff)
                if self.atol_fnlm is not None and a_error > self.atol_fnlm:
                    all_good = False
                    if self.use_gvar and a_error < f_diff.sdev:
                        all_good = True
                        print("Warning: f.sdev is larger than atol_fnlm. Need to make atol_f more precise.")
            if not all_good:
                blocks_to_refine += [2*n, 2*n+1]
                if verbose:
                    print("Extrapolation from block n={} insufficiently accurate.".format(n))
        return blocks_to_refine

    def flm_u(self, u):
        """The interpolated radial function <f|lm>(u), from Interpolator1d."""
        x = u/self.u0
        return self.f_x(x)

    def __call__(self, u):
        return self.flm_u(u)

class WaveletFnlm(EvaluateFnlm, Interpolator3d):
    """EvaluateFnlm, augmented with wavelet extrapolation methods.

    Applies AdaptiveFn to each spherical harmonic mode, starting with an
        initial list that can subsequently be expanded.
    Inherits EvaluateFnlm to keep a centralized self.f_nlm coefficient list

    Arguments:
    - EvaluateFnlm parameters. The intermediate steps (AdaptiveFn) will use gvar,
        but the high-level EvaluateFnlm won't if use_gvar=False.
    - AdaptiveFn parameters: p_order, epsilon, atol_f2norm.
    - power2_lm: dict of lm modes to initialize by default, potentially
        with different values of power2 for each one.
    - import_fnlm: can read in previously evaluated <f|nlm> coefficients,
        to be passed to AdaptiveFn.
    """
    def __init__(self, basis, fSph, integ_params, power2_lm={}, p_order=3,
                 max_depth=5, refine_at_init=False, import_fnlm=None,
                 epsilon=1e-6, atol_f2norm=None, atol_fnlm=None,
                 f_type=None, csvsave_name=None, use_gvar=True):
        self.p_order = p_order
        self.epsilon = epsilon
        self.atol_f2norm = atol_f2norm
        self.atol_fnlm = atol_fnlm
        self.max_depth = max_depth # for refineCompletely() cutoff.
        self.converged_lm = {}
        Interpolator3d.__init__(self, {}) #self.fI_lm is empty at init.
        # Initialize the high-level EvaluateFnlm object, with empty nlmlist:
        EvaluateFnlm.__init__(self, basis, fSph, integ_params, nlmlist=[],
                              f_type=f_type, csvsave_name=csvsave_name,
                              use_gvar=use_gvar)
        if import_fnlm is not None:
            for nlm,f in import_fnlm.items():
                if type(f) is gvar._gvarcore.GVar and not use_gvar:
                    fnlm = f.mean
                else:
                    fnlm = f
                self.f_nlm[nlm] = fnlm
            if csvsave_name is not None:
                self.writeFnlm_csv(csvsave_name,
                                   nlmlist=[nlm for nlm in import_fnlm.keys()])
        # Flm_n = {} #placeholding for saving AdaptiveFn objects.
        for lm,power2 in power2_lm.items():
            power_lm = self.initialize_lm(lm, power2)
            if refine_at_init and power_lm > 0.05*atol_f2norm:
                self.refine_lm(lm, max_depth=max_depth)
        # saves Flm_n as self.fI_lm

    def initialize_lm(self, lm, power2):
        "Run AdaptiveFn.__init__ for this lm. Returns the P_lm power."
        # First, check to see if enough <f|nlm> are already evaluated.
        import_fn = {}
        for nlm,fnlm in self.f_nlm.items():
            (n,l,m) = nlm
            if (l,m)==lm and n<2**power2:
                import_fn[n] = fnlm
        Flm_n = AdaptiveFn(self.basis, self.fSph, lm, self.integ_params,
                           power2=power2, p_order=self.p_order,
                           import_fn=import_fn, use_gvar=self.use_gvar,
                           epsilon=self.epsilon,
                           atol_f2norm=self.atol_f2norm,
                           atol_fnlm=self.atol_fnlm,
                           f_type=self.f_type, csvsave_name=self.csvsave_name)
        self.fI_lm[lm] = Flm_n
        for nlm,fnlm in Flm_n.f_nlm.items():
            self.f_nlm[nlm] = fnlm
        if len(Flm_n.blocks_to_refine)==0:
            self.converged_lm[lm] = True
        else:
            self.converged_lm[lm] = False
        self.t_eval += Flm_n.t_eval
        if self.p_order==0:
            return Flm_n.norm_fx2()
        # print prediction for convergence of polynomial extrapolation:
        if self.verbose:
            converging,diverging = Flm_n.diagnose_convergence(verbose=self.verbose)
            ngensLeft = np.array([divg[1] for divg in diverging])
            convergedFraction = len(converging)/(len(converging)+len(diverging))
            print("Converged fraction: {:.3g}%".format(convergedFraction*100))
            if convergedFraction < 0.5:
                print("! Most blocks have unconverged Taylor series.")
                print("Recommend re-initializing with finer grid spacing.")
            if len(diverging) > 0:
                print("Average N_gens until convergence: {}".format(ngensLeft.mean()))
                print("Maximum N_gens until convergence: {}".format(ngensLeft.max()))
            else:
                print("All blocks have converging Taylor series.")
        return Flm_n.norm_fx2()

    def refine_lm(self, lm, max_depth=None):
        t0 = time.time()
        if max_depth is None:
            max_depth = self.max_depth
        all_good = self.fI_lm[lm].refineCompletely(max_depth=max_depth)
        for nlm,fnlm in self.fI_lm[lm].f_nlm.items():
            self.f_nlm[nlm] = fnlm
        if all_good:
            self.converged_lm[lm] = True
            if self.verbose:
                print("{}: refined completely.".format(lm))
        else:
            self.converged_lm[lm] = False
            if self.verbose:
                print("{}: wavelet extrapolation not yet converged.".format(lm))
        self.t_eval += time.time() - t0

    def diagnose_convergence(self, lm, verbose=True):
        if lm in self.fI_lm.keys():
            self.fI_lm[lm].diagnose_convergence(verbose=verbose)
        else:
            print("Error: lm not in self.fI_lm.")
        return

    def blocks_to_refine(self, lm):
        return self.fI_lm[lm].blocks_to_refine

    def check_extrap_accuracy(self, lm, verbose=True):
        return self.fI_lm[lm].check_extrap_accuracy(verbose=verbose)

    def update_atol(self, atol_fnlm=None, atol_f2norm=None):
        if atol_fnlm is not None:
            self.atol_fnlm = atol_fnlm
            for lm in self.fI_lm.keys():
                self.fI_lm[lm].atol_fnlm = atol_fnlm
        if atol_f2norm is not None:
            self.atol_f2norm = atol_f2norm
            for lm in self.fI_lm.keys():
                self.fI_lm[lm].atol_f2norm = atol_f2norm

#
