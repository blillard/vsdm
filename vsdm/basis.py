"""VSDM: orthogonal basis functions for class Basis().

Functions and classes:
    ylm_real: real spherical harmonics. Uses plm_real
    lag_spherical: spherical Laguerre functions
    haar_sph: spherical wavelet functions
    class Basis: defines an orthogonal basis of functions
        methods for <f|nlm> integration, f(r) = sum_nlm(<nlm|f>|nlm>) summation

Functions here are all dimensionless: no need for .units
"""

__all__ = ['Basis', 'lag_spherical', 'cubic_sph_haar_x',
           'haar_sph', '_haar_sph_value', '_tophat_value']
           #'_hindex_LM', '_hindex_n'

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
# import h5py # database format

from .utilities import *


"""
    Spherical Laguerre functions:
"""

def _lag_half_ln(n, ell, z):
    return spf.genlaguerre(n, 1/2+ell, monic=False)(z)

def lag_spherical(n, ell, x):
    """Normalized spherical Laguerre function"""
    #includes exponential factor! Dimensionless x
    factor = math.sqrt(math.factorial(n) * 2**(2.5+ell) / spf.gamma(n+1.5+ell))
    return factor * math.exp(-x**2) * x**ell * _lag_half_ln(n, ell, 2*x**2)

"""
    Spherical wavelet functions:
"""

#Defining an index for the haar wavelets: n = 2**L + M (n=0 for L=-1)
def _hindex_n(L, M):
    if L==-1:
        return 0
    else:
        return 2**L + M

def _hindex_LM(n):
    if n==0:
        return [-1, 0]
    else:
        L = int(math.log2(n))
        M = n - 2**L
        return [L, M]

def _haar_sph_value(n):
    """Returns the value of h_n(x) where it is nonzero."""
    if n==0:
        return math.sqrt(3)
    else:
        [L,M] = _hindex_LM(n)
        x1 = 2**(-L) * M
        x2 = 2**(-L) * (M+0.5)
        x3 = 2**(-L) * (M+1)
        y1 = x1**3
        y2 = x2**3
        y3 = x3**3
        A = math.sqrt(3/(y3 - y1) * (y3-y2)/(y2-y1))
        B = math.sqrt(3/(y3 - y1) * (y2-y1)/(y3-y2))
        return [A,-B]

def haar_sph(n, x):
    """Normalized spherical Haar wavelet, n=0,1,2,..."""
    if n==0:
        if 0 <= x <= 1:
            return _haar_sph_value(n)
        else:
            return 0
    else:
        [L,M] = _hindex_LM(n)
        x1 = 2**(-L) * M
        x2 = 2**(-L) * (M+0.5)
        x3 = 2**(-L) * (M+1)
        if x1 <= x < x2:
            return _haar_sph_value(n)[0]
        elif x2 < x <= x3:
            return _haar_sph_value(n)[1]
        elif x==x2:
            return 0.5*(_haar_sph_value(n)[0] + _haar_sph_value(n)[1])
        else:
            return 0

def _tophat_value(x1, x2):
    """Returns the value of the tophat function where it is nonzero.

    i.e. for x1 < x < x2. Assuming x1 != x2, and that both are nonnegative.
    Normalized for int(x^2 dx * value**2, {x, x1, x2}) = 1.
    """
    return math.sqrt(3/(x2**3 - x1**3))


def _Hp_sph_n(p, n):
    "H_p function for spherical wavelet extrapolation"
    A, mB = _haar_sph_value(n)
    B = - mB
    L,M = _hindex_LM(n)
    # x1 = 2**(-L) * M
    # x2_Delta = (M+0.5)
    # x3 = 2**(-L) * (M+1)
    Delta = 2**(-L)
    term0 = ((-1)**p*A - B)/(p+3)
    term1 = -4*(M+0.5) * ((-1)**p*A + B)/(p+2)
    term2 = 4*(M+0.5)**2 * ((-1)**p*A - B)/(p+1)
    return Delta**3/8 * (term0 + term1 + term2)


def cubic_sph_haar_x(df_123, n):
    "Predicts <f|n> for radial function f(0<x<1) given df/dx derivatives at x2"
    # (df1, df2, df3) = df_123
    L,M = _hindex_LM(n)
    Delta = 2**(-L)
    sum = 0.0
    for p in [1,2,3]:
        Fp = (0.5*Delta)**p / math.factorial(p) * df_123[p-1]
        sum += Fp * _Hp_sph_n(p, n)
    return sum



class Basis():
    """Defines a set of (nlm) basis functions for any type.

    Arguments:
        basis: a dictionary including the following: (mandatory)
        * type = 'laguerre', 'wavelet', 'tophat'
        * u0: sets normalization of |nlm>
            is the scale factor for infinite-domain radial functions
        conditional/optional items:
        * uiList: mandatory for 'tophat', the list of u_i to be used
        * uMax: mandatory for 'laguerre', setting the limit of integration on u.
            For 'wavelet' and 'tophat', can set uMax=None.
        * dim = 3,4: if ==4, adds generation of (etype,e0,eiList,eMax)
            Not strictly mandatory. Defaults to dim=3 if missing.
    Methods and variables:
        .basis: dict with (type, u0, uMax, uiList, etc.)
        radRn(n, ell,u): radial function r^{ell}_n(u), with dimensionful u
            Passes dimensionless u/u0 to appropriate basis function
        phiNLM(nlm,uvec): basis function (n,l,m), for uvec=[r, theta, phi]
            Normalized: integral(d3r/u0**3 * phi(nlm) * phi(nlm)) = 1
        getFnlm(f, nlm): evaluates <f|nlm> for function f(uvec)
        get1nlm(nlm): returns <1|nlm>, used in comparisons with L1 normalized gX
        nlmAssembleFu(f_nlm, uvec): returns an approximation of f(uvec),
            from sum of <f|nlm> |nlm> over all (nlm)
    """
    # For now, not including spherical bessel functions.
    # For j_ell, need to initialize list of zeros, alpha_{l,n}
    def __init__(self, bdict):
        """Makes self.basis, self.u0, """
        self.basis = bdict
        update_dict = self.runBasisCheck(bdict)
        for lbl,entry in update_dict.items():
            self.basis[lbl] = entry
        self.uMax = self.basis['uMax']

    def runBasisCheck(self, bdict):
        """Ensures mandatory items are included, generates class variables.

        Arguments:
            bdict: dictionary of basis parameters
        Returns: dict 'updates' of entries in self.basis that should be updated,
                e.g. to define 'uMax' from 'u0' if appropriate
        """
        updates = {}
        assert('u0' in bdict), "Missing mandatory parameter: 'u0'."
        self.u0 = bdict['u0']
        assert('type' in bdict), "Missing mandatory parameter: 'type'."
        uType = bdict['type']
        if uType=="wavelet":
            assert('uMax' in bdict), "Missing mandatory parameter: 'uMax'."
            # else:
            #     assert(self.u0 == bdict['uMax']), "wavelet: require u0=uMax."
        elif uType=='tophat':
            assert('uiList' in bdict), "tophat: need uiList."
            #uiList should be ordered and nonnegative
            self.uiList = bdict['uiList']
            assert(all(self.uiList[i] < self.uiList[i+1]
                       for i in range(len(self.uiList) - 1))), "tophat: uiList must be ordered"
            assert(self.uiList[0] >= 0), "tophat: all u_i must be nonnegative"
            if 'uMax' not in bdict:
                updates['uMax'] = bdict['uiList'][-1]
            else:
                assert(bdict['uMax'] == bdict['uiList'][-1]), "tophat: require uMax=uiList[-1]."
        elif uType=="laguerre":
            assert('uMax' in bdict), "Missing mandatory parameter: 'uMax'."
        return updates

    def radRn(self, n, ell, u):
        """Dimensionless radial function r_n^(ell)(u).

        Normalized: integral(u**2 du/u0**3 radRn**2) = 1,
            in limit where uMax=infinity
        ell: specifies type of function for Laguerre, Bessel
        n: index n=0,1,2,...nMax.
            For 'tophat', counting starts at n=0,1,2,3,...nMax-1.
        """
        x = u/self.u0
        if self.basis['type']=="wavelet":
            return haar_sph(n, x)
        elif self.basis['type']=="laguerre":
            return lag_spherical(n, ell, x)
        elif self.basis['type']=="tophat":
            #x_n = u_n / u0
            x_n,x_np1 = self.uiList[n]/self.u0, self.uiList[n+1]/self.u0
            if x_n < x < x_np1:
                return _tophat_value(x_n, x_np1)
                # return _tophat_value(x_nm1, x_n)
            else:
                return 0


    def _baseOfSupport_n(self, n, getMidpoint=False):
        """Returns the range in u for which radRn(u) is nonzero

        For wavelets, also provides the mid-point if (getMidpoint)
            (Used for analytic MathcalI.)
        When used to provide integration range for Vegas,
            keep getMidpoint=False.
        """
        if self.basis['type']=='tophat':
            return [self.uiList[n], self.uiList[n+1]]
        elif self.basis['type']=='wavelet' and getMidpoint:
            if n==0:
                return [0, self.uMax, self.uMax]
            else:
                [lam, mu] = _hindex_LM(n)
                xA = 2**(-lam) * mu
                xM = 2**(-lam) * (mu+0.5)
                xB = 2**(-lam) * (mu+1)
                return [xA*self.u0, xM*self.u0, xB*self.u0]
        elif self.basis['type']=='wavelet' and not getMidpoint:
            if n==0:
                return [0, self.uMax]
            else:
                [lam, mu] = _hindex_LM(n)
                xA = 2**(-lam) * mu
                xB = 2**(-lam) * (mu+1)
                return [xA*self.u0, xB*self.u0]
        else:
            return [0, self.uMax]

    def _volume_fraction_n(self, n, dim=3):
        """Volume fraction occupied by nth basis function.

        Used to adjust VEGAS neval points for (dim) dimensional integrals,
            to sample all basis functions with the same density of points.
        """
        [uA, uB] = self._baseOfSupport_n(n, getMidpoint=False)
        (xA, xB) = (uA/self.uMax, uB/self.uMax)
        return xB**dim - xA**dim

    def phiNLM(self, nlm, uvec):
        """Dimensionless basis function |nlm> = r_n(r)*Y_lm(theta,phi).

        Normalized: integral(r**2 dr dOmega / u0**3 * phiNLM**2) = 1.
        """
        (n, ell, m) = nlm # index: a 3-tuple
        [r, theta, phi] = uvec # spherical coordinate: a vector
        return self.radRn(n, ell, r) * ylm_real(ell, m, theta, phi)


    def get1nlm(self, n):
        """Returns <1|nlm> inner product, for L1 integrals."""
        # vanishes for all l > 0
        if self.basis['type']=="laguerre":
            return (-1)**n * (math.sqrt(math.pi * 2**2.5 / math.factorial(n))
                              * math.sqrt(spf.gamma(n + 1.5)))
        elif self.basis['type']=="wavelet":
            if n==0:
                return math.sqrt(4*math.pi/3)
            else:
                return 0
        elif self.basis['type']=="tophat":
            x_n,x_np1 = self.uiList[n]/self.u0, self.uiList[n+1]/self.u0
            return math.sqrt(4*math.pi/3 * (x_np1**3 - x_n**3))

    def doVegas(self, integrand, volume, vegas_params, printheader=None):
        "Tool for performing VEGAS integrals using vegas_params dictionary."
        # Unpack VEGAS parameters
        nitn_init = vegas_params['nitn_init']
        nitn = vegas_params['nitn']
        neval = int(vegas_params['neval'])
        verbose = vegas_params['verbose']
        if 'neval_init' in vegas_params:
            neval_init = int(vegas_params['neval_init'])
        else:
            neval_init = neval
        # For volume-weighted neval:
        if 'weight_by_vol' in vegas_params:
            weight_by_vol = vegas_params['weight_by_vol']
        else:
            weight_by_vol = True #default
        if 'neval_min' in vegas_params:
            neval_min = vegas_params['neval_min']
        else:
            neval_min = 1e3 # lower bound for neval_n in weight_by_vol version
        # Adjust neval for volume of nth basis function:
        if type(weight_by_vol) in [int, float]:
            dim = weight_by_vol
        else:
            dim = len(volume) # dimensionality of the integral
        if weight_by_vol:
            urange = volume[0] # always in order (u, ang1, ang2...)
            [u1, u2] = urange
            volume_fraction = (u2/self.uMax)**dim - (u1/self.uMax)**dim
            neval_n = neval * volume_fraction
            neval_init_n = neval_init * volume_fraction
            if neval_n < neval_min:
                neval_n = neval_min
            if neval_init_n < neval_min:
                neval_init_n = neval_min
        else:
            neval_n = neval
            neval_init_n = neval_init
        neval_n = int(neval_n)
        neval_init_n = int(neval_init_n)
        if verbose and printheader is not None:
            print(printheader)
        """Perform the integration:"""
        integrator = vegas.Integrator(volume)
        # adjust integrator to integrand:
        integrator(integrand, nitn=nitn_init, neval=neval_init_n)
        result = integrator(integrand, nitn=nitn, neval=neval_n)
        if verbose:
            print(result.summary())
        return result

    def getFnlm(self, f_uvec, nlm, vegas_params, saveGnli=True):
        """Performs <nlm|f> integration for function f of spherical vector uvec.

        f_uvec can have the following attributes:
        * is_gaussian: (boolean) if f_uvec is a GaussianF instance,
            defined by a list of (c_i, uSph_i, sigma_i) parameters.
        * z_even: (boolean) if f_uvec(x,y,z) = f_uvec(x,y,-z)
            implies <lm|f> = 0 if (l+m) is odd
        * phi_even: (boolean) if f_uvec(u,theta,phi) = f_uvec(u,theta,-phi)
            implies <lm|f> = 0 if m is odd
        * phi_cyclic: (integer) if f_uvec(u,theta,phi) = f_uvec(u,theta,phi + 2*pi/n)
        * phi_symmetric: (boolean) if f_uvec(u,theta) independent of phi

        If is_gaussian, then f_uvec is a GaussianF instance, defined by
            a list of (c_i, uSph_i, sigma_i) parameters.
        In this case, the function f(uvec) is called by f_uvec.gU(uvec).
            <g|nlm> given by f_uvec.getGnlm(basisU, nlm, vegas_params).
        Note: f_uvec.getGnlm() only performs numeric integration when the
            corresponding f_uvec.G[(n,l,i)] have not been evaluated yet.
        """
        header = 'Calculating <f|nlm> for nlm: {}'.format(nlm)
        headerA = 'Calculating <f|nlm(A)> for nlm: {}'.format(nlm)
        headerB = 'Calculating <f|nlm(B)> for nlm: {}'.format(nlm)
        # don't let getFnlm modify vegas_params:
        if hasattr(f_uvec, 'is_gaussian') and f_uvec.is_gaussian==True:
            fnlm = f_uvec.getGnlm(nlm, vegas_params, saveGnli=saveGnli)
            return fnlm
        if not hasattr(f_uvec, 'z_even') or f_uvec.z_even in [0, None]:
            f_uvec.z_even = False
        # if z_even, only integrate theta on [0, pi/2]
        theta_Zn = 1
        if f_uvec.z_even:
            theta_Zn = 2
        theta_region = [0, math.pi/theta_Zn]
        # else: not is_gaussian. Use main getFnlm method:
        (n, l, m) = nlm
        if self.basis['type']=='wavelet' and n!=0:
            # Split the integral in two for these cases:
            [umin, umid, umax] = self._baseOfSupport_n(n, getMidpoint=True)
        else:
            [umin, umax] = self._baseOfSupport_n(n, getMidpoint=False)
        # Check for azimuthal symmetry:
        if hasattr(f_uvec, 'phi_symmetric') and f_uvec.phi_symmetric==True:
            if m!=0 or (f_uvec.z_even and (l % 2 != 0)):
                fnlm = gvar.gvar(0,0)
                return fnlm
            #else, m==0:  Skip the phi integral.
            def integrand_m0(u_rth):
                phi = 0.0 #arbitrary, function is constant wrt phi
                uvec = (u_rth[0], u_rth[1], phi)
                return (dV_sph(uvec)/(self.u0**3) * f_uvec(uvec)
                        * self.phiNLM(nlm, uvec))
            if self.basis['type']=='wavelet' and n!=0:
                volume_A = [[umin,umid], theta_region] # 2d
                volume_B = [[umid,umax], theta_region] # 2d
                fnlmA = self.doVegas(integrand_m0, volume_A, vegas_params,
                                     printheader=headerA)
                fnlmB = self.doVegas(integrand_m0, volume_B, vegas_params,
                                     printheader=headerB)
                fnlm = 2*math.pi * theta_Zn * (fnlmA + fnlmB)
            else:
                volume_nl = [[umin, umax], theta_region]
                fnlm = self.doVegas(integrand_m0, volume_nl,
                                    vegas_params, printheader=header)
            return 2*math.pi * theta_Zn * fnlm
        # else: fSph is a function of 3d uvec = (u, theta, phi)
        if not hasattr(f_uvec, 'phi_cyclic') or f_uvec.phi_cyclic in [0, None]:
            f_uvec.phi_cyclic = 1
        if not hasattr(f_uvec, 'phi_even') or f_uvec.phi_even in [0, None]:
            f_uvec.phi_even = False
        # check special cases:
        if (f_uvec.z_even and (l+m) % 2 != 0) or (f_uvec.phi_even and m<0):
            fnlm = gvar.gvar(0,0)
            return fnlm
        # for Z_n symmetric functions, only integrate phi on [0, 2*pi/n]
        phi_region = [0, 2*math.pi/f_uvec.phi_cyclic]
        def integrand_fnlm(uvec):
            return (dV_sph(uvec)/(self.u0**3) * f_uvec(uvec)
                    * self.phiNLM(nlm, uvec))
        if self.basis['type']=='wavelet' and n!=0:
            volume_A = [[umin, umid], theta_region, phi_region]
            volume_B = [[umid, umax], theta_region, phi_region]
            fnlmA = self.doVegas(integrand_fnlm, volume_A, vegas_params,
                                 printheader=headerA)
            fnlmB = self.doVegas(integrand_fnlm, volume_B, vegas_params,
                                 printheader=headerB)
            fnlm = fnlmA + fnlmB
        else:
            volume_nlm = [[umin, umax], theta_region, phi_region]
            fnlm = self.doVegas(integrand_fnlm, volume_nlm, vegas_params,
                                printheader=header)
        return fnlm * f_uvec.phi_cyclic * theta_Zn


    def nlmAssembleFu(self, f_nlm, uvec, nlmlist=None):
        """Returns f(uvec) from f = sum_nlm(f_nlm * phi_nlm)."""
        # here f_nlm is a dictionary of coefficients f_nlm indexed by (nlm)
        if nlmlist is None:
            nlmlist = f_nlm.keys()
        f_uvec = 0.0
        for nlm in nlmlist:
            fX = f_nlm[(nlm)] * self.phiNLM(nlm, uvec)
            f_uvec += fX
        return f_uvec

    def get_F123_sph_haar(self, f_0mp, n_0):
        """F_p at center of n_0 wavelet, from coefficients f_0mp.

        F_p = f^{(p)} (Delta/2)^p / p!           for f(x), 0 < x < 1,
            = (d/du)^p g * (Delta_u/2)^p / p!    for g(u) = f(u/u0).
        """
        assert self.basis['type']=='wavelet', "Extrapolation only for wavelets"
        # (L_minus, M_minus) = (L_0+1, 2*M_0)
        # (L_plus, M_plus) = (L_0+1, 2*M_0+1)
        n_m = 2*n_0
        n_p = 2*n_0 + 1
        H1_0 = _Hp_sph_n(1, n_0)
        H2_0 = _Hp_sph_n(2, n_0)
        H3_0 = _Hp_sph_n(3, n_0)
        H1_m = _Hp_sph_n(1, n_m)
        H2_m = _Hp_sph_n(2, n_m)
        H3_m = _Hp_sph_n(3, n_m)
        H1_p = _Hp_sph_n(1, n_p)
        H2_p = _Hp_sph_n(2, n_p)
        H3_p = _Hp_sph_n(3, n_p)
        A1_m = 0.5*(H1_m )
        A1_p = 0.5*(H1_p )
        A2_m = 0.5*(H2_m - H1_m)
        A2_p = 0.5*(H2_p + H1_p)
        A3_m = 0.5*(H3_m - 1.5*H2_m + 0.75*H1_m)
        A3_p = 0.5*(H3_p + 1.5*H2_p + 0.75*H1_p)
        # column vectors
        c1 = [H1_0, A1_m, A1_p]
        c2 = [H2_0, A2_m, A2_p]
        c3 = [H3_0, A3_m, A3_p]
        ff = f_0mp # f_nlm[n_0], f_nlm[n_m], f_nlm[n_p] values
        mD = np.array([c1, c2, c3]).transpose()
        m1 = np.array([ff, c2, c3]).transpose()
        m2 = np.array([c1, ff, c3]).transpose()
        m3 = np.array([c1, c2, ff]).transpose()
        dD = np.linalg.det(mD)
        d1 = np.linalg.det(m1)
        d2 = np.linalg.det(m2)
        d3 = np.linalg.det(m3)
        # fx1 = d1/dD * (2 / Delta) * 1
        # fx2 = d2/dD * (2 / Delta)**2 * 2
        # fx3 = d3/dD * (2 / Delta)**3 * 6
        F1 = d1/dD
        F2 = d2/dD
        F3 = d3/dD
        # These are df/dx derivatives in terms of dimensionless 0 < x < 1.
        # But, for g(u) = fx(u/u0), Delta^p fx^{(p)} = (Delta u)^p g^{(p)}
        return (F1, F2, F3)

    def get_df123_sph_haar(self, f_0mp, n_0):
        "Returns f', f'', f''', with units of f/(u0^p)"
        [u1,u3] = self._baseOfSupport_n(n_0, getMidpoint=False)
        Delta_u = u3-u1
        (F1, F2, F3)  = self.get_F123_sph_haar(f_0mp, n_0)
        g1 = F1 * math.factorial(1) * (2/Delta_u)**1
        g2 = F2 * math.factorial(2) * (2/Delta_u)**2
        g3 = F3 * math.factorial(3) * (2/Delta_u)**3
        return (g1, g2, g3)

    @staticmethod
    def _n_list_for_cubic_methods(n_max):
        "Breaks [0,1] interval into bases of support of the [n] wavelets."
        assert n_max >= 3, "Must have at least 3 wavelet generations (lambda>=2)"
        # n_max = 3 would use the n=1,2,3 wavelets to get one set of derivatives
        # would return n_ordered = [1], i.e. only one cubic interpolation region
        if n_max % 4 != 3:
            n_max = n_max - (n_max % 4) - 1
        # n_max should be 3 mod 4, so n_coeffs = n_max + 1 is zero mod 4
        n_coeffs = n_max + 1
        n_start = int(0.25*n_coeffs)
        n_end = int(0.5*n_coeffs) - 1
        lambda_start = int(math.log2(n_start))
        mu_start = n_start - 2**lambda_start
        Delta_start = 2**(-lambda_start)
        # Can use partial generations, wrapping around from x=1 back to x=0
        # the penultimate generation of coefficients determines the x->n map
        # order the n list by their x values
        n_start_b = 2**(lambda_start+1) #first element of (lambda_a+1) generation
        list1 = [n for n in range(n_start_b, n_end+1)] #can have zero length
        list2 = [n for n in range(n_start, n_start_b)]
        n_ordered = list1 + list2
        return n_ordered

    def cubic_extrapolator_lm(self, f_lm_n, n_max):
        """Cubic extrapolation from 2 generations of wavelets (ending at n_max).

        For calculating larger-n wavelet coefficients from f1, f2, f3 derivates.
        f_lm_n: ordered list of coefficients n for fixed (lm)
            Ordered: f_lm_n[n] = f_nlm[(n,l,m)] for n=0,1,...,n_max
            returns: f_lm(u) interpolation
        n_max: determines which coefficients to use to find f' derivatives
            -> (n_max+1)/2, (n_max+1)/2 + 1, ..., n_max - 1, n_max.
        u: radial coordinate, u = x*uMax.

        Output: an Interpolator object, with f0=0 in each bin.
            (can be updated later for cubic_interpolation)
        """
        n_ordered = self._n_list_for_cubic_methods(n_max)
        #get x0_vals and x_boundaries
        u_bounds = [self._baseOfSupport_n(n_ordered[0], getMidpoint=False)[0]]
        u0_vals = []
        f0123_vals = []
        for n in n_ordered:
            (u1, u2, u3) = self._baseOfSupport_n(n, getMidpoint=True)
            u0_vals += [u2]
            u_bounds += [u3]
            n_m = 2*n
            n_p = 2*n + 1
            f_0mp = (f_lm_n[n], f_lm_n[n_m], f_lm_n[n_p])
            (df1, df2, df3) = self.get_df123_sph_haar(f_0mp, n)
            f0123 = [0, df1, df2, df3]
            f0123_vals += [f0123]
        return Interpolator(u_bounds, u0_vals, f0123_vals)

    def cubic_interpolation(self, f_lm_n, n_max, f0_lm_n):
        """Uses f0_lm_n values of f(u_2) to interpolate f_lm(x).

        Arguments:
        f_lm_n: list of coefficients f_lm[n] for n=0,1,2...
        n_max: determines which coefficients to use for extrapolation
        f0_lm_n: the value of f_lm at the center of each wavelet n=0,1,2...

        Output:
        modified cubic_interpolator with f0 values taken from f0_lm_n
        """
        n_list = self._n_list_for_cubic_methods(n_max)
        cubic_interpolator = self.cubic_extrapolator_lm(f_lm_n, n_max)
        for n in n_list:
            f0 = f0_lm_n[n]
            # the nth wavelet maps onto the (ix)th bin of the interpolation
            (u1,u2,u3) = self._baseOfSupport_n(n, getMidpoint=True)
            ix = cubic_interpolator.map_u_ix(u2)
            cubic_interpolator.f0123_vals[ix][0] = f0
        return cubic_interpolator







#
