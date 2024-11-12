"""VSDM: orthogonal basis functions for class Basis().

Functions and classes:
    spherical harmonics: see utilities.py
    spherical wavelets: see haar.py
    lag_spherical: spherical Laguerre functions
    Basis: defines an orthogonal basis of functions, with methods for
        <f|nlm> integration, f(r) = sum_nlm(<nlm|f>|nlm>) summation

Functions here are all dimensionless (no need for .units).
"""

__all__ = ['Basis', 'Basis1d', 'lag_spherical', 'tophat_value',
           'f_tophat_to_sphwave']

import math
import numpy as np
import scipy.special as spf
import gvar # gaussian variables; for vegas

from .utilities import *
from .haar import *


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
    (Spherical wavelet functions: Moved to haar.py.)

    Tophat basis functions:
"""

def tophat_value(x1, x2, dim=3):
    """Returns the value of the tophat function where it is nonzero.

    i.e. for x1 < x < x2. Assuming x1 != x2, and that both are nonnegative.
    Normalized for int(x^2 dx * value**2, {x, x1, x2}) = 1.
    """
    return math.sqrt(dim/(x2**dim - x1**dim))

def f_tophat_to_sphwave(f_n_list, dim=3):
    """Change of basis from normalized tophat functions to spherical wavelets.

    Wavelet index: m = 2**lambda + mu
    Tophat coefficients: f_n = <f|bin_n>
    <wave_m|f> = sum_n <bin_n|f> <wave_m|bin_n>
    (assuming real-valued functions)
    Wavelet transform has 2**(lambdaMax+1) many coefficients
    """
    nMax = len(f_n_list) - 1
    power2 = math.ceil(math.log2(nMax+1))
    try:
        assert 2**power2 == nMax+1, "Warning: len(f_n_list) is not a power of 2. Padding..."
    except AssertionError:
        diff = 2**power2 - 1 - nMax
        f_n_list += diff*[0]
        nMax = len(f_n_list) - 1
    # First: undo the normalization, to get discretized f(x) rather than <bin_n|f>.
    # make list of C_n values, i.e. |n>(x) = (C_n inside base of support)
    c_n = [tophat_value(n/2**power2, (n+1)/2**power2, dim=dim)
                         for n in range(nMax+1)]
    f_x = [f_n_list[n] * c_n[n] for n in range(nMax+1)]
    return haar_transform(f_x, dim=dim)


class Basis1d():
    """Defines a set of (nlm) basis functions for any type.

    Arguments:
        basis: a dictionary including the following: (mandatory)
        * type = 'laguerre', 'wavelet', 'tophat'
            - new type: 'hybrid(lambda0)' (tophat/wavelet)
            first n=0,1,...,2**lambda0-1 are equal-width bins
            subsequent n=2**lambda0,2**lambda0+1, ... are all wavelets
        * u0: sets normalization of |nlm>
            is the scale factor for infinite-domain radial functions
        conditional/optional items:
        * uiList: mandatory for 'tophat', the list of u_i to be used
        * uMax: mandatory for 'laguerre', setting the limit of integration on u.
            For 'wavelet' and 'tophat', can set uMax=None.
        * dim = 1,3: specifies the dimensionality of the radial functions,
            for Haar and tophat basis functions. dim=3 is the default (e.g. for
            spherical Haar wavelets), dim=1 for standard Haar wavelets.
    Methods and variables:
        .basis: dict with (type, u0, uMax, uiList, etc.)
        _r_n_x(n, x, l=ell): radial function r^{ell}_n(x), with
            dimensionless x = u/u0. Index 'l' only relevant for 'laguerre' basis
        getFn(f, n): evaluates 1d integral <f|r_n> for radial function r_n.
    """
    # For now, not including spherical bessel functions.
    # For j_ell, need to initialize list of zeros, alpha_{l,n}
    def __init__(self, basis):
        """Makes self.basis, self.u0, """
        self.basis = basis
        update_dict = self.runBasisCheck(basis)
        for lbl,entry in update_dict.items():
            self.basis[lbl] = entry
        self.uMax = self.basis['uMax']
        self.xMax = self.uMax/self.u0

    def runBasisCheck(self, basis):
        """Ensures mandatory items are included, generates class variables.

        Arguments:
            basis: dictionary of basis parameters
        Returns: dict 'updates' of entries in self.basis that should be updated,
                e.g. to define 'uMax' from 'u0' if appropriate
        """
        updates = {}
        assert('u0' in basis), "Missing mandatory parameter: 'u0'."
        self.u0 = basis['u0']
        assert('type' in basis), "Missing mandatory parameter: 'type'."
        uType = basis['type']
        if uType=="wavelet":
            assert('uMax' in basis), "Missing mandatory parameter: 'uMax'."
        elif uType=='tophat':
            assert('uiList' in basis), "tophat: need uiList."
            #uiList should be ordered and nonnegative
            self.uiList = basis['uiList']
            self.xiList = [u/self.u0 for u in self.uiList] #dimensionless x
            assert(all(self.uiList[i] < self.uiList[i+1]
                       for i in range(len(self.uiList) - 1))), "tophat: uiList must be ordered"
            assert(self.uiList[0] >= 0), "tophat: all u_i must be nonnegative"
            if 'uMax' not in basis:
                updates['uMax'] = basis['uiList'][-1]
            else:
                assert(basis['uMax'] == basis['uiList'][-1]), "tophat: require uMax=uiList[-1]."
        elif uType=="laguerre":
            assert('uMax' in basis), "Missing mandatory parameter: 'uMax'."
        return updates

    def _r_n_x(self, n, x, l=0):
        """Dimensionless radial function r_n^(l)(x), x=u/u0.

        Normalization:
            integral(x**(dim-1) dx _r_n_x**2) = 1.
        ell: specifies type of radial function for Laguerre, Bessel, etc.
        n: radial index n=0,1,2,...nMax.

        Note: "wavelet" and "tophat" support arbitrary self.dim=1,2,3...
            "laguerre" functions assume 3d normalization, self.dim=3.
        """
        # x = u/self.u0
        if self.basis['type']=="wavelet":
            return haar_fn_x(n, x, dim=self.dim)
        elif self.basis['type']=="laguerre":
            return lag_spherical(n, ell, x)
        elif self.basis['type']=="tophat":
            x_n,x_np1 = self.uiList[n]/self.u0, self.uiList[n+1]/self.u0
            if x_n < x < x_np1:
                return tophat_value(x_n, x_np1, dim=self.dim)
            else:
                return 0

    def _u_baseOfSupport(self, n, getMidpoint=False):
        """Returns the range in u for which _r_n_x(u/u0) is nonzero

        For wavelets, also provides the mid-point if (getMidpoint)
            (Used for analytic MathcalI.)
        When used to provide integration range for Vegas,
            keep getMidpoint=False.
        """
        if self.basis['type']=='tophat':
            return [self.uiList[n], self.uiList[n+1]]
        elif self.basis['type']=='wavelet' and getMidpoint:
            if n==0:
                if getMidpoint:
                    return [0, self.uMax, self.uMax]
                else:
                    return [0, self.uMax]
            else:
                xA,xM,xB = haar_x123(n)
                if getMidpoint:
                    return [xA*self.u0, xM*self.u0, xB*self.u0]
                else:
                    return [xA*self.u0, xB*self.u0]
        else:
            return [0, self.uMax]

    def _x_baseOfSupport(self, n, getMidpoint=False):
        """Returns the range in x for which _r_n_x(x) is nonzero

        For wavelets, also provides the mid-point if (getMidpoint)
            (Used for analytic MathcalI.)
        When used to provide integration range for Vegas,
            keep getMidpoint=False.
        """
        if self.basis['type']=='tophat':
            return [self.xiList[n], self.xiList[n+1]]
        elif self.basis['type']=='wavelet' and getMidpoint:
            if n==0:
                if getMidpoint:
                    return [0, self.xMax, self.xMax]
                else:
                    return [0, self.xMax]
            else:
                xA,xM,xB = haar_x123(n)
                if getMidpoint:
                    return [xA, xM, xB]
                else:
                    return [xA, xB]
        else:
            return [0, self.xMax]

    def r_n(self, n, u, l=0):
        """Dimensionless radial function r_n^(ell)(x), x=u/u0.

        Normalized: integral(u**2 du r_n(u)**2) = u0**3.
        ell: specifies type of function for Laguerre, Bessel
        n: index n=0,1,2,...nMax.
        """
        return self._r_n_x(n, u/self.u0, l=l)

    def getFn(self, f_u, n, integ_params, dim=1, ell=0):
        """Performs <n|f> integration for function f_u of 1d coordinate u.

            <f|n> = integral(dx * x**(dim-1) * f_n(x) * r_n(x)), for x=u/u0.

        Optional parameters:
            dim: sets weighting function in integration measure, x**(dim-1) dx
            ell: in case the radial function r_n(n, l, u) takes index 'l'
        """
        header = 'Calculating <f|nlm> for nlm: {}'.format(nlm)
        headerA = 'Calculating <f|nlm(A)> for nlm: {}'.format(nlm)
        headerB = 'Calculating <f|nlm(B)> for nlm: {}'.format(nlm)
        # don't let getFnlm modify integ_params:
        # else: not is_gaussian. Use main getFnlm method:
        if self.basis['type']=='wavelet' and n!=0:
            # Split the integral in two for these cases:
            [xmin, xmid, xmax] = self._x_baseOfSupport(n, getMidpoint=True)
        else:
            [xmin, xmax] = self._x_baseOfSupport(n, getMidpoint=False)

        def integrand_fn(xarray):
            x = xarray[0]
            u = x*self.u0
            return (x**(dim-1) * f_u(u) * self._r_n_x(n, x, l=ell))
        if self.basis['type']=='wavelet' and n!=0:
            volume_A = [[xmin, xmid]]
            volume_B = [[xmid, xmax]]
            fnA = NIntegrate(integrand_fn, volume_A, integ_params,
                               printheader=headerA)
            fnB = NIntegrate(integrand_fn, volume_B, integ_params,
                               printheader=headerB)
            fn = fnA + fnB
        else:
            volume_n = [[xmin, xmax]]
            fn = NIntegrate(integrand_fn, volume_n, integ_params,
                              printheader=header)
        return fn



class Basis(Basis1d):
    """Defines a set of (nlm) basis functions for any type.

    Arguments:
        basis: a dictionary including the following: (mandatory)
        * type = 'laguerre', 'wavelet', 'tophat'
            - new type: 'hybrid(lambda0)' (tophat/wavelet)
            first n=0,1,...,2**lambda0-1 are equal-width bins
            subsequent n=2**lambda0,2**lambda0+1, ... are all wavelets
        * u0: sets normalization of |nlm>
            is the scale factor for infinite-domain radial functions
        conditional/optional items:
        * uiList: mandatory for 'tophat', the list of u_i to be used
        * uMax: mandatory for 'laguerre', setting the limit of integration on u.
            For 'wavelet' and 'tophat', can set uMax=None.
        * dim: sets normalization of radial basis function,
            integral(x**(dim-1) dx _r_n_x**2) = 1.
    Methods and variables:
        .basis: dict with (type, u0, uMax, uiList, etc.)
        _r_n_x(n, ell,x): radial function r^{ell}_n(u), with dimensionless x
            Passes dimensionless u/u0 to appropriate basis function
        r_n(n, ell,u): radial function r^{ell}_n(u), with dimensionful u
            _r_n_x(n, ell,u/u0)
        _phi_x(nlm,xvec): basis function (n,l,m), for xvec=[x, theta, phi]
            with dimensionless radial coordinate x = u/u0.
            Normalized: integral(d3x * phi_x(x) * phi_x(x)) = 1
        phi_u(nlm,uvec): basis function (n,l,m), for uvec=[u, theta, phi]
            Normalized: integral(d3u * phi_u(u) * phi_u(u)) = u0**3,
            or u0**dim if self.dim != 3.
        getFnlm(f, nlm): evaluates <f|nlm> for 3d function f(uvec)
        get1nlm(nlm): returns <1|nlm>, used in comparisons with L1 normalized gX
    """
    def __init__(self, basis):
        Basis1d.__init__(self, basis)
        if 'dim' in self.basis:
            self.dim = self.basis['dim']
        else:
            self.dim = 3 #default assumption: 3d basis functions

    def _phi_x(self, nlm, xvec):
        """Dimensionless basis function |nlm> = r_n(x)*Y_lm(theta,phi).

        Normalized: integral(x**(dim-1) dx dOmega * _phi_x**2) = 1,
            for x = u/u0.
        """
        (n, ell, m) = nlm # index: a 3-tuple
        [x, theta, phi] = xvec # spherical coordinate: a vector
        return self._r_n_x(n, x, l=ell) * ylm_real(ell, m, theta, phi)

    def phi_u(self, nlm, uvec):
        """Dimensionless basis function |nlm> = r_n(u)*Y_lm(theta,phi).

        Normalized: integral(u**(dim-1) du dOmega * phi_nlm**2) = u0**(dim).
        """
        [u, theta, phi] = uvec # spherical coordinate: a vector
        return self._phi_x(nlm, [u/self.u0, theta, phi])

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
            x_n,x_np1 = self.xiList[n], self.xiList[n+1]
            return math.sqrt(4*math.pi/3 * (x_np1**3 - x_n**3))

    def getFnlm(self, f_uSph, nlm, integ_params, saveGnli=True):
        """Performs <nlm|f> integration for function f of spherical vector uvec.

        f_uSph can have the following attributes:
        * is_gaussian: (boolean) if f_uSph is a GaussianF instance,
            defined by a list of (c_i, uSph_i, sigma_i) parameters.
        * z_even: (boolean) if f_uSph(x,y,z) = f_uSph(x,y,-z)
            implies <lm|f> = 0 if (l+m) is odd
        * phi_even: (boolean) if f_uSph(u,theta,phi) = f_uSph(u,theta,-phi)
            implies <lm|f> = 0 if m is odd
        * phi_cyclic: (integer) if f_uSph(u,theta,phi) = f_uSph(u,theta,phi + 2*pi/n)
        * phi_symmetric: (boolean) if f_uSph(u,theta) independent of phi

        If is_gaussian, then f_uSph is a GaussianF instance, defined by
            a list of (c_i, uSph_i, sigma_i) parameters.
        In this case, the function f(uvec) is called by f_uSph.gU(uvec).
            <g|nlm> given by f_uSph.getGnlm(basisU, nlm, integ_params).
        Note: f_uSph.getGnlm() only performs numeric integration when the
            corresponding f_uSph.G[(n,l,i)] have not been evaluated yet.
        """
        header = 'Calculating <f|nlm> for nlm: {}'.format(nlm)
        headerA = 'Calculating <f|nlm(A)> for nlm: {}'.format(nlm)
        headerB = 'Calculating <f|nlm(B)> for nlm: {}'.format(nlm)
        # don't let getFnlm modify integ_params:
        if hasattr(f_uSph, 'is_gaussian') and f_uSph.is_gaussian==True:
            fnlm = f_uSph.getGnlm(nlm, integ_params, saveGnli=saveGnli)
            return fnlm
        if not hasattr(f_uSph, 'z_even') or f_uSph.z_even in [0, None]:
            f_uSph.z_even = False
        # if z_even, only integrate theta on [0, pi/2]
        theta_Zn = 1
        if f_uSph.z_even:
            theta_Zn = 2
        theta_region = [0, math.pi/theta_Zn]
        # else: not is_gaussian. Use main getFnlm method:
        (n, l, m) = nlm
        if self.basis['type']=='wavelet' and n!=0:
            # Split the integral in two for these cases:
            [xmin, xmid, xmax] = self._x_baseOfSupport(n, getMidpoint=True)
        else:
            [xmin, xmax] = self._x_baseOfSupport(n, getMidpoint=False)
        # Check for azimuthal symmetry:
        if hasattr(f_uSph, 'phi_symmetric') and f_uSph.phi_symmetric==True:
            if m!=0 or (f_uSph.z_even and (l % 2 != 0)):
                fnlm = gvar.gvar(0,0)
                return fnlm
            #else, m==0:  Skip the phi integral.
            def integrand_m0(x_rth):
                phi = 0.0 #arbitrary, function is constant wrt phi
                xvec = (x_rth[0], x_rth[1], phi)
                uvec = np.array([x_rth[0]*self.u0, x_rth[1], phi])
                return (dV_sph(xvec) * f_uSph(uvec)
                        * self._phi_x(nlm, xvec))
            if self.basis['type']=='wavelet' and n!=0:
                volume_A = [[xmin,xmid], theta_region] # 2d
                volume_B = [[xmid,xmax], theta_region] # 2d
                fnlmA = NIntegrate(integrand_m0, volume_A, integ_params,
                                   printheader=headerA)
                fnlmB = NIntegrate(integrand_m0, volume_B, integ_params,
                                   printheader=headerB)
                fnlm = 2*math.pi * theta_Zn * (fnlmA + fnlmB)
            else:
                volume_nl = [[xmin, xmax], theta_region]
                fnlm = NIntegrate(integrand_m0, volume_nl,
                                  integ_params, printheader=header)
                fnlm *= 2*math.pi * theta_Zn
            return fnlm
        # else: fSph is a function of 3d uvec = (u, theta, phi)
        if not hasattr(f_uSph, 'phi_cyclic') or f_uSph.phi_cyclic in [0, None]:
            f_uSph.phi_cyclic = 1
        if not hasattr(f_uSph, 'phi_even') or f_uSph.phi_even in [0, None]:
            f_uSph.phi_even = False
        # check special cases:
        if (f_uSph.z_even and (l+m) % 2 != 0) or (f_uSph.phi_even and m<0):
            fnlm = gvar.gvar(0,0)
            return fnlm
        # for Z_n symmetric functions, only integrate phi on [0, 2*pi/n]
        phi_region = [0, 2*math.pi/f_uSph.phi_cyclic]
        def integrand_fnlm(xvec):
            uvec = np.array([xvec[0]*self.u0, xvec[1], xvec[2]])
            return (dV_sph(xvec) * f_uSph(uvec)
                    * self._phi_x(nlm, xvec))
        if self.basis['type']=='wavelet' and n!=0:
            volume_A = [[xmin, xmid], theta_region, phi_region]
            volume_B = [[xmid, xmax], theta_region, phi_region]
            fnlmA = NIntegrate(integrand_fnlm, volume_A, integ_params,
                               printheader=headerA)
            fnlmB = NIntegrate(integrand_fnlm, volume_B, integ_params,
                               printheader=headerB)
            fnlm = fnlmA + fnlmB
        else:
            volume_nlm = [[xmin, xmax], theta_region, phi_region]
            fnlm = NIntegrate(integrand_fnlm, volume_nlm, integ_params,
                              printheader=header)
        return fnlm * f_uSph.phi_cyclic * theta_Zn
#
