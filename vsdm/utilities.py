"""Functions for general use: physics functions, GVAR matrix conversions.

Physics functions:
    fdm2_n: DM-SM particle scattering form factor, normalized to F(q=q0) = 1.
        Value of q0 = q0_fdm is defined in units.py in units of qBohr.
    g_k0: scattering event rate scaling factor with target exposure
These functions require .units, e.g. for q0 and g_k0().

Mathematics:
    plm_norm: normalized associated Legendre polynomials
    ylm_real: real spherical harmonics (m<0 -> sin(m phi), m>0 -> cos(m phi))
        Using plm_norm for substantially faster evaluation.
    ylm_cx: complex-valued spherical harmonics (also using plm_norm).
    sph_to_cart, cart_to_sph: Cartesian/spherical coordinate conversions.
    NIntegrate: multipurpose numerical integrator. Methods include VEGAS
        or gaussian quadrature from scipy.

Utilities:
    makeNLMlist: produces complete list of (nlm) coefficients given
      values for nMax and ellMax, n=0,1...nMax and ell=0,1,...ellMax
    splitGVARarray: separates gvar valued matrix into f.mean, f.sdev matrices
    joinGVARarray: combines f.mean, f.sdev matrices into gvar valued matrix

Interpolation:
    Interpolator1d: 1d interpolation object, representing f(x) with a
        piecewise-defined polynomial function.
    Interpolator3d: represents a 3d function as a sum of spherical harmonics.
        Contains a dictionary of 1d Interpolator objects, labeled by (l,m).
"""

__all__ = ['g_k0', 'fdm2_n', 'mathsinc', 'dV_sph',
           'plm_real', 'plm_norm', 'ylm_real', 'ylm_cx', 'ylm_scipy',
           'makeNLMlist',
           'splitGVARarray', 'joinGVARarray', '_LM_to_x', '_x_to_LM',
           'sph_to_cart', 'cart_to_sph', 'compare_index_to_shape',
           '_map_int_to_index', 'gX_to_tgX', 'assign_indices',
           'getLpower', 'getLMpower', 'getNLMpower', 'NIntegrate',
           'Interpolator1d', 'Interpolator3d']

import math
import numba
import numpy as np
import scipy.special as spf
import scipy.integrate as sint # gaussian quadrature
import vegas # Monte Carlo integration
import gvar # gaussian variables; for vegas

from .units import *


### Scaling factor for exposure*sigma0*rhoX
def g_k0(exp_kgyr=1., mCell_g=1., sigma0_cm2=1e-40,
         rhoX_GeVcm3=0.4, v0=220.*km_s, q0=qBohr):
    # 1 year = 365.0 days in 'kgyr'
    return 3288.95 *( (1.0/mCell_g) * (exp_kgyr/1.) * (sigma0_cm2/1e-40)
        * (rhoX_GeVcm3/0.4) * (v0/(220.*km_s))**2 * (qBohr/q0) )

### Physics function: squared form factor for DM-SM particle scattering
def fdm2_n(q, n):
    """DM-SM scattering form factor, F_DM(q) = (q0/q)**n for integer n."""
    # F = (q0/q)^n. Normalized to F(q0)=1.
    #  Default: q0 = alpha*mE.
    # Function returns F^2, e.g. F^2 = 1 or F^2 = 1/q^4.
    if n==0:
        return 1
    return (q0_fdm/q)**(2*n)

"""
    Mathematics functions: dV_sph integration Jacobian
"""
### Differing conventions:
# numpy/scipy sinc functions are normalized with extra factor of pi
# the incomplete Gamma(s,z) function is implemented as Gamma(s)*spf.gammaincc(s,z)

def mathsinc(z):
    ## mathsinc = sin(z)/z, whereas numpy.sinc(z) = sin(pi z)/(pi z)
    return np.sinc(z / np.pi)

def dV_sph(rvec):
    ## rvec = [r, theta, phi]
    return rvec[0]**2 * math.sin(rvec[1])


"""
    Plm and Ylm functions
"""

def plm_real(ell, m, z):
    """scipy version of associated Legendre polynomials.

    Reliable up to ell=150 (m=0,1,...,150), but can produce
        incorrect results in Ylm when multiplied against sqrt(factorials).
    Very slow.
    """
    ## lpmn calculates P_mn(z) for all m and n less than or equal to the m,n provided.
    # A more efficient code would vectorize Ylm to take advantage of this
    return spf.lpmn(math.fabs(m), ell, z)[0][-1][-1]

def ylm_scipy(ell, m, theta, phi):
    absm = int(math.fabs(m))
    # math.factorial is faster by 20%, but scipy.gamma allows ell > 100
    sqrtfact = (-1)**m * (math.sqrt((2*ell+1)/(4*math.pi)
        # * math.factorial(ell - absm) / math.factorial(ell + absm)))
        * spf.gamma(ell - absm + 1) / spf.gamma(ell + absm + 1)))
    Plmpart = plm_real(ell, m,math.cos(theta))
    if m < 0:
        return math.sqrt(2) * sqrtfact * Plmpart * math.sin(absm * phi)
    elif m==0:
        return sqrtfact * Plmpart
    else:
        return math.sqrt(2) * sqrtfact * Plmpart * math.cos(m * phi)


@numba.jit("double(uint32,uint32,double)", nopython=True,
           locals={'m_sqrd':numba.double, 'Pk':numba.double,
                   'Pk_minus2':numba.double,
                   'Pk_minus1':numba.double,
                   'x2':numba.double, 'sqrt_1_x2':numba.double})
def plm_norm(ell, m, x):
    """The 'normalized' associated Legendre polynomials.

    Defined as: (-1)**m * sqrt[(l-m)! / (l+m)!] * P_lm(x)
    For m=0, this is identical to the usual P_l(x).

    Method:
    * Using Bonnet recursion for the m=0 special case (upwards from l=0,1).
    * For m>0, using 'horizontal' recursion from (m,m) to (l,m),
        using the 'associated' Bonnet recursion relations.

    Numerically stable for all x in [-1,1], even arbitrarily close to x**2=1.
    (e.g. x = 1 - 1e-15).
    Permits the accurate calculation of P_lm(x) up to at least ell=m=1e6.
    """
    # poch_plus = 1. # (l+m)!/l!
    # poch_minus = 1. # l!/(l-m)!
    if ell==0:
        return 1
    if x < -1:
        x = -1
    elif x > 1:
        x = 1
    x2 = x**2
    if x2==1:
        # Evaluate now, to avoid 1/sqrt(1-x**2) division errors.
        return int(m==0) * (x)**(ell%2)
    sqrt_1_x2 = (1-x2)**0.5
    if m==0:
        # Upward recursion along m=0 to (l,0). Bonnet:
        if ell==1:
            return x
        Pk_minus2 = 1
        Pk_minus1 = x
        for k in range(2, ell+1):
            Pk = ((2-1/k)*x*Pk_minus1 - (1-1/k)*Pk_minus2)
            Pk_minus2 = Pk_minus1
            Pk_minus1 = Pk
        return Pk
        # get the (l,1) term, from the (l-1,0) and (l,0) Legendre polynomials:
    # else: use horizontal recursion from (m,m) to (ell,m)
    # modified Bonnet:
    sqrt_1_x2 = (1-x2)**0.5
    m_sqrd = 1. # l!/(l-m)! * l!/(l+m)!
    for i in range(m):
        # until i=m-1:
        m_sqrd *= 1 - 0.5/(1+i)
    Pk_minus2 = sqrt_1_x2**m * m_sqrd**0.5 #l=m
    if ell==m:
        return Pk_minus2
    Pk_minus1 = (2*m+1)**0.5 * x * Pk_minus2 #l=m+1
    if ell==m+1:
        return Pk_minus1
    for k in range(m+2, ell+1):
        Pk = ((2*k-1)*x*Pk_minus1 - ((k-1)**2-m**2)**0.5*Pk_minus2)/(k**2-m**2)**0.5
        Pk_minus2 = Pk_minus1
        Pk_minus1 = Pk
    return Pk


@numba.jit("complex128(uint32,int32,double,double)", nopython=True)
def ylm_cx(ell, m, theta, phi):
    "Complex-valued spherical harmonics."
    phase_phi = np.exp(1j * m * phi)
    if m < 0:
        m = -m
    else:
        phase_phi *= (-1)**(m%2)
    return ((2*ell+1)/(4*np.pi))**0.5 * phase_phi * plm_norm(ell, m, np.cos(theta))

@numba.jit("double(uint32,int32,double,double)", nopython=True)
def ylm_real(ell, m, theta, phi):
    "Real-valued spherical harmonics."
    if m==0:
        return ((2*ell+1)/(4*np.pi))**0.5 * plm_norm(ell, m, np.cos(theta))
    if m < 0:
        m = -m
        return ((2*ell+1)/(2*np.pi))**0.5 * plm_norm(ell, m, np.cos(theta)) * np.sin(m*phi)
    return ((2*ell+1)/(2*np.pi))**0.5 * plm_norm(ell, m, np.cos(theta)) * np.cos(m*phi)




"""
    functions for (nlm) lists and gvar/[mean,sdev] conversions:
"""

def _LM_to_x(ell, m, phi_symmetric=False):
    # Maps (ell, m) onto an index ix=0,1,2,3...
    if phi_symmetric:
        return ell
    else:
        return ell*(ell+1) + m

def _x_to_LM(x, phi_symmetric=False):
    if phi_symmetric:
        ell = x
        m = 0
    else:
        ell = int(math.floor(math.sqrt(x)))
        m = x - ell*(ell+1)
    return ell,m

def makeNLMlist(nMax, lMax, nMin=0, lMin=0,
                mSymmetry=None, lSymmetry=None, phi_even=False):
    # For most basis choices, nMin = 0. For tophat, nMin = 1.
    lList = []
    if lSymmetry is None or lSymmetry==0 or lSymmetry==False:
        for l in range(lMin, lMax+1):
            lList += [l]
    else:
        assert (type(lSymmetry) is int), "lSymmetry must be integer-valued or None"
        for l in range(lMin, lMax+1):
            if (l%lSymmetry)==0:
                lList += [l]
    nlmlist = []
    if mSymmetry in [None,0,False]:
        if phi_even:
            nlmlist = [(n,l,m) for l in lList for m in range(0, l+1)
                       for n in range(nMin, nMax+1) ]
        else:
            nlmlist = [(n,l,m) for l in lList for m in range(-l, l+1)
                       for n in range(nMin, nMax+1)]
    elif mSymmetry==True or mSymmetry=='U(1)':
        nlmlist = [(n,l,0) for n in range(nMin, nMax+1) for l in lList]
    else:
        assert (type(mSymmetry) is int), "mSymmetry must be integer-valued, boolean or None"
        lmList = []
        for l in lList:
            if phi_even:
                for m in range(0, l+1):
                    if (m%mSymmetry)==0:
                        lmList += [(l,m)]
            else:
                for m in range(-l, l+1):
                    if (m%mSymmetry)==0:
                        lmList += [(l,m)]
        nlmlist = [(n,l,m) for (l,m) in lmList for n in range(nMin, nMax+1)]
    return nlmlist

def compare_index_to_shape(index, shape):
    "Returns an array large enough to accommodate new index and original shape."
    # index: for entry in array. shape: of array
    # Returns: shape of original array, or one enlarged to include index
    assert(len(index)==len(shape)), "Error: incompatible objects"
    dim = len(index) # dimensionality of array
    larger_shape = shape
    for i in range(dim):
        if (index[i] + 1) > larger_shape[i]:
            larger_shape[i] = index[i] + 1
    return larger_shape

def _map_int_to_index(ix, shape):
    """Given array shape, maps integer ix to array index."""
    # Get multiplicative factors
    dim = len(shape)
    ncoeffs = 1
    for i in range(dim):
        ncoeffs *= shape[i]
    if ix >= ncoeffs or ix < 0:
        return "Error: index out of range."
    mult_factor = []
    mfactor = ncoeffs
    for i in range(dim):
        mfactor = int(mfactor / shape[i])
        mult_factor += [mfactor]
    # mult_factors is done. Ordered to match consecutive for loop style
    new_x = ix
    index = []
    for i in range(dim):
        index_i = new_x // mult_factor[i]
        rem_i = new_x % mult_factor[i] # remainder carried to i+= 1
        i += 1
        new_x = rem_i
        index += [index_i]
    return tuple(index)

def _map_index_to_int(index, shape):
    """Given array shape, maps array index to integer. Currently unused."""
    dim = len(shape)
    if (len(index)!=dim):
        return "Error: incompatible objects."
    if shape!=compare_index_to_shape(index, shape):
        return "Error: index out of range."
    # Get multiplicative factors
    ncoeffs = 1
    for i in range(dim):
        ncoeffs *= shape[i]
    mult_factor = []
    mfactor = ncoeffs
    for i in range(dim):
        mfactor = int(mfactor / shape[i])
        mult_factor += [mfactor]
    # mult_factors is done. Ordered to match consecutive for loop style
    out = 0
    for i in range(dim):
        out += mult_factor[i] * index[i]
    # out: a unique integer in 0, 1, 2, ..., ncoeffs-1.
    return out

def assign_indices(listlength, pn_cpus):
    """Assigns a list of indices to processor 'p', one of 'n' total."""
    (p_index, nprocesses) = pn_cpus
    # this is for the pth process out of a total of n
    index_list = []
    index = p_index
    while index < listlength:
        index_list += [index]
        index += nprocesses
    return index_list

def splitGVARarray(mxgvar):
    mxshape = np.shape(mxgvar)
    dim = len(mxshape) # dimensionality of array
    ncoeffs = 1
    for i in range(dim):
        ncoeffs *= mxshape[i]
    mxMean = np.zeros(mxshape, dtype='float')
    mxSdev = np.zeros(mxshape, dtype='float')
    for ix in range(ncoeffs):
        index = _map_int_to_index(ix, mxshape)
        mgv = mxgvar[index]
        if (type(mgv) is float) or (type(mgv) is int):
            mgv *= gvar.gvar(1., 0)
        mxMean[index] = mgv.mean
        mxSdev[index] = mgv.sdev
    return mxMean, mxSdev

def joinGVARarray(mxmean, mxsdev):
    mxshape = np.shape(mxmean)
    dim = len(mxshape) # dimensionality of array
    ncoeffs = 1
    for i in range(dim):
        ncoeffs *= mxshape[i]
    mxgvar = np.zeros(mxshape, dtype='object')
    for ix in range(ncoeffs):
        index = _map_int_to_index(ix, mxshape)
        m_mean = mxmean[index]
        if mxsdev is not None:
            m_sdev = mxsdev[index]
        else:
            m_sdev = 0.0
        mxgvar[index] = gvar.gvar(m_mean, m_sdev)
    return mxgvar

def sph_to_cart(uSph):
    """Converts vectors from spherical to Cartesian coordinates."""
    (u, theta, phi) = uSph
    ux = u * math.sin(theta) * math.cos(phi)
    uy = u * math.sin(theta) * math.sin(phi)
    uz = u * math.cos(theta)
    return (ux, uy, uz)

def cart_to_sph(uXYZ):
    """Converts vectors from Cartesian to spherical coordinates."""
    (ux, uy, uz) = uXYZ
    u = math.sqrt(ux**2 + uy**2 + uz**2)
    uxy = math.sqrt(ux**2 + uy**2)
    # first address uxy=0 and ux=0 special cases...
    phi = 0 #arbitrary; phi not well defined at theta=0,pi
    if uxy==0:
        if uz >= 0:
            theta = 0
        else:
            theta = math.pi
        return (u, theta, phi)
    theta = 0.5*math.pi - math.atan(uz/uxy)
    # ux=0...
    if ux == 0:
        if uy > 0:
            phi = 0.5*math.pi
        elif uy < 0:
            phi = 1.5*math.pi
        return (u, theta, phi)
    # Now, non-special cases...
    if ux > 0 and uy > 0:
        phi = math.atan(uy/ux)
    elif ux < 0 and uy < 0:
        phi = math.atan(uy/ux) + math.pi
    elif ux < 0 and uy > 0:
        phi = math.atan(uy/ux) + math.pi
    elif ux > 0 and uy < 0:
        phi = math.atan(uy/ux) + 2*math.pi
    return (u, theta, phi)

def gX_to_tgX(gauss, u0):
    tgauss_vecs = gauss.rescaleGaussianF(u0**3)
    return vs3dm.GaussianF(tgauss_vecs)

def getNLMpower(f_nlm):
    #assemble
    powerNLM = {}
    for nlm,fnlm in f_nlm.items():
        powerNLM[nlm] = fnlm**2
    #sort
    sortnlm = sorted(powerNLM.items(),
                    key=lambda z: z[1], reverse=True)
    powerNLM = {}
    for key,power in sortnlm:
        powerNLM[key] = power
    return powerNLM

def getLMpower(f_nlm):
    #assemble
    powerLM = {}
    for nlm,fnlm in f_nlm.items():
        (n, l, m) = nlm
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

def getLpower(f_nlm):
    #assemble
    powerL = {}
    for nlm,fnlm in f_nlm.items():
        (n, l, m) = nlm
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


# Numerical integration with gaussian quadrature:
def intGaussQuad(integrand, volume, quadg_params):
    """Gaussian quadrature integrals, using quadg_params dictionary.

    Using VEGAS form of integrand([x1,x2,...]), volume=[[x1a,x1b],[x2a,x2b]...]
    quadg_params: a dict containing the precision goals
    """
    if 'rtol' in quadg_params:
        rtol = quadg_params['rtol']
    else:
        rtol = 1e-6
    if 'atol' in quadg_params:
        atol = quadg_params['atol']
    elif 'atol_f' in quadg_params:
        atol = quadg_params['atol_f']
    elif 'atol_e' in quadg_params:
        atol = quadg_params['atol_e']
    else:
        atol = 1e-6
    verbose = False
    if 'verbose' in quadg_params:
        verbose = quadg_params['verbose']
    # complex = False
    # if 'complex' in quadg_params:
    #     complex = quadg_params['complex']
    ndim = len(volume) # dimensionality
    assert ndim in [1,2,3], "Only using quadrature for 1d, 2d, or 3d integrands"
    if ndim==1:
        def sciFunc(x):
            return integrand([x])
        result = sint.quad(sciFunc, volume[0][0], volume[0][1],
                           epsabs=atol, epsrel=rtol)
    elif ndim==2:
        def sciFunc(x,y):
            return integrand([x,y])
        # note: scipy uses backwards func(y,x) ordering
        result = sint.dblquad(sciFunc, volume[1][0], volume[1][1],
                              volume[0][0], volume[0][1],
                              epsabs=atol, epsrel=rtol)
    elif ndim==3:
        def sciFunc(x,y,z):
            return integrand([x,y,z])
        # note: scipy uses backwards func(z,y,x) ordering
        result = sint.tplquad(sciFunc, volume[2][0], volume[2][1],
                              volume[1][0], volume[1][1],
                              volume[0][0], volume[0][1],
                              epsabs=atol, epsrel=rtol)
    value, error = result
    resultGVAR = gvar.gvar(value, error)
    if verbose:
        print(resultGVAR)
    return resultGVAR


# Numerical integration with VEGAS:
def intVegas(integrand, volume, vegas_params):
    "Tool for performing VEGAS integrals using vegas_params dictionary."
    # Unpack VEGAS parameters
    nitn_init = vegas_params['nitn_init']
    nitn = vegas_params['nitn']
    neval = int(vegas_params['neval'])
    if 'verbose' in vegas_params:
        verbose = vegas_params['verbose']
    else:
        verbose = False
    if 'neval_init' in vegas_params:
        neval_init = int(vegas_params['neval_init'])
    else:
        neval_init = neval
    dim = len(volume) # dimensionality of the integral
    # Perform the integration:
    integrator = vegas.Integrator(volume)
    # adjust integrator to integrand:
    integrator(integrand, nitn=nitn_init, neval=neval)
    result = integrator(integrand, nitn=nitn, neval=neval)
    if verbose:
        print(result.summary())
    return result

def NIntegrate(integrand, volume, integ_params, printheader=None):
    "Numerical integration, using VEGAS or scipy.integrate.quad."
    if 'method' not in integ_params:
        integ_params['method'] = 'vegas'
    if 'verbose' not in integ_params:
        integ_params['verbose'] = False
    if integ_params['verbose']==True and printheader is not None:
        print(printheader)
    # Numerical Integration:
    if integ_params['method'] == 'vegas':
        out = intVegas(integrand, volume, integ_params)
    elif integ_params['method'] == 'gquad':
        out = intGaussQuad(integrand, volume, integ_params)
    return out


class Interpolator1d():
    """Interpolated representation of a 1d function f(u).

    Arguments:
        u_bounds: list of boundaries between interpolation regions, length [n+1]
        u0_vals: points at which f derivatives are evaluated --> (u-u0)**p
        f_p_list: derivatives d^(p)f/du^p, for p=0,1,2,3...
            Note the inclusion of df_p[0] = f(x0).
    """

    def __init__(self, u_bounds, u0_vals, f_p_list):
        self.u_bounds = u_bounds
        self.u0_vals = u0_vals
        self.f_p_list = f_p_list

    def __call__(self, u):
        return self.fU(u)

    def map_u_ix(self, u):
        assert self.u_bounds[0] <= u <= self.u_bounds[-1], "u out of range"
        n = 0
        while u > self.u_bounds[n+1]:
            n += 1
        return n

    def fU(self, u):
        ix = self.map_u_ix(u)
        u0 = self.u0_vals[ix]
        df_p_x = self.f_p_list[ix]
        fu = 0.0
        for p,f_p in enumerate(df_p_x):
            fu += f_p/math.factorial(p) * (u-u0)**p
        return fu

    def df_du_p(self, p, u):
        "Derivative d^(p)f/du^p"
        ix = self.map_u_ix(u)
        df_p_x = self.f_p_list[ix]
        sum = 0.0
        for k in range(p, len(df_p_x)):
            sum += df_p_x[k] * (u-u0)**(k-p) / math.factorial(k-p)
        return sum



class Interpolator3d():
    """Interpolated representation of a 3d function f(u) = sum_lm f_lm(u).

    Arguments:
        fI_lm_dict: list of Interpolator1d objects, indexed by (lm)
        complex: whether to use real or complex spherical harmonics
    """

    def __init__(self, fI_lm_dict, complex=False):
        self.fI_lm = fI_lm_dict
        self.complex = complex

    def __call__(self, uSph):
        return self.fU(uSph)

    def fU(self, uSph):
        "Evaluating f(u) at a point uSph=(u,theta,phi)."
        (u,theta,phi) = uSph
        fu = 0.
        if self.complex:
            for lm,Flm in self.fI_lm.items():
                (ell, m) = lm
                fI_lm_u = Flm(u)
                fu += ylm_cx(ell, m, theta, phi) * fI_lm_u
        else:
            for lm,Flm in self.fI_lm.items():
                (ell, m) = lm
                fI_lm_u = Flm(u)
                fu += ylm_real(ell, m, theta, phi) * fI_lm_u
        return fu

    def flm_u(self, lm, u):
        "Evaluating <f|lm>(u) at radial coordinate u."
        return self.fI_lm[lm](u)

    def flm_grid(self, ulist):
        """Evaluates all fI_lm(u) for all [u in ulist].

        Output is a 2d numpy array, with rows lm ordered by self.fI_lm.keys().
        """
        flmgrid = np.zeros([len(fI_lm.keys()), len(ulist)])
        ix = 0
        for lm,Flm in self.fI_lm.items():
            fI_lm_ug = np.array([Flm(u) for u in ulist])
            flmgrid[ix] = fI_lm_ug
            ix += 1
        return flmgrid




#
