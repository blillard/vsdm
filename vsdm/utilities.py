"""Functions for general use: physics functions, GVAR matrix conversions.

Physics functions:
    fdm2_n: DM-SM particle scattering form factor, normalized to F(q=q0) = 1.
        Value of q0 = q0_fdm is defined in units.py in units of qBohr.
    g_k0: scattering event rate scaling factor with target exposure
Physics functions require .units, e.g. for q0 and g_k0().

Utilities:
    makeNLMlist: produces complete list of (nlm) coefficients given
      values for nMax and ellMax, n=0,1...nMax and ell=0,1,...ellMax
    splitGVARarray: separates gvar valued matrix into f.mean, f.sdev matrices
    joinGVARarray: combines f.mean, f.sdev matrices into gvar valued matrix
"""

__all__ = ['g_k0', 'fdm2_n', 'mathsinc', 'dV_sph', 'legendrePl',
           'plm_real', 'ylm_real', 'makeNLMlist',
           'splitGVARarray', 'joinGVARarray', '_LM_to_x', '_x_to_LM',
           'sph_to_cart', 'cart_to_sph', 'compare_index_to_shape',
           '_map_int_to_index', 'gX_to_tgX', 'assign_indices',
           'getLpower', 'getLMpower', 'getNLMpower',
           'Interpolator']

import math
import numpy as np
import scipy.special as spf
# import vegas # numeric integration
import gvar # gaussian variables; for vegas
# import time
# import quaternionic # For rotations
# import spherical #For Wigner D matrix
# import csv # file IO for projectFnlm
# import os.path
# import h5py

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
    Mathematics functions: legendrePl, dV_sph integration Jacobian
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

def legendrePl(ell, z):
    poly = spf.legendre(ell, monic=False)
    return poly(z)


"""
    Plm and Ylm functions
"""

def plm_real(ell, m, z):
    ## lpmn calculates P_mn(z) for all m and n less than or equal to the m,n provided.
    # A more efficient code would vectorize Ylm to take advantage of this
    return spf.lpmn(math.fabs(m), ell, z)[0][-1][-1]

def ylm_real(ell, m, theta, phi):
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

class Interpolator():
    """Interpolated representation of a function f(u).

    Arguments:
        u_bounds: list of boundaries between interpolation regions, length [n+1]
        u0_vals: points at which f derivatives are evaluated --> (u-u0)**p
        f0123_vals: derivatives d^(p)f/du^p, for p=0,1,2,3...
    """

    def __init__(self, u_bounds, u0_vals, f0123_vals):
        self.u_bounds = u_bounds
        self.u0_vals = u0_vals
        self.f0123_vals = f0123_vals

    def map_u_ix(self, u):
        assert self.u_bounds[0] <= u <= self.u_bounds[-1], "u out of range"
        n = 0
        while u > self.u_bounds[n+1]:
            n += 1
        return n

    def fU(self, u):
        ix = self.map_u_ix(u)
        u0 = self.u0_vals[ix]
        f0123 = self.f0123_vals[ix]
        fu = 0.0
        for p,f_p in enumerate(f0123):
            fu += f_p/math.factorial(p) * (u-u0)**p
        return fu

    def df_du_p(self, p, u):
        "Derivative d^(p)f/du^p"
        ix = self.map_u_ix(u)
        f0123 = self.f0123_vals[ix]
        sum = 0.0
        for k in range(p, len(f0123)):
            sum += f0123[k] * (u-u0)**(k-p) / math.factorial(k-p)
        return sum

    def f_p_u(self, p, ulist):
        if p==0:
            return np.array([self.fU(u) for u in ulist])
        else:
            return np.array([self.df_du_p(p, u) for u in ulist])










#
