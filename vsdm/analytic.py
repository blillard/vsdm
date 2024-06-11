"""Analytic results for mcalI matrix calculation.

Dimensionless integrals of functions of q and v_min(q), as functions
of x = (q / q_star)**2, for q_star**2 = 2 * omegaS * mX.

Functions:
    _t: integral for rectangular volume, [[v1,v2],[q1,q2]]
    _u: integral where v_min(q) sets lower bound, [v_min(q),v2]
    _b: intermediate function for rectangular integral
    _c, _d, _s: intermediate function for non-rectangular integrals (for _u)
"""

__all__ = ['mI_star', 't_ln_vq_int', 'u_ln_vq_int',
           '_b_nk_int', '_c_alpha_int', '_s_n_int']

import math
import numpy as np
import scipy.special as spf
# import vegas # numeric integration
# import gvar # gaussian variables; for vegas
# import time
# import quaternionic # For rotations
# import spherical #For Wigner D matrix
# import csv # file IO for projectFnlm
# import os.path
# import h5py # database format for mathcalI arrays

# from .units import *

def _b_nk_int(n, k, x):
    # x = q^2 / q_*^2.
    assert k-int(k)==0, "'k' should be integer valued"
    k = int(k)
    assert x!=0, "Range in 'x' should not include x=0, even if basis function includes q=0."
    sum = 0.
    for j in range(k+1):
        comb = math.factorial(k) / (math.factorial(j) * math.factorial(k-j))
        ipower = j - n - k/2 + 1
        if ipower==0:
            summand = math.log(x)
        else:
            summand = (x**ipower)/ipower
        sum += comb*summand
    return 0.5*sum

def _c_alpha_int(alpha, x):
    # Updated to 'PRD' format
    if alpha==-2:
        return -0.5*math.log( (1+x)/x ) + 0.5/x - 0.125/x**2
    elif alpha==-1:
        return math.log( (1+x)/x ) - 0.5/x
    elif alpha==0:
        return (-0.25*(math.log(x))**2 + math.log(x) * math.log(1 + x)
                + spf.spence(1 + x))
    elif alpha==1:
        return 0.5*x - math.log(1 + x)
    elif alpha==2:
        return 0.125*x**2 - 0.5*x + 0.5*math.log(1 + x)
    else:
        return x**(alpha)/alpha**2 * (0.5 - spf.hyp2f1(1, alpha, 1 + alpha, -x))

def _v_n_int(n, x):
    if n==0:
        return 0.5*math.log(x) + x + 0.25*x**2
    elif n==1:
        return 0.5/x + math.log(x) + 0.5*x
    elif n==2:
        return -0.25/x**2 - 1/x + 0.5*math.log(x)
    else:
        return 0.5*(x**(-n)/(-n)) + x**(-n+1)/(-n+1) + 0.5*x**(-n+2)/(-n+2)

def _s_n_int(n, x):
    logfactor = 0.5*math.log(x/(1 + 2*x + x**2))
    return (logfactor*_v_n_int(n, x) + 0.5*_c_alpha_int(-n, x)
            + _c_alpha_int(1-n, x) + 0.5*_c_alpha_int(2-n, x))

def t_ln_vq_int(l, n, v12_star, q12_star):
    """Rectangular integral T_{l,n}.

    With [v1,v2] in units of v_star = q_star/mX, [q1,q2] in units of
        q_star = sqrt(2*mX*omegaS).

    Always v1 >= 1. Also require q1 > 0.
    """
    assert int(l)-l==0, "'l' must be integer valued"
    l = int(l)
    [v1,v2] = v12_star
    [q1,q2] = q12_star
    x1 = q1**2
    x2 = q2**2
    sum = 0.
    for k in range(l%2, l+1, 2):
        # only terms with (l-k)%2==0 contribute to the sum:
        term_k = 2**(l-k) * spf.gamma(0.5*(k+1+l)) / spf.gamma(0.5*(k+1-l))
        if k==2:
            termV = math.log(v2/v1) / (2 * math.factorial(l-2))
            termQ = (_b_nk_int(n, 2, x2) - _b_nk_int(n, 2, x1))
            sum += termV*term_k*termQ
        else:
            termV = ((v2**(2-k) - v1**(2-k))
                     / ((2-k)*math.factorial(k)*math.factorial(l-k)))
            termQ = (_b_nk_int(n, k, x2) - _b_nk_int(n, k, x1))
            sum += termV*term_k*termQ
    return sum

def u_ln_vq_int(l, n, v2_star, q12_star):
    """Non-rectangular integral U_{l,n}, with lower bound v1 = v_min(q).

    With v2 in units of v_star = q_star/mX, [q1,q2] in units of
        q_star = sqrt(2*mX*omegaS).
    """
    assert int(l)-l==0, "'l' must be integer valued"
    l = int(l)
    v2 = v2_star # only need v2, v1 is irrelevant
    [q1,q2] = q12_star
    x1 = q1**2
    x2 = q2**2
    sum = 0.
    for k in range(l%2, l+1, 2):
        term_k = (spf.gamma(0.5*(k+1+l)) / spf.gamma(0.5*(k+1-l))
                  * 2**(l-k)/(math.factorial(k) * math.factorial(l-k)))
        if k==2:
            summand = (math.log(2*v2)*_b_nk_int(n, 2, x2) + _s_n_int(n, x2)
                       - math.log(2*v2)*_b_nk_int(n, 2, x1) - _s_n_int(n, x1))
            sum += term_k*summand
        else:
            summand = (v2**(2-k)*(_b_nk_int(n, k, x2) - _b_nk_int(n, k, x1))
                       - 2**(k-2)*(_b_nk_int(n, 2, x2) - _b_nk_int(n, 2, x1)))
            sum += term_k*summand/(2-k)
    return sum

def mI_star(ell, n, v12_star, q12_star):
    """Dimensionless integral related to MathcalI.

    This is 'I^{(\ell)}_\star' without the prefactor (qBohr/qStar)**(2n)

    With v12, q12 in units of vStar, qStar.

    There are 0, 1, 2 or 3 regions that contribute to mcalI:
        qA < (R1) < qB < (R2) < qC < (R3) < qD.
        R2 is rectangular, bounded by v1 < v < v2. -> t_ln_vq_int
        R1 and R3 are not rectangular: vMin(q) < v < v2. -> u_ln_vq_int
    If vmin(q) > v1 for all q1 < q < q2, then mcalI is given by u_ln_vq_int
    """
    [v1,v2] = v12_star
    [q1,q2] = q12_star
    if v1==v2 or q1==q2:
        return 0 # No integration volume
    assert q1 < q2, "Need q12 to be ordered"
    assert v1 < v2, "Need v12 to be ordered"
    include_R2 = True
    if v2 < 1:
        # v2 is below the velocity threshold. mcaI=0
        return 0
    tilq_m,tilq_p = v2 - math.sqrt(v2**2-1), v2 + math.sqrt(v2**2-1)
    if tilq_m > q2 or tilq_p < q1:
        # in this case v2 < vmin(q) for all q in [q1,q2]
        return 0
    # Else: there are some q satisfying vmin(q) < v2 in this interval.
    if v1 < 1:
        # There is no v1 = vmin(q) solution for any real q
        include_R2 = False
    # Else: There are two real solutions to v1 = vmin(q)
    else:
        q_m,q_p = v1 - math.sqrt(v1**2-1), v1 + math.sqrt(v1**2-1)
        if q_m > q2 or q_p < q1:
            # in this case v1 < vmin(q) for all q in [q1,q2]
            include_R2 = False
    if include_R2 is False:
        q_A = np.max([q1, tilq_m])
        q_B = np.min([q2, tilq_p])
        return u_ln_vq_int(ell, n, v2, [q_A, q_B])
    # Else: at least part of the integration volume is set by v1 < v.
    q_a = np.max([q1, tilq_m])
    q_b = np.max([q1, q_m]) # q_m > tilq_m iff v2 > v1
    q_c = np.min([q2, q_p]) # q_p < tilq_p iff v2 > v1
    q_d = np.min([q2, tilq_p])
    includeRegion = [True, True, True]
    if q_a==q_b:
        includeRegion[0] = False
    if q_c==q_d:
        includeRegion[2] = False
    if v1>1:
        assert q_b!=q_c, "If q_b==q_c then there should be no R2 region..."
    mI_0,mI_1,mI_2 = 0, 0, 0
    if includeRegion[0]:
        mI_0 = u_ln_vq_int(ell, n, v2, [q_a, q_b])
    if includeRegion[1]:
        mI_1 = t_ln_vq_int(ell, n, [v1,v2], [q_b, q_c])
    if includeRegion[2]:
        mI_2 = u_ln_vq_int(ell, n, v2, [q_c, q_d])
    return mI_0 + mI_1 + mI_2
#









#
