"""Functions for the generalized Haar wavelets.

Conventions.
    All functions and derivatives in haar.py are with respect to the
        dimensionless variable x on the interval [0,1].
        In basis.py, x = u/u0 (u0==uMax) for dimensionful u,u0.
    Basis function index: n=0,1,2,...,nMax.
        n=0: special 'constant wavelet', proportional to phi_n(x) = 1.
        n>=1: for wavelets of generation L=0,1,... and relative position M,
            M = 0, 1, ..., 2**L-1, to complete the generation 'L'.
        Mapping: n = 2**L + M, for n>0.
    Each new generation doubles the total number of coefficients.
    Any wavelet 'n' (n>0) can be subdivided into its immediate descendants,
        n -> [2*n, 2*n+1], followed by [4n, 4n+1, 4n+2, 4n+3], [8n...8n+7], etc.
    The basis functions are normalized so that the inner products satisfy
        <m|n> = (1 if m==n, 0 otherwise),
    where the inner products include the d-dimensional volume factor:
        <m|n> = integral(x**(d-1) * phi_m(x) * phi_n(x), on 0 < x < 1).
    A wavelet in generation L has a 'base of support' of width 1/2**L. This
        is the interval for which phi_n(x) is nonzero.

Wavelet expansions.
    A function f(x) is approximated by the sum: f = sum_n f_n*|n>
        for basis functions |n>(x), wavelet coefficients f_n = <n|f>.
    The sum over n does not need to include all intermediate values of n,
        e.g. if f(x) has localized small-scale features.
    The evaluated <n|f> SHOULD fill out a complete 'tree' of coefficients:
        i.e. for every n, the sum_n should also include the parent wavelet,
        parent(n) = floor(n/2).

Wavelet transformations.
    A discretized version of f(x) splits the interval [0,1] into 2**L bins
        of equal width, with f_j = average(f(x), for x in the jth bin).
    * haar_transform(f_j) returns the wavelet coefficients that would reproduce
        the discretized f(x).
    * haar_inverse(f_n) maps a list of wavelet coefficients, f_n = <n|f>,
        onto the discretized version of f(x).
    Both methods require sequential lists, f_n = [f[0], f[1], f[2], etc.]

Basic utilities (_haar_):
    'hindex_n', 'hindex_LM': map wavelet index (L,M) -> n or n -> (L,M)
    'haar_sph_value': values of the normalized wavelet basis function, [A, -B]
    'haar_x123': the base of support of the nth wavelet, [x1,x2,x3],
        with x2 listing the midpoint where phi(x) switches from +A to -B.
    'haar_fn_x': the nth wavelet basis function phi_n(x), for x on [0,1]
    'haar_transform','haar_inverse': wavelet transformations to/from discretized
        f_j = [f[0],f[1],...f[nMax]] ~= f(x).

Tracking wavelet expansions (HaarString):
* Consecutive wavelet branchings (n -> [2n, 2n+1]) form a tree of evaluated
    wavelet coefficients. The endpoints of each 'branch' are sufficient to
    describe the entire tree. (n is an endpoint if the wavelets <f|2n> and
    <f|2n+1> are unevaluated.) Each endpoint can be traced to its ancestors
    via n <-- floor(n/2) <-- floor(n/4) <-- ... <-- n=1.
* A wavelet tree can be inferred from a 'HaarString', a list of the endpoint
    'n' values. For a valid HaarString H, the bases of support of the wavelets
    in H cover a single interval in x, [xL, xR], with no overlap (except at
    the edges of each wavelet).
* A complete HaarString H covers the full range of x, [0,1]. Any complete H
    can be derived from the fundamental H_0 = [1] by consecutive subdivision,
    applying the replacement [n] -> [2n, 2n+1] to any entry within H.
    e.g: [1] -> [2, 3] -> [2, 6, 7] or [4, 5, 3].
* A 'block-complete' HaarString tracks the descendants of any one wavelet [n]
    using the same replacement rule.
class HaarString implements these replacement rules. Its methods include
    finding the descendants/ancestors of H; concatenation, H1+H2; tracing
    the wavelet tree back to n=1; finding the greatest common denominator for
    partial HaarStrings; etc.)
* HaarString uses __call__, __iter__, etc so that class instances can be treated
    like the list of 'n' values. E.g. for H = HaarString(hstr=[n0,n1,...]),
    one can still use list comprehension ([n for n in H]) or H[:],
    without having to directly use the saved HaarString list (H.hstr).

Wavelet extrapolation (HaarBlock, HaarExtrapolate).
* Combinations of the last few wavelet generations can produce the coefficients
    of locally-defined Taylor series.
class HaarBlock implements the local wavelet extrapolation.
    A block B_n = HaarBlock(n) provides a local Taylor series, valid on
        the base of support of wavelet 'n'.
    Wavelet extrapolation at polynomial order p_order=p requires 'p' many
        evaluated coefficients in the block, starting with <f|n>.
    For cubic order (p=3), the list [n, 2n, 2n+1] is sufficient; there is also
        a p=7 order method that adds <nd|f> for nd in [4n, 4n+1, 4n+2, 4n+3].
    HaarBlock.get_dfk(f_n_dict) returns the k=1,2,...,p derivatives of f(x),
        evaluated at the block midpoint x2(n), using f_n_dict[n] = <n|f>.
    HaarBlock.extrapolate_n(nd) uses these values to estimate <nd|f> for any
        descendant wavelets nd.
        (The value of f(x2) is not needed here, just its derivatives f^{k}(x2).)
    HaarBlock inherits HaarString. HaarBlock.hstr lists the highest-L wavelets
        that are required for successful wavelet extrapolation at order p.
        p=1: B_n.hstr = [n]. p=3: B_n.hstr = [2n, 2n+1]. Etc.
class HaarExtrapolate defines wavelet extrapolations for the full interval [0,1].
    [0,1] is segmented into a complete list of HaarBlocks, each with its
        local Taylor series and extrapolation coefficients.
    HaarExtrapolate has access to the full list of wavelet coefficients, so it
        can evaluate f(x2) at the midpoint of every block. This info is
        sufficient to define an Interpolator (from utilities.py).

"""
_haar_ = ['hindex_n', 'hindex_LM', 'haar_sph_value', 'haar_fn_x', 'haar_x123',
          'haar_inverse', 'haar_transform', 'sparse_haar_inverse']
_haartree_ = ['HaarString', 'hs_n_to_hstr', 'hs_list_all_n']
_hextrap_ = ['get_dfk_at_n', 'extrapolate_fn_from_dfk', 'dfk_block',
             'extrapolate_block_n', 'extrapolate_block_newgen',
             'HaarBlock']
__all__ = _haar_ + _haartree_ + _hextrap_


import math
import numpy as np
import gvar # gaussian variables; for vegas

from .utilities import *
# from .basis import Basis


"""
    Spherical wavelet functions:
"""

#Defining an index for the haar wavelets: n = 2**L + M (n=0 for L=-1)
def hindex_n(L, M):
    if L==-1:
        return 0
    # else:
    return 2**L + M

def hindex_LM(n):
    if n==0:
        return [-1, 0]
    # else:
    L = int(math.log2(n))
    M = n - 2**L
    return [L, M]

def haar_sph_value(n, dim=3):
    """Returns the value of h_n(x) where it is nonzero."""
    if n==0:
        return math.sqrt(dim)
    # else:
    [L,M] = hindex_LM(n)
    x1 = 2**(-L) * M
    x2 = 2**(-L) * (M+0.5)
    x3 = 2**(-L) * (M+1)
    y1 = x1**dim
    y2 = x2**dim
    y3 = x3**dim
    A = math.sqrt(dim/(y3 - y1) * (y3-y2)/(y2-y1))
    B = math.sqrt(dim/(y3 - y1) * (y2-y1)/(y3-y2))
    return [A,-B]

def _bin_integral(x1, x2, dim=3):
    """integral of int(x**(d-1) dx) on the interval [x1, x2].

    Needed for haar_transform.
        This is <1|bin_n>, for unnormalized 'bin_n(x) = 1 iff x in [x1,x2]'.
    """
    return (x2**dim - x1**dim)/dim


def _haar_sph_integral(n, dim=3):
    """Returns the integrals int(x**(d-1)dx h_n(x)) on the regions A and B.

    Volume integrals A and B are equal in magnitude.
    """
    if n==0:
        return 1/math.sqrt(dim)
    # else:
    [L,M] = hindex_LM(n)
    x1 = 2**(-L) * M
    x2 = 2**(-L) * (M+0.5)
    x3 = 2**(-L) * (M+1)
    y1 = x1**dim
    y2 = x2**dim
    y3 = x3**dim
    integralAB = math.sqrt((y2-y1)*(y3-y2)/(dim*(y3-y1)))
    return [integralAB,-integralAB]

def _haar_sph_value_LM(L, M, dim=3):
    """Returns the value of h_{L,M}(x) where it is nonzero, for n>0."""
    x1 = 2**(-L) * M
    x2 = 2**(-L) * (M+0.5)
    x3 = 2**(-L) * (M+1)
    y1 = x1**dim
    y2 = x2**dim
    y3 = x3**dim
    A = math.sqrt(dim/(y3 - y1) * (y3-y2)/(y2-y1))
    B = math.sqrt(dim/(y3 - y1) * (y2-y1)/(y3-y2))
    return [A,-B]

def haar_x123(n):
    "Base of support of nth wavelet, including midpoint."
    if n==0:
        return [0, 1]
    else:
        L,M = hindex_LM(n)
        x1 = 2**(-L) * M
        x2 = 2**(-L) * (M+0.5)
        x3 = 2**(-L) * (M+1)
        return [x1, x2, x3]

def _haar_x13(n):
    "Base of support of nth wavelet, not including midpoint."
    if n==0:
        return [0, 1]
    else:
        L,M = hindex_LM(n)
        x1 = 2**(-L) * M
        x3 = 2**(-L) * (M+1)
        return [x1, x3]

def haar_fn_x(n, x, dim=3):
    """Normalized spherical Haar wavelet, n=0,1,2,..."""
    if n==0:
        if 0 < x < 1:
            return haar_sph_value(n, dim=dim)
        elif x==0 or x==1:
            return haar_sph_value(n, dim=dim)
        else:
            return 0
    else:
        x1,x2,x3 = haar_x123(n)
        if x1 < x < x2:
            return haar_sph_value(n, dim=dim)[0]
        elif x2 < x < x3:
            return haar_sph_value(n, dim=dim)[1]
        elif x==x1:
            if x1==0:
                return (haar_sph_value(n, dim=dim)[0])
            else:
                return 0.5*(haar_sph_value(n, dim=dim)[0])
        elif x==x2:
            return 0.5*(haar_sph_value(n, dim=dim)[0]
                        + haar_sph_value(n, dim=dim)[1])
        elif x==x3:
            if x3==1:
                return (haar_sph_value(n, dim=dim)[1])
            else:
                return 0.5*(haar_sph_value(n, dim=dim)[1])
        else:
            return 0


def _h_n_covers_nd(n, nd):
    "'True' if nd is a descendant of 'n', 'False' otherwise."
    if n >= nd:
        return False
    # else:
    while nd > 1:
        nd = math.floor(nd/2)
        if nd==n:
            return True
    return False

def haar_inverse(f_n_list, dim=3):
    """Inverse Haar wavelet transform.

    Input: list of wavelet coefficients, f_n_list = [<0|f>, <1|f>, ..., <nMax|f>].
    Output: histogram of f(x) with nCoeffs bins on [0,1].
    * Note: works for float or gvar valued entries <n|f>.
    """
    nGens = math.ceil(math.log2(len(f_n_list)))
    nCoeffs = 2**nGens
    Lmax = nGens - 1
    f_x = np.array([f_n_list[0]*haar_sph_value(0, dim=dim)]*nCoeffs)
    for L in range(Lmax+1):
        w_L = [] # contributions to f_x from generation L
        for M in range(2**L):
            A_n, mB_n = _haar_sph_value_LM(L, M, dim=dim)
            n = hindex_n(L, M)
            f_n = f_n_list[n]
            bin_size = 2**(Lmax - L) # number of x bins per one-half of a wavelet
            w_L += [A_n*f_n]*bin_size + [mB_n*f_n]*bin_size
        w_L = np.array(w_L)
        f_x += w_L
    return f_x


def haar_transform(f_n_list, dim=3):
    """Change of basis from discretized f_x to spherical wavelets.

    Wavelet index: m = 2**lambda + mu
    Tophat coefficients: f_n = <f|bin_n> with bin_n = [0,0,...,0,1,0,...0]
    <wave_m|f> = sum_n <bin_n|f> <wave_m|bin_n>
    (assuming real-valued functions)
    Wavelet transform has 2**(lambdaMax+1) many coefficients
    Uses np.array for final result, because it is convenient for haar_inverse.
    * Note: works for float or gvar valued entries <n|f>.
    """
    nMax = len(f_n_list) - 1
    power2 = math.ceil(math.log2(nMax+1))
    try:
        assert 2**power2 == nMax+1, "Warning: len(f_n_list) is not a power of 2. Padding with [0,0,...]"
    except AssertionError:
        diff = 2**power2 - 1 - nMax
        f_n_list += diff*[0]
        nMax = len(f_n_list) - 1
    lambdaMax = power2 - 1
    # first, the m=0 wavelet
    fw_m = 0.
    for n in range(nMax+1):
        # x1, x2: bin edges in x for nth bin
        x1 = n/2**power2
        x2 = (n+1)/2**power2
        iprodMN = haar_sph_value(0, dim=dim) * _bin_integral(x1, x2, dim=dim)
        fw_m += f_n_list[n] * iprodMN
    f_wave = [fw_m]
    for lam in range(lambdaMax+1):
        for mu in range(2**lam):
            m = 2**lam + mu
            haarVals = haar_sph_value(m, dim=dim)
            fw_m = 0.
            n_per_halfwavelet = 2**(lambdaMax-lam)
            n_init_A = mu * 2**(power2-lam)
            n_init_B = n_init_A + n_per_halfwavelet
            for n in range(n_init_A, n_init_A + n_per_halfwavelet):
                x1 = n/2**power2
                x2 = (n+1)/2**power2
                iprodMN = haarVals[0] * _bin_integral(x1, x2, dim=dim)
                fw_m += f_n_list[n] * iprodMN
            for n in range(n_init_B, n_init_B + n_per_halfwavelet):
                x1 = n/2**power2
                x2 = (n+1)/2**power2
                iprodMN = haarVals[1] * _bin_integral(x1, x2, dim=dim)
                fw_m += f_n_list[n] * iprodMN
            # append to f_wave:
            f_wave += [fw_m]
    return np.array(f_wave)

def sparse_haar_inverse(hstr, f_n_dict, dim=3, include_hstr=True):
    """Finds discrete f(x) using only n covered by the HaarString 'hstr'.

    Returns list of f(x_i) in the same shape as self.nextGen(level=1).
    Or, if include_hstr==False, the inverse wavelet transform excludes
        the coefficients in hstr. The output is then the same shape as hstr.
    """
    # n=0
    fh_now = [haar_sph_value(0, dim=dim) * f_n_dict[0]]
    n_to_divide = hs_list_all_n(hstr, include_self=include_hstr)
    hs = [1]
    made_change = True
    while made_change and len(n_to_divide)>0:
        fh_update = []
        old_hs = hs
        made_change = False
        # go through every n in HaarString hs:
        # if n is on the 'n_to_divide' list, bisect it and add the values
        #     from f_n_dict[n] to the new left and right bins.
        #     Append these two bins to fh_update.
        # otherwise, pass along the old value of fh_now
        # If one pass through the full list caused any additions to fh_now,
        # do another round. Stop when no further changes are made.
        for ix in range(len(old_hs)):
            n = old_hs[ix]
            if n in n_to_divide:
                A,mB = haar_sph_value(n, dim=dim)
                fh_A = A * f_n_dict[n]
                fh_B = mB * f_n_dict[n]
                fh_update += [fh_now[ix]+fh_A, fh_now[ix]+fh_B]
                hs = _hs_subdivideAt(hs, n)
                n_to_divide.remove(n)
                made_change = True
            else:
                fh_update += [fh_now[ix]]
        # end of this round: hs is already current. Update fh_now:
        fh_now = fh_update
        # Now, fh_now and hs have same shape again.
    return fh_now


"""
    Taylor series and wavelets:
"""

def _mH_k(u0, u2, k, dim=3):
    "int_(u2)^(u2+1) u^(d-1) du (u-u0)**k"
    sum = 0.
    for j in range(dim):
        comb = math.factorial(dim-1)/(math.factorial(j)*math.factorial(dim-1-j))
        diff = (u2 - u0 + 1)**(k+dim-j) - (u2 - u0)**(k+dim-j)
        sum += comb * u0**j * diff / (k + dim - j)
    return sum

def _mD_n_k_x0(n, k, x0, dim=3):
    "<n|(x-x0)**k> / 2**(k*lambda+k)"
    assert n>=1, "Don't use this function for n=0 wavelet."
    [A, mB] = haar_sph_value(n, dim=dim)
    L,M = hindex_LM(n)
    x1,x2,x3 = haar_x123(n)
    u0 = 2**(L+1) * x0
    u1,u2 = x1*2**(L+1), x2*2**(L+1)
    termA = A / 2**(dim*(L+1)) * _mH_k(u0, u1, k, dim=dim)
    termB = mB / 2**(dim*(L+1)) * _mH_k(u0, u2, k, dim=dim)
    return termA + termB


def get_dfk_at_n(f_dn_vec, n, p=3, dim=3):
    """Converts <f|delta n> into k=1,2,... derivatives f^{(k)} centered at x2(n).

    f_dn_vec = [<f|n>, <f|2n>, <f|2n+1>, ...]
    n: basis function index. Note: need n!=0.
    p: polynomial order of derivative expansion (p=3 for cubic)
    dim: for 'dim'-dimensional spherical Haar wavelets

    Returns list of derivatives [f^(1), f^(2), ...,f^(p)] evaluated at x2(n)
        Not including the f(x0) term!
        This is different from the df_p notation in utilities.Interpolator.
    """
    assert len(f_dn_vec)==p, "Need square matrix for _mD..."
    assert math.log2(p+1)%1==0., "Only supports p=1,3,7,..."
    ngens = int(math.log2(p+1))
    nlist = []
    for dl in range(ngens):
        nlist += [2**dl*n + j for j in range(2**dl)]
    x0 = haar_x123(n)[1] # the x2 component
    mD = np.empty([p,p])
    for dn in range(1, p+1):
        for k in range(1, p+1):
            delta_n = nlist[dn-1] #actual value of n
            L,M = hindex_LM(delta_n) #wavelet index
            kLfact = 2**(-k*(L+1)) / math.factorial(k)
            mD[dn-1,k-1] = kLfact*_mD_n_k_x0(delta_n, k, x0, dim=dim)
    wavevec = np.array(f_dn_vec)
    if type(f_dn_vec[0]) is gvar._gvarcore.GVar:
        df_k = gvar.linalg.solve(mD, wavevec)
    else:
        df_k = np.linalg.solve(mD, wavevec)
    return [df for df in df_k]

def extrapolate_fn_from_dfk(dfk_list, x0, n, dim=3):
    """Estimates <f|n> from Taylor series centered at x=x0.

    dfk_list = [df^(1), df^(2), ...] derivatives wrt dimensionless x on [0,1].
    n: basis function index.
    p: polynomial order of derivative expansion (p=3 for cubic)
    dim: for 'dim'-dimensional spherical Haar wavelets

    returns: f_n = <f|n>.
    """
    assert n!=0, "Not valid for n=0."
    p = len(dfk_list)
    f_n = 0.
    L,M = hindex_LM(n)
    for k in range(1,p+1):
        df_k_term = dfk_list[k-1] * 2**(-k*(L+1)) / math.factorial(k)
        f_n += df_k_term * _mD_n_k_x0(n, k, x0, dim=dim)
    return f_n


"""
    Basic HaarString functions.
"""

def _hs_subdivideAt(hstr, n, level=1):
    out = []
    for item in hstr:
        if item==n:
            out += [2**level*n + mu for mu in range(2**level)]
        else:
            out += [item]
    return out

def _hs_nextGen(hstr, level=1):
    out = []
    for n in hstr:
        out += [2**level*n + mu for mu in range(2**level)]
    return out

def hs_list_all_n(hstr, include_self=True, assume_complete=True):
    """Returns complete list of parent wavelets that generate 'hstr'.

    If 'assume_complete', then take 'hstr' to cover the full interval [0,1].
        i.e. every parent 'n' not in hstr branched into [2n, 2n+1].
    So, the full tree can be found by tracing back from just the even entries.

    If not assume_complete, trace back from odd entries as well, avoiding
        double-counting.

    * Not including the n=0 'constant wavelet'.
    """
    # Unbranching algorithm:
    listAll = []
    if assume_complete:
        for n in hstr:
            if include_self:
                listAll += [n]
            while n%2==0:
                n = int(n/2)
                listAll += [n]
        return listAll
    #else:
    for n in hstr:
        if include_self:
            listAll += [n]
        while n>=1:
            if n%2==0:
                n = int(n/2)
            elif n>1:
                n = int((n-1)/2) #don't include n=0
            if n not in listAll:
                listAll += [n]
    return listAll


def _hs_trimAtLevel(hstr, level=1):
    """Ensures that wavelets appear in tuplets of size 2**level."""
    if level==0:
        return hstr
    assert not any([m < 2**level for m in hstr]), "hstr can't be smaller than 'level'."
    out = [1]
    skip_me = []
    try_again = True
    while try_again:
        try_again = False
        newlist = out
        for n in out:
            if n in skip_me:
                continue
            try_again = True
            descendants = [2**level*n + mu for mu in range(2**level)]
            if any([m in hstr for m in descendants]):
                # end this branch in a complete n-tuplet.
                newlist = _hs_subdivideAt(newlist, n, level=level)
                # tell the while loop to skip these values next time.
                skip_me += descendants
            else:
                # if no overlap between out and hstr, subdivide (once):
                newlist = _hs_subdivideAt(newlist, n, level=1)
        # after checking all n from last loop's newlist, save the updates:
        out = newlist
        # while loop terminates once all descendants appear in skip_me.
    return out

def _hs_prevGen(hstr, level=1):
    trim = _hs_trimAtLevel(hstr, level=level)
    out = []
    for n in trim:
        if n%(2**level) == 0:
            out += [int(n/(2**level))]
    return out

def _hs_n_to_hstr_inclusive(n_list):
    """Returns the smallest HaarString that covers all n in n_list."""
    # Subdivide each n in hstr until the descendant
    to_subdivide = []
    for n in n_list:
        if n==0: continue
        m = n
        while m!=1:
            if m%2==0:
                m = int(m/2)
            else:
                m = int((m-1)/2)
            if m not in to_subdivide:
                to_subdivide += [m]
    # return to_subdivide
    # to_subdivide now lists all n that should be divided.
    # want to keep hstr in domain order:
    out = [1]
    try_again = (len(to_subdivide)!=0)
    skip_me = []
    while try_again:
        new = out
        try_again = False
        for n in out:
            if n in skip_me:
                continue
            if n in to_subdivide:
                try_again = True
                new = _hs_subdivideAt(new, n, level=1)
            else:
                skip_me += [n]
        # at end of round, update 'out'
        out = new
        # loop terminates (try_again=False) once a round goes by without any further subdivisions
    return out

def _hs_n_to_hstr_exclusive(n_list):
    """Returns the largest HaarString that does not exceed n_list."""
    hstr = [1]
    try_again = True
    skip_me = []
    while try_again:
        new = hstr
        try_again = False
        for n in hstr:
            if n in skip_me:
                continue
            if all([m in n_list for m in [2*n, 2*n+1]]):
                try_again = True
                new = _hs_subdivideAt(new, n, level=1)
            else:
                skip_me += [n]
        # at end of round, update 'out'
        hstr = new
        # loop terminates (try_again=False) once a round goes by without any further subdivisions
    return hstr

def hs_n_to_hstr(n_list, inclusive=False):
    "inclusive: hstr covers all n in n_list. exclusive: n_list covers all n in hstr."
    if inclusive:
        return _hs_n_to_hstr_inclusive(n_list)
    else:
        return _hs_n_to_hstr_exclusive(n_list)


"""
    Wavelet Extrapolation Methods.
"""

def _hs_block_n(n, level=2):
    "List of the first descendants of n: [n, 2n, 2n+1, 4n, etc.]"
    block = []
    for l in range(level):
        block += [n*2**l +j for j in range(2**l)]
    return block

def _block_hstr(block):
    "The HaarString fragment from the last 'half' of block."
    last_gen_size = int((len(block)+1)/2)
    ignorables = len(block) - last_gen_size
    last_gen = [block[j+ignorables] for j in range(last_gen_size)]
    return last_gen

def _block_descendants(block, level=1):
    last_gen = _block_hstr(block)
    # returns the level=level block for which n is a descendant
    out = []
    for n in last_gen:
        out += [2**level*n + j for j in range(2**level)]
    return out


def _hs_getBlock(hstr, level=2):
    """Organizes last (level) generations in wavelet tree into (cubic) triplets.

    Generic: n-tuplets of size 2**level-1, including:
        n; 2n, 2n+1; 4n, 4n+1, 4n+2, 4n+3; etc.
    Each block in blockList is a flat list of length 2**level-1.

    level=1: linear method. 2: cubic. 3: 7th-order. Etc.

    * _hs_prevGen() uses _hs_trimAtLevel to ensure a non-overlapping blockList.
    """
    toplevel_hstr = _hs_prevGen(hstr, level=level-1)
    #
    blockList = []
    for n in toplevel_hstr:
        block = _hs_block_n(n, level=level)
        blockList += [block]
    return blockList

def dfk_block(f_n_dict, block, dim=3):
    """Calculates Taylor series for this block of wavelets, given <f|n>.

    f_n_dict: a dict or list, f_n_dict[n] = <f|n>.
    block: list of [n, 2n, 2n_1, ...] specific coefficients.
    """
    p = len(block) # polynomial order
    n0 = block[0] # smallest-n wavelet
    # x0 = haar_x123(n0)[1] # wavelet center
    f_dn_vec = [f_n_dict[n] for n in block]
    df_k = get_dfk_at_n(f_dn_vec, n0, p=p, dim=dim)
    return df_k

def _delta_f0_n(df_k_list, n, dim=3):
    """Uses 'volume conservation' to calculate f(x0) given df_k derivatives.

    Equivalently, the inverse wavelet transformation using 'block' in the
        limit of infinitely many descendant coefficients, calculated using
        wavelet extrapolation.

    Assumes Taylor series defined on interval [x1,x3], with derivatives
        evaluated at x0 = x2(n).

    df_k_list: the derivatives [df_1, df_2, ..., df_p] for k=1,2,....

    returns: the "delta_f0" that is added to the inverse wavelet transformation
        generated by the 'n < block' coefficients, i.e.:
        f0 = haar_inverse(smaller_n)(x = x2) + delta_f0.
    """
    x1,x2,x3 = haar_x123(n)
    k_sum = 0.
    k = 1 # first term in df_k is first derivative
    for df_k in df_k_list:
        j_sum = 0.
        for j in range(k+1):
            xterm = (-x2)**j * (x3**(k-j+dim) - x1**(k-j+dim))
            j_sum += xterm/((k-j+dim)*math.factorial(j)*math.factorial(k-j))
        k_sum += df_k * j_sum
        k += 1
    return -dim/(x3**dim - x1**dim) * k_sum


def extrapolate_block_n(f_n_dict, block, nd, dim=3):
    """Calculates <f|n> for a descendant wavelet from 'block'.

    f_n_dict: a dict or list, f_n_dict[n] = <f|n>.
    block: list of [n, 2n, 2n_1, ...] specific coefficients.
    nd: a descendant of n (nd = 2**l * n + j).
    """
    df_k = dfk_block(f_n_dict, block, dim=dim)
    n0 = block[0]
    x0 = haar_x123(n0)[1]
    return extrapolate_fn_from_dfk(df_k, x0, nd, dim=dim)

def extrapolate_block_newgen(f_n_dict, block, level=1, dim=3):
    """Calculates <f|n> for all descendant wavelets at 'level'.

    f_n_dict: a dict or list, f_n_dict[n] = <f|n>.
    block: list of [n, 2n, 2n_1, ...] specific coefficients.

    Output: dict of (n, <f|n>) for all descendants descendants
    """
    df_k = dfk_block(f_n_dict, block, dim=dim)
    n0 = block[0]
    x0 = haar_x123(n0)[1]
    new_n = _block_descendants(block, level=level)
    out = {}
    for n in new_n:
        out[n] = extrapolate_fn_from_dfk(df_k, x0, n, dim=dim)
    return out


class HaarString():
    """Object to track iterative subdivisions of the interval [0,1].

    Each HaarString represents a sequence of wavelets whose bases of support
        cover a specific inverval, e.g. [0,1].
    The smallest (complete) HaarString is defined to be {1}. The next-smallest
        HaarStrings are: {2,3}; {4,5,3}, {2,6,7}; {4,5,6,7}.
    Arbitrary combinations of subdivisions produce a "tree" of wavelets. The
        HaarString tracks only the smallest descendants. The full tree can be
        recovered by dividing all even (n) by 2, recursively, until reaching (1)

    Input:
        hstr: begin with self.hstr = hstr
        power2: or, an initial subdivision into 2**power2 bins
            (self.hstr = [2**(power2-1),...,2**power2-1])

    Object:
        hstr: the current list of final-descendant wavelets
    Class instances largely act like lists, thanks to __add__, __iter__, etc.
    Multiple ways to get self.hstr:
        instance[:], instance(),

    Methods:
    * Updating self.hstr:
        subdivideAt(n): returns new hstr, replacing (n) with descendants
            (n) --> (2n),(2n+1)
        subdivideAll() replaces all wavelets with next-generation descendants
        appendHaarString(other_hstr) concatenates self.hstr and other_hstr.hstr.
    * Not updating self.hstr:
        nextGen, prevGen, list_all_n: return _hs_nextGen(hstr), etc.
        fullList: returns all wavelets in the tree, from (0),(1) to hstr
        updateString(nList): subdivides self.hstr until it includes all
            n in nList.
    """
    def __init__(self, hstr=None, power2=None):
        # power2 parameterizes the initial subdivision into 2**power2 total wavelets
        if hstr is not None:
            self.hstr = hstr
            # check to make sure this self.hstr is valid.
            assert self.checkConsecutive(), "hstr is not position-ordered and complete."
        elif power2 is None or power2<=1:
            self.hstr = [1]
        else:
            self.hstr = [2**(power2-1) + mu
                            for mu in range(2**(power2-1))]

    def __call__(self):
        return self.hstr

    def __getitem__(self, ix):
        return self.hstr[ix]

    def __iter__(self):
        for n in self.hstr:
            yield n

    def __repr__(self):
        return f"{self.hstr}"

    def __add__(self, other):
        return self.hstr + other.hstr

    def __len__(self):
        return len(self.hstr)

    def __contains__(self, n):
        return n in self.hstr

    def index(self, n):
        return self.hstr.index(n)

    def checkOrdered(self):
        "Ensures that the n in self.hstr are position-ordered."
        if 0 in self.hstr:
            print("Error: don't include n=0 in HaarString.")
            return False
        x2_prev = 0.
        for n in self.hstr:
            x2 = haar_x123(n)[1] #midpoint
            if x2 < x2_prev:
                print("Error: entry n={} is out of order.".format(n))
                return False
            # else:
            x2_prev = x2
            #contine
        return True

    def checkConsecutive(self):
        "Ensures that the n in self.hstr are position-ordered and complete."
        if 0 in self.hstr:
            print("Error: don't include n=0 in HaarString.")
            return False
        nprev = self.hstr[0] # leftmost n
        x3_prev = haar_x123(nprev)[0] # leftmost x1
        for n in self.hstr:
            x1,x2,x3 = haar_x123(n)
            if x1 != x3_prev:
                print("Error: x1(n={}) not matching x3(n={}).".format(n, nprev))
                return False
            # else: Update prev <-- current
            nprev = n
            x3_prev = x3
            #continue
        # if the for loop completes, then the HaarString is complete.
        return True

    def GCD(self):
        "Traces back the leftmost and rightmost edges to their common parent 'n'."
        leftpoint = self.hstr[0]
        n = leftpoint
        leftpath = [n]
        while n>1:
            if n%2==0:
                n = int(n/2)
            else:
                n = int((n-1)/2)
            leftpath += [n]
        rightpoint = self.hstr[-1]
        n = rightpoint
        rightpath = [n]
        while n>1:
            if n%2==0:
                n = int(n/2)
            else:
                n = int((n-1)/2)
            rightpath += [n]
        # now traverse one path until 'n' is not in the other list
        for j in range(1,len(leftpath)+1):
            n = leftpath[-j]
            if n in rightpath:
                nBlock = n
            else:
                break
        return nBlock

    def appendHaarString(self, other):
        self.hstr = self.hstr + other.hstr

    def subdivideAt(self, n, level=1):
        self.hstr = _hs_subdivideAt(self.hstr, n, level=level)
        return self.hstr

    def subdivideAll(self, level=1):
        self.hstr = _hs_nextGen(self.hstr, level=level)
        return self.hstr

    def nextGen(self, level=1):
        return _hs_nextGen(self.hstr, level=level)

    def prevGen(self, level=1):
        return _hs_prevGen(self.hstr, level=level)

    def list_all_n(self, include_self=True):
        # return complete list of parent wavelets and the current haarstring
        return hs_list_all_n(self.hstr, include_self=include_self)

    def regions_x(self):
        "List of boundary points on the interval [0,1], in style of uiList from Basis."
        uiList = [haar_x123(self.hstr[0])[0]]
        for n in self.hstr:
            x3 = haar_x123(n)[2]
            uiList += [x3]
        return uiList

    def midpoints_x(self):
        "List of midpoints of haar(n) in self.hstr."
        u0List = []
        for n in self.hstr:
            x2 = haar_x123(n)[1]
            u0List += [x2]
        return u0List

    def x_to_n(self, x):
        "Maps the point x to the wavelet base of support 'n'."
        uiList = self.regions_x()
        if x < uiList[0] or x > uiList[-1]:
            print("Error: x outside range of hstr base of support.")
            return False
        #else:
        ix = 0
        for x3 in uiList[1:]:
            if x <= x3:
                # if x==x3, assign x to the lower-n wavelet.
                return self.hstr[ix]
            ix += 1

    def sparse_inverse(self, f_n_dict, dim=3, include_hstr=True):
        """Finds discrete f(x) using only n covered by the Haar tree.

        Returns list of f(x_i) in the same shape as self.nextGen(level=1).
        Or, if include_hstr==False, the inverse wavelet transform excludes
            the coefficients in self.hstr. The output is then the same
            shape as self.hstr.
        """
        return sparse_haar_inverse(self.hstr, f_n_dict, dim=dim,
                                   include_hstr=include_hstr)

    def blockHeaders(self, level=2):
        # hstr of top-level [n] for each block
        toplevel_hstr = _hs_prevGen(self.hstr, level=level-1)
        return toplevel_hstr

    def getBlocks(self, level=2):
        return _hs_getBlock(self.hstr, level=level)

    def blockRegions_x(self, level=2):
        "List of boundary points on the interval [0,1], in style of uiList from Basis."
        block_hstr = self.blockHeaders(level=level)
        uiList = [0]
        for n in block_hstr:
            x1,x2,x3 = haar_x123(n)
            uiList += [x3]
        return uiList

    def coveringBlock_n(self, nd, level=2):
        "Returns the block index 'n' for which nd is a descendant"
        block_hstr = [block[0] for block in self.getBlock(level=level)]
        x_n = haar_x123(nd)[1] #x coordinate of center of n
        len_list = len(block_hstr)
        for j in range(1, len_list+1):
            block_n = block_hstr[len_list-j]
            x1 = haar_x123(block_n)[0]
            if x1 < x_n:
                return block_n
        return 1 # shouldn't come to this



class HaarBlock(HaarString):
    """Applies wavelet extrapolation to the descendants of wavelet 'n'.

    Each integer in the string labels a wavelet lambda,mu in generation lambda:
        n = 2**lambda + mu
    A wavelet (n) is subdivided with descendants (2n) and (2n+1);
        then (4n), (4n+1), (4n+2), (4n+3) in the next generation; etc.

    Input:
        n: the largest parent wavelet index.
        p_order: polynomial order for Taylor series extrapolation

    """
    def __init__(self, n, p_order=3, dim=3):
        # self.n = n
        self.p_order = p_order
        self.dim = dim
        self.depth = int(math.log2(p_order+1))
        if self.depth==0:
            self.depth = 1
        self.block_n = _hs_block_n(n, level=self.depth) # complete list of n in the block
        hstr = _block_hstr(self.block_n)
        HaarString.__init__(self, hstr=hstr)

    def get_dfk(self, f_n_dict):
        return dfk_block(f_n_dict, self.block_n, dim=self.dim)

    def df_p(self, f_n_dict):
        """Returns list of [delta_f0, df_1, df_2, ... df_p].

        delta_f0 is the contribution to f(x2) from <f|n> and its descendants,
            in the limit of infinitely many n' > n.
        Equivalently, it is the difference between f(x) and the block-level
            inverse wavelet transformation.
            (block-level meaning constant across this block).
        """
        df_k = dfk_block(f_n_dict, self.block_n, dim=self.dim)
        delta_f0 = _delta_f0_n(df_k, self.block_n[0], dim=self.dim)
        return [delta_f0] + df_k

    def extrapolate_n(self, f_n_dict, nd):
        return extrapolate_block_n(f_n_dict, self.block_n, nd, dim=self.dim)

    def extrapolate_newgen(self, f_n_dict, level=1):
        return extrapolate_block_newgen(f_n_dict, self.block_n, level=level, dim=self.dim)




#
