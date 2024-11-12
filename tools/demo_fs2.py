"""Example code for generating <f|nlm> coefficients for a momentum form factor.
The physical system is a particle in a rectangular box, with dimensions
    Lx, Ly, Lz = (4, 7, 10) [in units of Bohr radius] for the primary example.
From the ground state (nx, ny, nz) = (1, 1, 1), the first excited state is
    n_i = (1, 1, 2), followed by (1, 2, 1), (1, 1, 3), (1, 2, 2), (1, 1, 4)...
The box has three Z2 symmetries, including central inversion: so, <f|nlm> = 0
for odd l. Aligning the edges of the box with the z and x directions, there is
a Z2 azimuthal symmetry and a Z2 parity, so <f|nlm> = 0 for odd or negative m.

This example code takes three arguments: l_min, l_max, and power2, e.g.:

    python3 demo_fs2.py 0 12 9

where power2 specifies the size of the initial grid for ExtrapolateF
(nCoeffs = 2**power2), and l_min and l_max specify which (l,m) harmonic modes
should be calculated. For each l=l_min,l_min+2,...,l_max, only nonnegative even
m=0,2,...,l are included in the list. (Odd l_min or l_max inputs are rounded
down to even l-1.)
    If only two inputs are supplied, the program runs with l_min = 0: e.g.

    python3 demo_fs2.py 12 9

for l_min=0, l_max=12, power2=9.
    By default, the <f|nlm> are saved to an hdf5 file under the model name
'box_4_7_10'. Running this program multiple times (e.g. with different l ranges)
will invoke the Portfolio naming scheme for conflicting database names,
e.g. 'fnlm', 'fnlm__2', 'fnlm__3', etc. These coefficients can be imported
by running Fnlm.importFnlm() once for each dataset, using non-default
d_fnlm and lm_ix values. E.g: d_fnlm='fnlm__2', lm_ix='lm_index__2', etc.

    Alternatively, the model name can be changed through the command line, by
adding a fourth (string-valued) input:

    python3 demo_fs2.py 14 36 9 box_4_7_10_largeL

Subsequently, the saved coefficients can be imported using importFnlm() with
the default values of d_fnlm and lm_ix.

The default hdf5 file name is 'out/demo_fs2.hdf5'. As the wavelet coefficients
<f|nlm> are evaluated, they are saved to 'out/demo_fs2.csv'. The CSV save mode
is well suited to parallel computation: new coefficients are simply appended
to the previously saved list, so there is no need to change the file name or
any model name if more coefficients are added later. However, each physical
model needs its own CSV file (likewise if the basis function parameters are
changed, e.g. for different values of QMAX).
    To turn off the CSV save option, set 'csvsave_name=None' in ExtrapolateF.
"""
import math
import numpy as np
import numba
# import scipy.special as spf
# import vegas # numeric integration
# import gvar # gaussian variables; for vegas
import time
# import quaternionic # For rotations
# import spherical #For Wigner D matrix
import h5py
import sys

sys.path.insert(0,'../') #load the local version of vsdm

import vsdm
from vsdm.units import *
from vsdm.utilities import *

@numba.jit("double(uint32,double)", nopython=True)
def fj2(nj, qLj):
    if qLj==0:
        if nj==1:
            return 1
        else:
            return 0
    qlp = np.abs(qLj)/np.pi
    # mathsinc(x) = np.sinc(x/pi)
    s_minus = np.sinc(0.5*(qlp - nj + 1))/(1 + (nj-1)/qlp)
    s_plus = np.sinc(0.5*(qlp - nj - 1))/(1 + (nj+1)/qlp)
    return (s_minus + s_plus)**2

### MOMENTUM DISTRIBUTION EXAMPLES

# Long thin box limit: assuming that Lz > Lx,Ly,
# so the lowest excited states are nz=2, nz=3, with nx=ny=1.

@numba.jit("double(double[:],uint32,double[:])", nopython=True)
def fs2_nz(Lvec, nz, q_xyz):
    # q: the DM particle velocity (cartesian, lab frame)
    # L: the dimensions of the box
    # nz = 2, 3, 4... The final state. (n=1 defined as ground state)
    # fs2 is dimensionless
    # note: np.sinc(x/pi) = sin(x) / (x). Included in defs. of qL below
    [Lx, Ly, Lz] = Lvec
    [qx, qy, qz] = q_xyz
    qLx = Lx*qx
    qLy = Ly*qy
    qLz = Lz*qz
#     qL = qLx + qLy + qLz
    fx2 = fj2(1, qLx)
    fy2 = fj2(1, qLy)
    fz2 = fj2(nz, qLz)
    return fx2*fy2*fz2

@numba.jit("double(double[:],int64[:],double[:])", nopython=True)
def fs2_nxyz(Lvec, n_xyz, q_xyz):
    # q: the DM particle velocity (cartesian, lab frame)
    # L: the dimensions of the box
    # nz = 2, 3, 4... The final state. (n=1 defined as ground state)
    # fs2 is dimensionless
    # note: np.sinc(x/pi) = sin(x) / (x). Included in defs. of qL below
    [Lx, Ly, Lz] = Lvec
    [qx, qy, qz] = q_xyz
    [nx, ny, nz] = n_xyz
    qLx = Lx*qx
    qLy = Ly*qy
    qLz = Lz*qz
    fx2 = fj2(nx, qLx)
    fy2 = fj2(ny, qLy)
    fz2 = fj2(nz, qLz)
    return fx2*fy2*fz2

@numba.jit("double(uint32,double)", nopython=True)
def DeltaE_nz(nz, Lz):
    # for nx=ny=1 final states, in units of [q**2]/mElec
    return 0.5*math.pi**2 / mElec * (nz**2 - 1)/Lz**2

@numba.jit("double(double[:])", nopython=True)
def fs2_m1(qDMsph):
    # Taking Lz = 10*a0 for all examples, with DeltaE = 4.0285 eV for nz=2.
    Lx = 4/qBohr
    Ly = 4/qBohr
    Lz = 10/qBohr
    Lvec = np.array([Lx, Ly, Lz])
    [q, theta, phi] = qDMsph
    q_xyz = np.array([q * np.sin(theta) * np.cos(phi),
                      q * np.sin(theta) * np.sin(phi),
                      q * np.cos(theta)])
    return fs2_nz(Lvec, 2, q_xyz)

@numba.jit("double(double[:])", nopython=True)
def fs2_m2(qDMsph):
    # Taking Lz = 10*a0 for all examples, with DeltaE = 4.0285 eV for nz=2.
    Lx = 8/qBohr
    Ly = 8/qBohr
    Lz = 10/qBohr
    Lvec = np.array([Lx, Ly, Lz])
    [q, theta, phi] = qDMsph
    q_xyz = np.array([q * np.sin(theta) * np.cos(phi),
                      q * np.sin(theta) * np.sin(phi),
                      q * np.cos(theta)])
    return fs2_nz(Lvec, 2, q_xyz)

@numba.jit("double(double[:])", nopython=True)
def fs2_m3(qDMsph):
    # Taking Lz = 10*a0 for all examples, with DeltaE = 4.0285 eV for nz=2.
    Lx = 4/qBohr
    Ly = 7/qBohr
    Lz = 10/qBohr
    Lvec = np.array([Lx, Ly, Lz])
    [q, theta, phi] = qDMsph
    q_xyz = np.array([q * np.sin(theta) * np.cos(phi),
                      q * np.sin(theta) * np.sin(phi),
                      q * np.cos(theta)])
    return fs2_nz(Lvec, 2, q_xyz)

@numba.jit("double(double[:])", nopython=True)
def fs2_model4_cart(q_xyz):
    Lvec = np.array([4/qBohr, 7/qBohr, 10/qBohr])
    return fs2_nz(Lvec, 2, q_xyz)

# Cartesian version of fs2 for higher excited state:
@numba.jit("double(double[:])", nopython=True)
def fs2_model4_cart_alt(q_xyz):
    Lvec = np.array([4/qBohr, 7/qBohr, 10/qBohr])
    n_xyz = np.array([3, 2, 1])
    return fs2_nxyz(Lvec, n_xyz, q_xyz)


@numba.jit("double(double[:])", nopython=True)
def fs2_model4(qSph):
    [q, theta, phi] = qSph
    qx = q*math.sin(theta) * math.cos(phi)
    qy = q*math.sin(theta) * math.sin(phi)
    qz = q*math.cos(theta)
    q_xyz = np.array([qx, qy, qz])
    Lvec = np.array([4/qBohr, 7/qBohr, 10/qBohr])
    return fs2_nz(Lvec, 2, q_xyz)
fs2_model4.is_gaussian = False
fs2_model4.z_even = True
fs2_model4.phi_even = True
fs2_model4.phi_cyclic = 2

@numba.jit("double(double[:])", nopython=True)
def fs2_model4_alt(qSph):
    [q, theta, phi] = qSph
    qx = q*math.sin(theta) * math.cos(phi)
    qy = q*math.sin(theta) * math.sin(phi)
    qz = q*math.cos(theta)
    q_xyz = np.array([qx, qy, qz])
    return fs2_model4_cart_alt(q_xyz)
fs2_model4_alt.is_gaussian = False
fs2_model4_alt.z_even = True
fs2_model4_alt.phi_even = True
fs2_model4_alt.phi_cyclic = 2




def main(l_min, l_max, power2, modelname='box_4_7_10'):
    QMAX = 10*qBohr # Global value for q0=qMax for wavelets
    basisQ = dict(u0=QMAX, type='wavelet', uMax=QMAX)
    """
    Note: here the precision goal 'atol_f2norm' is set based on an estimate of
        the total L2 norm, approximately 3.5e-4 for the nz=2 excited state.
    """
    use_alt_model = False
    if use_alt_model:
        fs2_model = fs2_model4_alt
        csvname = 'out/demo_fs2_alt.csv'
        modelname += '_alt'
    else:
        fs2_model = fs2_model4
        csvname = 'out/demo_fs2.csv'
    energy = 3.5e-4
    epsilon = 1e-6
    atol_E = energy * epsilon
    atol_f = 0.05 * math.sqrt(atol_E)
    integ_params = dict(method='gquad', verbose=True,
                        atol_f=0.05*atol_f,
                        rtol_f=epsilon)
    t0 = time.time()

    wave_extp = vsdm.WaveletFnlm(basisQ, fs2_model, integ_params,
                                 power2_lm={}, p_order=3,
                                 epsilon=epsilon,
                                 atol_f2norm=atol_E,
                                 atol_fnlm=atol_f,
                                 max_depth=5,
                                 refine_at_init=False,
                                 f_type='fs2',
                                 csvsave_name=csvname,
                                 use_gvar=True)
    l2min = int(l_min/2)
    l2max = int(l_max/2)
    lm_list = [(2*i, 2*j) for i in range(l2min, l2max+1) for j in range(i+1)]
    # Start by initializing all lm in list...
    t0_lm = {}
    for lm in lm_list:
        t_lm = time.time()
        wave_extp.initialize_lm(lm, power2)
        t0_lm[lm] = time.time() - t_lm
        print("Integration time:", t0_lm[lm])
    # Step 2: if necessary, refine...
    for lm in lm_list:
        t_lm = time.time()
        wave_extp.refine_lm(lm, max_depth=3)
        t0_lm[lm] += time.time() - t_lm

    # Save to hdf5, then read off the integration times.
    hdf5name = 'out/demo_fs2.hdf5'
    # For this example, using 'box_4_7_10' for the hdf5 model name
    wave_extp.writeFnlm(hdf5name, modelname, use_gvar=True)
    print("Integration times:")
    for lm,t in t0_lm.items():
        print("\t", lm, t)
    print("Total evaluation time:", time.time()-t0)


# len(sys.argv) is one more than the number of inputs
if len(sys.argv)==3:
    main(0, int(sys.argv[1]), int(sys.argv[2]))
elif len(sys.argv)==4:
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
elif len(sys.argv)==5:
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]),
         modelname=sys.argv[4])
else:
    print("Error: wrong number of variables")

# python3 demo_fs2.py 32 36 10
#
