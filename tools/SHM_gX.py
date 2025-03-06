"""Wavelet-harmonic projection of a Standard Halo Model (SHM) DM distribution.

In the lab frame, the SHM velocity distribution is azimuthally symmetric,
with 'z' aligned with the (instantaneous) Earth velocity w.r.t. the galactic
center. Here, the Earth velocity is an annually-periodic function of time.

    Following Lewin & Smith (1996), "Review of mathematics, numerical factors,
and corrections for dark matter experiments based on elastic nuclear recoil"
with updated numerical values from 2105.00599 (D. Baxter et. al., "Recommended
conventions for reporting results from direct dark matter searches")

Default values: (all velocities in km/s)
- local standard of rest velocity: (0, 238., 0)
- galactic escape speed: 544.
- solar peculiar velocity: (11.1, 12.2, 7.3)
- average Earth speed: 29.8
These values are used in vEt_precise(date) to recover the instantaneous Earth
    speed in the galactic rest frame, as a function of 'date'.
* Example: vE(t) maximum at 2024 May 30, approx. 05:28:00: vE = 266.2 km/s

This code uses sys so that variables can be specified using the command line.
Variables:
    n_days (float): time since last maximum in vE(t), measured in days (24h)
    l_max (int): largest value of 'ell' to calculate.
    power2 (int): input for ExtrapolateF.
        Evaluates the initial set of <f|n> using a grid of 2**power2 bins.
        Further <f|nlm> integrals are completed using the refine_lm() method.

The 'z' direction is defined to be parallel to the instantaneous Earth velocity,
so the SHM is always azimuthally symmetric. All <f|nlm> with nonzero 'm' vanish,
and the projection integral is 2d (over 'v' and 'cos theta').
As a result, the numerical integration is very fast.

With sys, can run main(n_days,l_max,power2) as:

    python3 SHM_gX.py n_days l_max power2

e.g. python3 SHM_gX.py 12.5 36 9 for the 't=12.5 days' velocity distribution,
going up to ell=36, with an initial grid of 512 intervals.
Or, for annual modulation, can generate many snapshots:

    for i in {0..25}; do python3 -u SHM_gX.py $(($i * 14)) l_max power2; done

All snapshots in time are saved to the same hdf5 file, under model names
determined by n_days. (Using n_days if it is an integer, or its first 4 digits.)

Also including an alternative program, alt_vE(vE_km_s,l_max,power2), with
the input 'vE_km_s' providing the Earth speed in km/s. This provides a simpler
accurate interpolation for annual modulation analyses. E.g:

    for i in {237..267}; do python3 -u SHM_gX.py $i l_max power2; done

(Expecting linear interpolation for gX(vE) to be more accurate than linear
interpolation on gX(t).)

Each version of the SHM is saved with a modelName SHM_d{n_days} for main(),
or SHM_v{vE_km_s} for alt_vE(). All of these versions of gX(t) can be saved
to the same hdf5 file.
    If a model of this name already exists in the hdf5 file, then any new
coefficients will be saved to a database 'fnlm__2', 'fnlm__3', etc., using
a naming scheme chosen by Portfolio.


Normalization: the velocity distribution gX has units of inverse velocity cubed:
    [gX] = [VUNIT]**(-3)
By default, units.py reports all velocities (e.g. km_s) in units of c, i.e.
    km_s = 2.99792e5**(-1)

"""
import math
import numpy as np
import numba
import scipy.special as spf
import vegas # numeric integration
import gvar # gaussian variables; for vegas
import time
import datetime as dts #for calendar functions
# import quaternionic # For rotations
# import spherical #For Wigner D matrix
import h5py # database format for mathcalI arrays
import sys

# sys.path.insert(0,'../') #load the local version of vsdm

from earthspeed import vEt
import vsdm
from vsdm.units import *
from vsdm.utilities import *



"""Parameters: """
VMAX = 820*km_s
vEsc = 544*km_s
vDisp = 238*km_s
# Formula for normalization constants:
sqrtpi = math.sqrt(math.pi)
beta = vEsc/vDisp
vDisp_0 = vDisp/VMAX
N0 = ((sqrtpi*vDisp)**3
      * (spf.erf(beta) - 2/sqrtpi * beta * math.exp(-beta**2)))

@numba.jit("double(double,double,double)", nopython=True)
def gSHM_sph(vE, vDM_r, costheta):
    # vE: Earth speed in galactic rest frame, always taken to be in z direction
    # vDM_r, costheta: vector in spherical coordinates (r, theta, phi)
    relspeed_2 = vE**2 + vDM_r**2 + 2*vE*vDM_r * costheta
    if vEsc**2 < relspeed_2:
        return 0
    return np.exp(-relspeed_2/vDisp**2) / N0


def gSHM(vDisp, vEsc, vE, vDMsph):
    # vE: Earth speed in galactic rest frame, always taken to be in z direction
    # vDMsph: vector in spherical coordinates (r, theta, phi)
    (v, theta, phi) = vDMsph
    vx = v*math.sin(theta) * math.cos(phi)
    vy = v*math.sin(theta) * math.sin(phi)
    vz = v*math.cos(theta)
    relspeed = math.sqrt(vx**2 + vy**2 + (vz + vE)**2)
    # relative speed wrt DM wind velocity, vW = (0, 0, -vE) in Cartesian
    if vEsc < relspeed:
        return 0
    esc0 = vEsc/vDisp
    sqrtpi = math.sqrt(math.pi)
    N0 = ((sqrtpi*vDisp)**3
          * (spf.erf(esc0) - 2/sqrtpi * esc0 * math.exp(-esc0**2)))
    return math.exp(-(relspeed/vDisp)**2) / N0



def tilde_E(vDisp_0, beta):
    # for getting the energy of the distribution
    sqrtpi = math.sqrt(math.pi)
    d1 = spf.erf(2**0.5 * beta) - 2**1.5/sqrtpi * beta * np.exp(-2* beta**2)
    d2 = spf.erf(beta) - 2/sqrtpi * beta * np.exp(-beta**2)
    return d1 / d2**2 / (2**1.5) / (sqrtpi * vDisp_0)**3



def main(n_days, l_max, power2):
    #n_days: time elapsed since the last maximum in vE(t), measured in days
    # (can be float-valued)
    #reference maximum: 2024 May 30, 5:28:00 UTC
    date_ref = dts.datetime(2024, 5, 30, 5, 28, 0)
    date = date_ref + dts.timedelta(days=n_days)
    vE = np.linalg.norm(vEt(date))

    @numba.jit("double(double[:])", nopython=True)
    def gModel(vSph):
        # units of [velocity]**(-3)
        [v_r, v_theta, v_phi] = vSph
        costheta = np.cos(v_theta)
        return gSHM_sph(vE, v_r, costheta)

    gModel.is_gaussian = False
    gModel.phi_symmetric = True
    energy = tilde_E(vDisp_0, beta)
    print("energy: ", energy)

    bdict = dict(u0=VMAX, type='wavelet', uMax=VMAX)
    bdict['n_days'] = n_days #save this info
    mname = "SHM_d{:.4g}".format(n_days) #string model name for saving HDF5
    epsilon = 1e-5
    atol_E = energy * epsilon
    atol_f = 0.05 * math.sqrt(atol_E)
    integ_params = dict(method='gquad', verbose=True,
                        atol_f=0.01*atol_f,
                        rtol_f=epsilon)
    t0 = time.time()
    wave_extp = vsdm.WaveletFnlm(bdict, gModel, integ_params,
                                 power2_lm={}, p_order=3,
                                 epsilon=epsilon,
                                 atol_f2norm=atol_E,
                                 atol_fnlm=atol_f,
                                 max_depth=5,
                                 refine_at_init=False,
                                 f_type='gX',
                                 csvsave_name=None,
                                 use_gvar=True)

    lm_list = [(l, 0) for l in range(l_max+1)]
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
    hdf5name = 'out/gX_SHM.hdf5'
    wave_extp.writeFnlm(hdf5name, mname, use_gvar=True)
    # wave_extp.add_data(hdf5name, 'f2_m4', use_gvar=True)
    print("Integration times:")
    for lm,t in t0_lm.items():
        print("\t", lm, t)
    print("Total evaluation time:", time.time()-t0)


def alt_vE(vE_km_s, l_max, power2):
    #vE: lab speed in galactic frame.
    # SHM: vE varies from 237.2 to 266.2
    vE = vE_km_s * km_s

    @numba.jit("double(double[:])", nopython=True)
    def gModel(vSph):
        # units of [velocity]**(-3)
        [v_r, v_theta, v_phi] = vSph
        costheta = np.cos(v_theta)
        return gSHM_sph(vE, v_r, costheta)

    gModel.is_gaussian = False
    gModel.phi_symmetric = True
    energy = tilde_E(vDisp_0, beta)
    print("energy: ", energy)

    bdict = dict(u0=VMAX, type='wavelet', uMax=VMAX)
    bdict['vE'] = vE #save this info
    mname = "SHM_v{:.4g}".format(vE_km_s) #string model name for saving HDF5
    epsilon = 1e-5
    atol_E = energy * epsilon
    atol_f = 0.05 * math.sqrt(atol_E)
    integ_params = dict(method='gquad', verbose=True,
                        atol_f=0.002*atol_f,
                        rtol_f=epsilon)
    # save_csv = None
    save_csv = 'out/' + mname + '.csv'
    t0 = time.time()
    wave_extp = vsdm.WaveletFnlm(bdict, gModel, integ_params,
                                 power2_lm={}, p_order=3,
                                 epsilon=epsilon,
                                 atol_f2norm=atol_E,
                                 atol_fnlm=atol_f,
                                 max_depth=5,
                                 refine_at_init=False,
                                 f_type='gX',
                                 csvsave_name=save_csv,
                                 use_gvar=True)

    lm_list = [(l, 0) for l in range(l_max+1)]
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
    hdf5name = 'out/gX_SHM.hdf5'
    wave_extp.writeFnlm(hdf5name, mname, use_gvar=True)
    # wave_extp.add_data(hdf5name, 'f2_m4', use_gvar=True)
    print("Integration times:")
    for lm,t in t0_lm.items():
        print("\t", lm, t)
    print("Total evaluation time:", time.time()-t0)



# main(float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
alt_vE(float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))

#
