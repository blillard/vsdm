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
    power2 (int): input for ExtrapolateFnlm.
        Evaluates the initial set of <f|n> using a grid of 2**power2 bins.
        Further <f|nlm> integrals are completed using the refine_lm() method.

The 'z' direction is defined to be parallel to the instantaneous Earth velocity,
so the SHM is always azimuthally symmetric. All <f|nlm> with nonzero 'm' vanish,
and the projection integral is 2d (over 'v' and 'cos theta').
As a result, the numerical integration is very fast.

With sys, can run this code as: python3 SHM_gX.py n_days l_max power2, e.g.:

    python3 SHM_gX.py 0 36 9

for the 't=0' velocity distribution, going up to ell=36, with an initial grid
of 512 intervals. Or, for annual modulation, can generate many snapshots:

    for i in {0..25}; do python3 -u SHM_gX.py $(($i * 14)) l_max power2; done

All snapshots in time are saved to the same hdf5 file, under model names
determined by n_days. (Using n_days if it is an integer, or its first 4 digits.)
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
# import csv # file IO for projectFnlm
# import os.path
import h5py # database format for mathcalI arrays
import importlib
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as clr

sys.path.insert(0,'../') #load the local version of vsdm
# sys.path.append('../')

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



def vEt_simple(t_yr):
    # from Lewin & Smith, 1996, approximation Eq(3.6)
    # time in years 'from (approximately) March 2nd'
    # returns vE in km_s.
    # L&S: (244 + 15*math.sin(2*math.pi*t_yr))*km_s
    # with update from 2105.00599:
    return (252. + 15*math.sin(2*math.pi*t_yr))*km_s

def vEt_precise(date):
    # from Lewin & Smith, 1996, Appx B
    # with updated numeric values from 2105.00599
    # date: a datetime object (year, month, day, hour=...)
    # returns vE in km_s
    uR = (0, 238*km_s, 0) #local group velocity. Most uncertain. 1996: (230)
    uS = (11.1*km_s, 12.2*km_s, 7.3*km_s) # Sun wrt local group. 1996: (9, 12, 7)
    els = 0.016722 # ellipticity of Earth orbit
    # angular constants (all in degrees)
    lam0 = 13. # longitude of orbit minor axis
    bX = -5.5303
    bY = 59.575
    bZ = 29.812
    lX = 266.141
    lY = -13.3485
    lZ = 179.3212
    # time reference: noon UTC, 31 Dec 1999
    datetime0 = dts.datetime.fromisoformat('1999-12-31T12:00:00')
    difftime = date - datetime0
    nDays = difftime.days + difftime.seconds/(24*3600)
    L = (280.460 + 0.9856474*nDays) % 360 # (degrees)
    g = (357.528 + 0.9856003*nDays) % 360# (degrees)
    # ecliptic longitude:
    lam = L + 1.915*math.sin(g * math.pi/180) + 0.020*math.sin(2*g * math.pi/180)
    uEl = 29.79*km_s * (1 - els*math.sin((lam - lam0)*math.pi/180))
    uEx = uEl * math.cos(bX * math.pi/180) * math.sin((lam - lX)*math.pi/180)
    uEy = uEl * math.cos(bY * math.pi/180) * math.sin((lam - lY)*math.pi/180)
    uEz = uEl * math.cos(bZ * math.pi/180) * math.sin((lam - lZ)*math.pi/180)
    vEabs = math.sqrt((uR[0]+uS[0]+uEx)**2
                      + (uR[1]+uS[1]+uEy)**2
                      + (uR[2]+uS[2]+uEz)**2)
    return vEabs

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
    vE = vEt_precise(date)

    @numba.jit("double(double[:])", nopython=True)
    def gModel(vSph):
        # dimensionless "g_tilde = VMAX**3 * gX"
        [v_r, v_theta, v_phi] = vSph
        costheta = np.cos(v_theta)
        return VMAX**3 * gSHM_sph(vE, v_r, costheta)

    gModel.is_gaussian = False
    gModel.phi_symmetric = True
    energy = tilde_E(vDisp_0, beta)
    print("energy: ", energy)

    bdict = dict(u0=VMAX, type='wavelet', uMax=VMAX)
    bdict['n_days'] = n_days #save this info
    mname = "SHM_{:.4g}".format(n_days) #string model name for saving HDF5
    epsilon = 1e-5
    atol_E = energy * epsilon
    atol_f = 0.05 * math.sqrt(atol_E)
    integ_params = dict(method='gquad', verbose=True,
                        atol_f=0.01*atol_f,
                        rtol_f=epsilon)
    t0 = time.time()
    wave_extp = vsdm.ExtrapolateFnlm(bdict, gModel, integ_params,
                                     power2_lm={}, p_order=3,
                                     epsilon=epsilon,
                                     atol_energy=atol_E,
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



main(float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))

#
