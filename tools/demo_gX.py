"""Wavelet-harmonic projection of a four-gaussian gX model.

Normalization: the velocity distribution gX has units of inverse velocity cubed:
    [gX] = [VUNIT]**(-3)
By default, units.py reports all velocities (e.g. km_s) in units of c, i.e.
    km_s = 2.99792e5**(-1)

This example code generates CSV files of all <gX|nlm> coefficients up to and
including n=n_max and l=l_max, for all m = -l, ..., l-1, l.
Saved to: out/gX_model4.csv
It can be run using the terminal command:
    python3 demo_gX.py n_max l_max
for some specified n_max and l_max.

Note: this example uses EvaluateFnlm, which is faster than WaveletFnlm
(especially for Gaussian-type functions). The integration precision is
generally much better when using WaveletFnlm, however.
"""
import math
import numpy as np
# import numba
# import scipy.special as spf
# import vegas # numeric integration
# import gvar # gaussian variables; for vegas
import time
# import datetime as dts #for calendar functions
# import quaternionic # For rotations
# import spherical #For Wigner D matrix
# import h5py # database format for mathcalI arrays
import sys

# sys.path.insert(0,'../') #load the local version of vsdm

import vsdm
from vsdm.units import *
from vsdm.utilities import *



"""Parameters for 'Model 4': """

def gaussian_stream_sum(ci, vWsph_i, sigma_i):
    # Arguments: lists of amplitudes gi, dispersions v0_i, and
    #     lab-frame DM wind vectors vWsph_i, in spherical coordinates
    gvec_list = []
    for i in range(len(gi)):
        gaus = (ci[i], vWsph_i[i], sigma_i[i])
        gvec_list += [gaus]
    return gvec_list

# Model 4: a bunch of streams, not symmetric.
# Including a halo component without vEsc.
v0_main = 220*km_s / np.sqrt(2)
v0_a = 70*km_s / np.sqrt(2)
v0_b = 50*km_s / np.sqrt(2)
v0_c = 25*km_s / np.sqrt(2)
vX_main = vsdm.cart_to_sph((0, 0, -230*km_s))
vX_a = vsdm.cart_to_sph((80*km_s, 0, -80*km_s))
vX_b = vsdm.cart_to_sph((-120*km_s, -250*km_s, -150*km_s))
vX_c = vsdm.cart_to_sph((50*km_s, 30*km_s, -400*km_s))
sigma_i = [v0_main, v0_a, v0_b, v0_c]
vWsph_i = [vX_main, vX_a, vX_b, vX_c]
gi = [0.4, 0.3, 0.2, 0.1]
gvec_list_4 = gaussian_stream_sum(gi, vWsph_i, sigma_i)

VMAX = 960.*km_s # Global value for v0=vMax for wavelets
basisV = dict(u0=VMAX, type='wavelet', uMax=VMAX)
gXmodel_4 = vsdm.Gnli(basisV, gvec_list_4)


def main(n_max, l_max):
    gXmodel_4.is_gaussian = True
    power2 = int(np.ceil(np.log2(n_max+1)))
    # csvout = None
    csvout = 'out/gX_new.csv'

    energy = gXmodel_4.norm_energy()
    print("energy: ", energy)

    bdict = dict(u0=VMAX, type='wavelet', uMax=VMAX)
    epsilon = 1e-8
    atol_E = energy * epsilon
    atol_f = 0.05 * math.sqrt(atol_E)
    integ_params = dict(method='gquad', verbose=True,
                        atol_f=0.01*atol_f,
                        rtol_f=epsilon)
    t0 = time.time()
    nlmlist = [(n, l, m) for n in range(n_max+1)
               for l in range(l_max+1) for m in range(-l, l+1)]
    wave_extp = vsdm.EvaluateFnlm(bdict, gXmodel_4, integ_params,
                                 nlmlist=nlmlist,
                                 f_type='gX',
                                 csvsave_name=csvout,
                                 use_gvar=True)


    # Save to hdf5, then read off the integration times.
    csvname = 'out/gX_model4.csv'
    wave_extp.writeFnlm_csv(csvname)
    print("Integration times:")
    for lm,t in t0_lm.items():
        print("\t", lm, t)
    print("Total evaluation time:", time.time()-t0)




main(int(sys.argv[1]), int(sys.argv[2]))

#
