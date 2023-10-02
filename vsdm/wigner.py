"""Implementation of WignerG rotation matrices for real spherical harmonics.

"""

__all__ = ["WignerG"]

# import math
import numpy as np
# import scipy.special as spf
# import vegas # numeric integration
# import gvar # gaussian variables; for vegas
# import time
import quaternionic # For rotations
import spherical #For Wigner D matrix
# import csv # file IO for projectFnlm
# import os.path
# import h5py # database format for mathcalI arrays

class WignerG():
    """Assembles the real form of the Wigner D matrix.

    Argument:
        ellMax: largest ell for which G(R) is needed

    Returns:
        self.Glist, self.rotations: dictionaries for saving G_ell matrices
            indexed 0,1,... by order in which G_ell(R) is evaluated

    Method:
        G_ell(R, save=True): calculates G for specified rotation
            if save: add rotation to self.rotations, G_ell to self.Glist
            returns dictionary gL, gL[ell] = G(ell), gL['R'] = R
    """
    def __init__(self, ellMax, center_Z2=False):
        self.wigD = spherical.Wigner(ellMax)
        self.ellMax = ellMax
        # evaluate once per rotation: mxD = wigD.D(rotation)
        #     this "matrix" is saved as 1d array of coefficients...
        # get coefficient for index (ell, mprime, m):
        #     D^ell_{mp,m} = mxD[wigD.Dindex(ell, mp, m)]
        self.rotations = {} # save list of rotations
        self.Glist = {} # dict of G_ell matrices for each rotation
        self.rIndex = -1
        self.center_Z2 = center_Z2

    def G_l(self, R, save=False):
        """Calculates G(ell) for all ell=0...ellMax.

        Arguments:
        * R: an SO(3) element, in quaternion representation
        * save: if True, adds R to self.rotations, and G_l(R) to self.Glist

        Output:
        * gL, a dict of G(ell) matrices, gL[ell] = G(ell).
            Includes gL['R'] entry, the value of the quaternion R
        """
        gL = {}
        gL['R'] = R
        # R is a quaternion: doesn't need to be a unit quaternion
        mxD = self.wigD.D(R)
        for ell in range(self.ellMax+1):
            gL[ell] = np.zeros([2*ell+1, 2*ell+1])
            if self.center_Z2 and ell%2!=0: continue
            # mp=m=0:
            ix00 = [ell, ell]
            gL[ell][ix00] = np.real(mxD[self.wigD.Dindex(ell, 0, 0)])
            # mp=0, m>0
            for m in range(1, ell+1):
                ix0p = [ell, ell+m]
                ix0m = [ell, ell-m]
                d_0p = mxD[self.wigD.Dindex(ell, 0, m)]
                d_0m = mxD[self.wigD.Dindex(ell, 0, -m)]
                gL[ell][ix0m] = np.sqrt(2) * np.imag(d_0m)
                gL[ell][ix0p] = np.sqrt(2) * np.real(d_0p)
            # mp>0, m=0:
            for mp in range(1,ell+1):
                ixm0 = [ell-mp, ell]
                ixp0 = [ell+mp, ell]
                d_m0 = mxD[self.wigD.Dindex(ell, -mp, 0)]
                d_p0 = mxD[self.wigD.Dindex(ell, mp, 0)]
                gL[ell][ixm0] = -np.sqrt(2) * np.imag(d_m0)
                gL[ell][ixp0] = np.sqrt(2) * np.real(d_p0)
            # mp>0, m>0:
            for mp in range(1, ell+1):
                for m in range(1, ell+1):
                    ixpp = [ell+mp, ell+m]
                    ixpm = [ell+mp, ell-m]
                    ixmp = [ell-mp, ell+m]
                    ixmm = [ell-mp, ell-m]
                    d_pp = mxD[self.wigD.Dindex(ell, mp, m)]
                    d_mp = mxD[self.wigD.Dindex(ell, -mp, m)]
                    d_pm = mxD[self.wigD.Dindex(ell, mp, -m)]
                    d_mm = mxD[self.wigD.Dindex(ell, -mp, -m)]
                    gL[ell][ixmm] = np.real(d_mm) - (-1)**mp * np.real(d_pm)
                    gL[ell][ixmp] = -np.imag(d_mp) + (-1)**mp * np.imag(d_pp)
                    gL[ell][ixpm] = np.imag(d_pm) + (-1)**mp * np.imag(d_mm)
                    gL[ell][ixpp] = np.real(d_pp) + (-1)**mp * np.real(d_mp)
        if save:
            self.rIndex += 1 # new index for new entry
            self.rotations[self.rIndex] = R
            self.Glist[self.rIndex] = gL
        return gL
