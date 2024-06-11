"""Units for velocity, energy and global constants.

Definitions:
    VUNIT_c: the fundamental velocity unit, divided by the speed of light
        VUNIT_c = 1, for [VUNIT = c] units.
        VUNIT_c = (2.99792e5)**(-1), for [VUNIT = 1 km/s] units
    EUNIT_eV: the fundamental energy unit, divided by 1 eV.
    _Q0_FDM_QBOHR: the reference momentum 'q0' used in the DM-SM scattering
        form factor F_DM(q,n) = (q0/q)**n,
        with q0 in units of inverse Bohr radius qBohr.

*** All energies/momenta units set the Planck constant to hbar=1.

Default choice: VUNIT_c = 1, EUNIT_eV = 1, _Q0_FDM_QBOHR=1.
    All other dimensionful parameters (MeV, km_s, mElec, q0_fdm) are
    functions of these three quantities.

Defining 1 year = 365.0 days = 3.1536e7 s, for 'kg*year' exposure.
"""

__all__ = ["VUNIT_c", "EUNIT_eV", "g_c", "km_s", "qBohr", "q0_fdm",
           "eV", "keV", "MeV", "mElec", "alphaE", "SECONDS_PER_YEAR"]

"""
    Unit choices for velocity, energy, momentum:
    (Can be altered by the user)
"""
VUNIT_c = 1. # [velocity] unit, in units of c.
EUNIT_eV = 1. # [energy] unit, in units of eV.
_Q0_FDM_QBOHR = 1. # sets the ratio q0_fdm/qBohr, for scattering form factor F_DM.
SECONDS_PER_YEAR = 3.1536e7 # for detector exposures in "kg*year"

"""
    Dependent quantities:
    (Not to be altered)
"""
g_c = 1./VUNIT_c # The speed of light in units of [velocity]
km_s = (2.99792e5)**(-1) * g_c # 1 km/s in units of [velocity]
eV = 1./EUNIT_eV # 1 eV in units of [energy]
keV = 1e3*eV
MeV = 1e6*eV
mElec = 0.511*MeV / g_c**2 # electron mass, units of [energy] / [velocity]**2
alphaE = 1/137.036 # fine structure constant
qBohr = mElec*alphaE * g_c # in units of [energy]/[velocity]
q0_fdm = _Q0_FDM_QBOHR*qBohr # the reference momentum for F_DM form factor


#
