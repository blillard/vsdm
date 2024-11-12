# __init__.py

from . import units
from . import utilities
from . import analytic
from . import haar
from . import gaussians
from . import portfolio
from . import basis
from . import projection
from . import adaptive
from . import matrixcalc
from . import wigner
from . import ratecalc

from .units import *
from .utilities import *
from .haar import *
from .analytic import *
from .gaussians import *
from .portfolio import *
from .basis import *
from .projection import *
from .adaptive import *
from .matrixcalc import *
# only wigner and ratecalc require 'spherical' and 'quaternionic' packages
from .wigner import *
from .ratecalc import *

__all__ = (units.__all__ + utilities.__all__ + haar.__all__
           + basis.__all__ + wigner.__all__
           + projection.__all__ + matrixcalc.__all__ + ratecalc.__all__
           + gaussians.__all__ + adaptive.__all__
           + portfolio.__all__ + analytic.__all__)

__version__ = "0.3.6"


"""
Dependencies: the following packages are required:
    math, time, csv, sys
    numba
    numpy
    scipy
    gvar
    vegas
    quaternionic
    spherical
    h5py
"""
