# __init__.py

from . import units
from . import utilities
from . import analytic
from . import gaussians
from . import portfolio
from . import basis
from . import projection
from . import matrixcalc
from . import wigner
from . import ratecalc

from .units import *
from .utilities import *
from .analytic import *
from .gaussians import *
from .portfolio import *
from .basis import *
from .projection import *
from .matrixcalc import *
# wigner and ratecalc require 'spherical' and 'quaternionic' packages
from .wigner import *
from .ratecalc import *

__all__ = (units.__all__ + utilities.__all__ + basis.__all__ + wigner.__all__
           + projection.__all__ + matrixcalc.__all__ + ratecalc.__all__
           + gaussians.__all__
           + portfolio.__all__ + analytic.__all__)

__version__ = "0.1.0"


"""
Dependencies: the following packages are required:
    math, time, os.path, csv, sys
    numpy
    scipy
    gvar
    vegas
    quaternionic
    spherical
    h5py

Note: spherical and quaternionic packages may require specific
    versions of Python. I have tested with Python 3.7 and 3.9.
Only .wigner and .ratecalc require 'spherical'.
- If a user does not need to perform rotations with the WignerG matrices,
    can load 'vsdm' with the following lines commented out:
# from .wigner import *
# from .ratecalc import *
"""
