################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> ptpi.__init__.py  <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################


# Define the __all__ variable
__all__ = [ 
    "DataLoader",
    "test_utils", 
]

# Import the submodules
from .DataLoader import DataLoader
from . import test_utils
