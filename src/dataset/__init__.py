################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> dataset.__init__.py  <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################


# Define the __all__ variable
__all__ = [ 
    "Dataset",
    "raw_ops", 
]

# Import the submodules
from .Dataset import Dataset
from . import raw_ops

