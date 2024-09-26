################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> src.__init__.py  <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################


# Define the __all__ variable
__all__ = ["dataset", "nns", "utils", "Trainer"]

# Import the submodules
from . import dataset
from . import nns
from . import utils
from .Trainer import Trainer