################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> nns.__init__.py  <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################


# Define the __all__ variable
__all__ = [
    "DenseNetwork", 
    "dlroms", 
    "FourierFeatureLayer", 
    "ops" , 
    "LowRankDecNetwork"
]

# Import the submodules
from .DenseNetwork import DenseNetwork
from . import dlroms
from .FourierFeatureLayer import FourierFeatureLayer
from . import ops
from .LowRankDecNetwork import LowRankDecNetwork