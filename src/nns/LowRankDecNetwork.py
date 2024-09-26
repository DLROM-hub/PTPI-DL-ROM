################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Implementation of LowRankDecModel class <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################

import os 
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras
import jax.numpy as jnp

np.random.seed(1)
keras.utils.set_random_seed(1)


from .. import utils


class LowRankDecNetwork(keras.Model):
    """ The LowRankDecNetwork class implements a generic deep learning-based 
        architecture that involves a low-rank decomposition (e.g. DeepONet, 
        POD-DL-ROM).
    """

    def __init__(
        self,
        branch : keras.Model,
        trunk
    ):
        """
        
        Args:
            branch: the branch network
            trunk: the trunk network (either a utils.POD or a keras.Model 
                   object)
        """
        super(LowRankDecNetwork, self).__init__()
        self.branch = branch
        self.trunk = trunk
        assert isinstance(self.trunk, utils.POD) or \
            isinstance(self.trunk, keras.Model)
        self.is_pod = isinstance(self.trunk, utils.POD)
        if self.is_pod:
            self.basis = jnp.array(self.trunk.subspaces)



    def call(self, mu, x):
        """ Implements the call function.
            
        Args:
            mu: the parameters
            x: the locations
        
        Returns:
            The output.
        """  
        b_pred = self.branch(mu)
        t_pred = self.basis if self.is_pod else self.trunk(x)
        u_pred = keras.ops.einsum(
            'bjc,ijc->bic', 
            b_pred, 
            t_pred
        )
        return u_pred


    
    def predict(self, mu, x):
        """ Implements the predict function.
            
        Args:
            mu: the parameters.
            x: the locations.
        
        Returns:
            The output.
        """  
        b_pred = self.branch.predict(mu, verbose = False, batch_size = 1)
        t_pred = self.basis if self.is_pod else self.trunk(x)
        u_pred = keras.ops.einsum(
            'bjc,ijc->bic', 
            b_pred, 
            t_pred
        )
        return u_pred