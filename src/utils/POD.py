################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Implementation of POD utilities <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################

import os 
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import jax.numpy as jnp
from scipy.linalg import svd
from sklearn.utils.extmath import randomized_svd

np.random.seed(1)





class POD:
    """ Implements all the functionalities pertaining to POD computations.
    """

    def __init__(
            self, 
            pod_dim : int, 
            n_channels : int, 
            rpod : bool
        ):
        """ 
        
        Args:
            pod_dim (int): The reduced POD dimension.
            n_channels (int): The number of problem's vector channels.
            rpod (bool): If True, it uses randomized POD.
        """

        self.pod_dim = pod_dim
        self.n_channels = n_channels
        self.rpod = rpod



    def compute_channelwise(
            self, 
            snapshot_matrix : np.array
        ) -> np.array:
        """ Computes the POD subspace matrix channelwise (i.e. for a single 
            channel).

        Args:
            snapshot_matrix (np.array): The snapshot matrix for the selected 
                                        output variable.
        
        Returns:
            subspace: The projection subspace.
        """

        if self.rpod == False:
            subspace, _, _ = svd(
                np.transpose(snapshot_matrix), 
                full_matrices = False
            )
            subspace = subspace[:,:self.pod_dim]
        else:
            subspace, _, _ = randomized_svd(
                np.transpose(snapshot_matrix),
                n_components = self.pod_dim,
                random_state = 0
            )
        return subspace



    def compute(
            self, 
            snapshot_matrix : np.array
        ):
        """ Computes the full POD subspace matrix.

        Args:
            snapshot_matrix (np.array): The full snapshot matrix.
        """
        self.N_h = snapshot_matrix.shape[1] // self.n_channels
        subspaces = list()
        for c in range(self.n_channels):
            curr_target = snapshot_matrix[:,:,c]
            curr_subspace = self.compute_channelwise(curr_target)
            subspaces.append(curr_subspace)
        self.subspaces = np.array(subspaces).transpose((1, 2, 0))
        
    

    def project(
            self, 
            u_hf : jnp.array
        ) -> jnp.array:
        """ Projects the input onto the computed low-rank subspace.

        Args:
            u_hf (jnp.array): The high fidelity batch of vectors.

        Returns:
            coeffs (jnp.array): The reduced coefficients.
        """

        coeffs = jnp.einsum('bic,ijc->bjc', u_hf, self.subspaces)

        return coeffs
    


    def lift(
            self, 
            coeffs : jnp.array
        ) -> jnp.array:
        """ Lifts the input from the low-rank space onto the high-fidelity. 
            space.

        Args:
            coeffs (jnp.array): The reduced coefficients.

        Returns:
            u_hat (jnp.array): The high fidelity batch of vectors.
        """
        u_hat = jnp.einsum('bjc,ijc->bic', coeffs, self.subspaces)
        return u_hat
    


    def compute_projection_error(self, u_true):
        """ Computes the POD projection error according to the 2-norm.

        Args:
            u_true: The target data.

        Returns:
            The projection error.
        """
        u_proj = self.lift(self.project(u_true))
        proj_err = np.mean(
            (np.sum(np.linalg.norm(u_proj - u_true, axis = 1)**2, axis = 0) / \
                np.sum(np.linalg.norm(u_true, axis = 1)**2, axis = 0))**(1/2)
        )
        return proj_err