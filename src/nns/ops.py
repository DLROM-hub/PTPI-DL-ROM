################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Implementation of operations pertaining to neural networks <-
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
    




def get_dlrom_dummy(
    branch : keras.Model,
    mu_phys 
):
    """ Creates a dummy variable for the encoder input of DL-ROM. It is useful
        for the physics-based loss computation.

    Args:
        branch (keras.Model): The DL-ROM-based branch network.
        mu_phys: The input time-parameter vector.
    
    Returns:
        dummy: The dummy variable.
    """

    dummy = jnp.zeros(
        shape = (
            mu_phys.shape[0], 
            branch.input[-1].shape[1],
            branch.input[-1].shape[2]
        )
    )

    return dummy





def get_time_derivative(
    branch : keras.Model,
    mu_phys,
    tau : float
):
    """ Computes the first time derivative using the centered second order FD 
        scheme.

    Args:
        branch (keras.Model): The branch network.
        mu_phys: The input time-parameter vector.
        tau (float): The time grid size.

    Returns:
        b_prime: The computed derivative.

    Raises:
        ValueError: If the branch is neither a FFNN or a DL-ROM.
    """

    # Extract lengths
    n_inputs_branch = len(branch.input)
    p = mu_phys.shape[1]

    # Construct FD tensor
    fd_tensor = jnp.array([0.,] * (p - 1) + [tau,])

    # Compute output+ and output-
    if n_inputs_branch == 1:
        b_phys_plus = branch(
            mu_phys + fd_tensor
        )
        b_phys_minus = branch(
            mu_phys - fd_tensor
        )
    elif n_inputs_branch == 2:
        dummy = get_dlrom_dummy(branch = branch, mu_phys = mu_phys)
        b_phys_plus, _, _ = branch(
            (mu_phys + fd_tensor, dummy)
        )
        b_phys_minus, _, _ = branch(
            (mu_phys - fd_tensor, dummy)
        )
    else:
        raise ValueError("The branch network is neither a FFNN or a DL-ROM")
    
    # Compute derivative
    b_prime = (b_phys_plus - b_phys_minus) / (2 * tau)

    return b_prime





def hstack_nns(
    input_dims : list[int], 
    blocks : list[keras.Model]
) -> keras.Model:
    """ Stacks horizontally (in series) given neural network blocks.

    Args:
        input_dims (list[int]): The input shape dimensions.
        blocks (list[keras.Model]): The neural network blocks.
    
    Returns:
        model (keras.Model): The full model composed of h-stacked blocks.
    """
    
    # Builds the network
    input = keras.Input(input_dims, name = 'input_reduced')
    output = input
    for i in range(len(blocks)):
        output = blocks[i](output)
    
    # Creates the model
    model = keras.models.Model(
        inputs = input,
        outputs = output
    )

    return model




def vstack_nns(
    input_dims : list[int],
    blocks : list[keras.Model]
) -> keras.Model:
    """ Stacks vertically (in parallel) given neural network blocks (their input
        is the same). 

    Args:
        input_dims (list[int]): The input shape dimensions.
        blocks (list[keras.Model]): The neural network blocks.

    Returns:
        model (keras.Model): The full model composed of v-stacked blocks.
    """

    # Builds the network
    input = keras.Input(input_dims, name = 'input_reduced')
    output = keras.ops.concatenate(
        [blocks[i](input)[:,:,None] for i in range(len(blocks))],
        axis = -1
    )
    
    # Creates the model
    model = keras.models.Model(
        inputs = input,
        outputs = output
    )

    return model
