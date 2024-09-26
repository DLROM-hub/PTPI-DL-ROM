################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Implementation of DenseNetwork class <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################

import os

os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras

np.random.seed(1)
keras.utils.set_random_seed(1)





class DenseNetwork(keras.Model):
    """ Implements a dense block of constant width and fixed depth.
    """



    def __init__(
        self, 
        width : int, 
        depth : int, 
        output_dim : int,
        activation : callable = keras.layers.LeakyReLU(), 
        kernel_initializer = 'he_uniform'
    ):
        """
        
        Args:
            width (int): The block's width.
            depth (int): The block's depth.
            output_dim (int): The output dimension.
            activation (callable): The activation function (defaults to 
                                   keras.layers.LeakyReLU()).
            kernel_initializer: Layer kernels' initializer (defaults to 
                                'he_uniform')
        """
        super(DenseNetwork, self).__init__()
        self.width = width
        self.depth = depth
        self.output_dim = output_dim
        # Defines the first (depth - 1) layers
        self.dense_layers = [
            keras.layers.Dense(
                self.width,
                activation = activation,
                kernel_initializer = kernel_initializer
            )
            for _ in range(depth-1)
        ]
        # Defines the last layer
        self.dense_layers.append(
            keras.layers.Dense(
                output_dim,
                activation = 'linear',
                kernel_initializer = kernel_initializer
            )
        )
    



    def call(
            self, 
            x, 
            training : bool = False
        ):
        """ Implements the call function for the dense block.
         
        Args:
            x: The block's input.
            training (bool): True if the call is performed during training (
                             defaults to False).
        
        Returns:
            x: The block's output.
        """

        for i in range(self.depth):
            x = self.dense_layers[i](x)
        return x
    