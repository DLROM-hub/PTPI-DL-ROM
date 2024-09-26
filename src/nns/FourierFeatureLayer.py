################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Implementation of FourierFeatureLayer class <-
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




class FourierFeatureLayer(keras.layers.Layer):
    """ Implements the Fourier Feature Layer, which is employed to enhance the 
        input feature space.
    """

    def __init__(
        self, 
        output_dim : int,
        freq_magnitude : float = 1.
    ):
        """
        
        Args:
            output_dim (int): the output dimension of the layer, i.e. the 
                              dimension of the encoding space.
            freq_magnitude (float): the frequence magnitude for the Fourier     
                                    embedding (defaults to 1).
            
        """

        super(FourierFeatureLayer, self).__init__()
        self.output_dim = output_dim
        self.freq_magnitude = freq_magnitude
    
    
    
    def build(
        self, 
        input_shape
    ):
        """ Builds the layer explicitly once called the first time.

        Args:
            input_shape: the dimension of the layer input.
        """
        self.encoding_matrix = self.add_weight(
            shape = (input_shape[-1], self.output_dim),
            initializer = "random_normal",
            trainable = False
        )
    
    

    def call(
        self, 
        x
    ):
        """ Implements the call function for the Fourier Feature layer.
            
        Args:
            x: The layer input.
        
        Returns:
            y: The layer output.
        """  
        encoding = 2 * np.pi * self.freq_magnitude * keras.ops.einsum(
            'bi,ij->bj', x, self.encoding_matrix
        )
        y = keras.ops.concatenate(
            (keras.ops.cos(encoding), keras.ops.sin(encoding)),
            axis = -1
        )
        return y
