################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Implementation of the Normalizer class <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################

import os 
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np

np.random.seed(1)





class Normalizer:
    """ Implements the normalizer to be used inside a neural network.
    """

    def __init__(
            self, 
            input_train,
            target_train
        ):
        """ Computes min-max constants from training data.

        Args:
            input_train: the (batched) training input.
            output_train: the batched training target.
        """
        self.input_min = np.min(input_train, axis = 0)
        self.input_max = np.max(input_train, axis = 0)
        self.target_min = np.min(target_train, axis = (0,1))
        self.target_max = np.max(target_train, axis = (0,1))



    def forward_input(self, op_input):
        """ Forward normalization pass for the input data.

        Args:
            op_input: the (batched) input.

        Returns:
            The normalized input.
        """
        normalized_input = (op_input - self.input_min) \
            / (self.input_max - self.input_min)
        return normalized_input
    


    def forward_target(self, op_target):
        """ Forward normalization pass for the target data.

        Args:
            op_target: the (batched) target.

        Returns:
            The normalized target.
        """
        normalized_target = (op_target - self.target_min) / \
            (self.target_max - self.target_min)
        return normalized_target
    


    def backward_output(self, op_output):
        """ Backward normalization pass for the neural network output.

        Args:
            op_output: the (batched) output.

        Returns:
            The un-normalized output.
        """
        unnormalized_output = op_output * (self.target_max - self.target_min) +\
            self.target_min
        return unnormalized_output
