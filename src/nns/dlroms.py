################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Function to create dlroms with <-
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





def create_dlrom(
    input_dims : list[int], 
    outputs_dims : list[int], 
    architecture: dict[keras.Model],
    normalizer = None
) -> tuple[keras.Model, keras.Model]:
    """ Builds a DL-ROM model from architectures components.

    Args:
        input_dims (list[int]): The input shape dimensions.
        output_dims (list[int]): The output shape dimensions.
        architecture (dict[keras.Model]): The underlying architecture 
                                          components.
        normalizer: If not None, performs online normalization of input and 
                    outputs (defaults to None).
                       
    Returns:
        model_train (keras.Model): The model employed for training purposes.
        model_test (keras.Model): The model employed in the inference phase (it
                                  does not feature the encoder)
    """

    # Gets the architecture
    reduced_network = architecture.get('reduced_network')
    encoder_network = architecture.get('encoder_network')
    decoder_network = architecture.get('decoder_network')
    
    # Builds the inputs
    input_reduced = keras.Input(input_dims, name = 'input_reduced')
    input_encoder = keras.Input(outputs_dims, name = 'input_encoder')

    # Eventually normalizing the inputs
    if normalizer is not None:
        input_reduced_norm = normalizer.forward_input(input_reduced)
        input_encoder_norm = normalizer.forward_target(input_encoder)
    else:
        input_reduced_norm = input_reduced
        input_encoder_norm = input_encoder
    
    # Builds the neural network
    reduced_network_repr = reduced_network(input_reduced_norm)
    latent_repr = encoder_network(input_encoder_norm)
    output = decoder_network(reduced_network_repr)

    # Eventually un-normalizing the inputs
    if normalizer is not None:
        output = normalizer.backward_output(output)

    # Creates train model
    model_train = keras.models.Model(
        inputs = [input_reduced, input_encoder],
        outputs = [output, reduced_network_repr, latent_repr]
    )
    # Creates the test model
    model_test = keras.models.Model(
        inputs = input_reduced,
        outputs = output
    )

    return model_train, model_test
