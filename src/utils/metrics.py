################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Implementation of keras-enabled metrics <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################

import os 
os.environ["KERAS_BACKEND"] = "jax"

import keras

keras.utils.set_random_seed(1)





def mean_rel_err_metric(u_true, u_pred):
    """ Computes the (mean) relative error metric.

    Args:
        u_true: the ground truth.
        u_pred: the predicted value.

    Returns:
        The (mean) relative error value.
    """
    mean_err = keras.ops.mean(
        keras.ops.mean(
            keras.ops.linalg.norm(u_pred - u_true, axis = 1)**2 / \
            keras.ops.linalg.norm(u_true, axis = 1)**2,
            axis = 0
        )**(1/2),
    )
    return mean_err 

