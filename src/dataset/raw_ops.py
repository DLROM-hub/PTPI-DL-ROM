################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Implementation of raw loaders <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################

import os 
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import scipy.io as sio

np.random.seed(1)



def loadmat(
        filename : str, 
        id : str
    ):
    """ Loads a matrix for a MAT file.

    Args: 
        filename (str): The full filepath.
        id (str): The matrix identifier.

    Returns:
        The loaded matrix.
    """
    return sio.loadmat(filename)[id].astype('float32').T





def loadfile(
        filename : str, 
        id : str
    ):
    """ Loads a matrix for a MAT file.

    Args: 
        filename (str): The full filepath.
        id (str): The matrix identifier.

    Returns:
        The loaded matrix.

    Raises:
        ValueError: If the file extension is not recognised.
    """
    if filename.endswith('.mat'):
        filecontent = loadmat(filename, id)
    else:
        raise ValueError('Unrecognised file extension') 
    return filecontent





def reshape_channels(
        x : np.array, 
        n_channels : int
    ) -> np.array:
    """ Reshapes the snapshot matrix of shape [N_s, N_h*c] into a tensor of 
        shape [N_s, N_h, c]
    
    Args:
        x (np.array): The snapshot matrix.
        n_channels (int): The number of vector channels. 

    Returns:
         y (np.array): The snapshot tensor.
    """
    
    N_s = x.shape[0]
    N_h = x.shape[1] // n_channels
    y = np.zeros(
        shape = (N_s, N_h, n_channels), dtype = 'float32'
    )
    for c in range(n_channels):
        y[:,:,c] = x[:,c*N_h : (c+1)*N_h]
    return y





def reverse_reshape_channels(
        x : np.array
    ):
    """ Reshapes the snapshot tensor of shape [N_s, N_h, c] into a matrix of 
        shape [N_s, N_h*c]
    
        Args:
            x (np.array): The snapshot tensor.

        Returns:
            y (np.array): The snapshot matrix.
    """

    N_s = x.shape[0]
    N_h = x.shape[1]
    n_channels = x.shape[2]
    y = np.zeros(
        shape = (N_s, N_h * n_channels), dtype = 'float32'
    )
    for c in range(n_channels):
        y[:,c*N_h : (c+1)*N_h] = x[:,:,c] 
    return y