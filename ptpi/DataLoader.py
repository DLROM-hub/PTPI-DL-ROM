################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Implementation of the DataLoader class <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################


import os
import sys

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np

keras.utils.set_random_seed(1)
np.random.seed(1)

sys.path.insert(0, '..')

from src.dataset import raw_ops
from src.utils.POD import POD




class DataLoader:
    """ Data loader class: retains train, validation and test set
    """

    def __init__(
            self, 
            alpha_train_branch : float = 0.8
        ):
        """

        Args: 
            alpha_train_branch (float): Training splitting specifics for 
                                        snapshot data (defaults to 0.8)
        """

        self.alpha_train_branch = alpha_train_branch
        self.data_model = dict()
        self.data_trunk = dict()
        self.data_model['train'] = dict()
        self.data_model['val'] = dict()
        self.data_model['test'] = dict()



    def read(
        self, 
        filenames : dict, 
        filenames_test = None,  
        N_t_sup = 1,
        N_t_test = 1,
        n_channels = 1,
        which_instances = None
    ):
        """Reads data from given files

        Args:
            filenames (dict): The filepaths for training and validation data 
            filenames_test: The filepaths for testing data (defaults to None). 
                            If None, test data are not read from a separate 
                            file.
            N_t_sup (int): The number of supervised training timesteps 
                           (defaults to 1).
            N_t_test (int): The number of test timesteps (defaults to 1).
            n_channels (int): The number of channels (defaults to 1).
            which_instances: Employed to slice the training set and collect only
                             the specified instances.
        """

        self.N_t_sup = N_t_sup
        self.N_t_test = N_t_test
        self.n_channels = n_channels
        data = dict()
        data_test = dict()
        data['mu'] = raw_ops.loadfile(filenames['mu'], 'I')
        data['u_true'] = raw_ops.loadfile(filenames['u_true'], 'S')
        data['x'] = raw_ops.loadfile(filenames['x'], 'X')
        data_test['mu'] = raw_ops.loadfile(filenames_test['mu'], 'I')
        data_test['u_true'] = raw_ops.loadfile(filenames_test['u_true'], 'S')
        data_test['x'] = raw_ops.loadfile(filenames_test['x'], 'X')
        if which_instances is not None:
            data['mu'] = data['mu'][which_instances]
            data['u_true'] = data['u_true'][which_instances]
        data['u_true'] = raw_ops.reshape_channels(
            data['u_true'], n_channels = n_channels
        )
        data_test['u_true'] = raw_ops.reshape_channels(
            data_test['u_true'], n_channels = n_channels
        )
        self.process(data, data_test)
        


    def process(
            self, 
            data : dict, 
            data_test = None
    ):
        """ Processes the data (shuffling, splitting, normalization)

        Args:
            data (dict): The read data
            data_test: The read test data (defaults to None). If None, test data
                       are not read from a separate file.
        """

        # Computes total (and training) number fo samples
        n_samples = data['u_true'].shape[0]
        n_train = int(self.alpha_train_branch * (n_samples / self.N_t_sup)) \
            * self.N_t_sup

        # Shuffle and splits the training set / collects data for testing
        idxs = np.arange(n_samples)
        np.random.shuffle(idxs)
        self.data_model['train']['mu'] = data['mu'][idxs][:n_train]
        self.data_model['val']['mu'] = data['mu'][idxs][n_train:]
        self.data_model['test']['mu'] = data_test['mu']
        self.data_model['train']['u_true'] = data['u_true'][idxs][:n_train]
        self.data_model['val']['u_true'] = data['u_true'][idxs][n_train:]
        self.data_model['test']['u_true'] = data_test['u_true']
        for dataset_id in ('train', 'val', 'test'):
            self.data_model[dataset_id]['x'] = data['x']
        del n_train 



    def process_trunk_data(
        self,
        pod : POD,
        alpha_train_trunk : float = 0.9
    ):
        """ Constructs the dataset for the approximation of the POD basis via 
            the trunk net
        
        Args:
            pod (POD): the POD object to retrieve the subspaces with
            alpha_train_trunk (float): the splitting hyperparameter (defaults to
                                       0.9)
        
        """

        self.alpha_train_trunk = alpha_train_trunk
        self.data_trunk['train'] = dict()
        self.data_trunk['val'] = dict()
        self.data_trunk['test'] = dict()

        train_val_u_true = np.vstack(
            (
                self.data_model['train']['u_true'], 
                self.data_model['val']['u_true']
            )
        )
        pod.compute(train_val_u_true)
        
        # Shuffles and splits data for trunk net
        N_h = pod.subspaces.shape[0]
        idxs = np.arange(N_h)
        np.random.shuffle(idxs)
        n_train = int(self.alpha_train_trunk * N_h)
        n_test = int((1.0 - self.alpha_train_trunk) / 2 * N_h)
        loc_data = self.data_model['train']['x'] # train == val == test
        self.data_trunk['train']['x'] = loc_data[idxs][:n_train]
        self.data_trunk['train']['V'] = pod.subspaces[idxs][:n_train]
        self.data_trunk['val']['x'] = loc_data[idxs][n_train:-n_test]  
        self.data_trunk['val']['V'] = pod.subspaces[idxs][n_train:-n_test]  
        self.data_trunk['test']['x'] = loc_data[idxs][-n_test:]
        self.data_trunk['test']['V'] = pod.subspaces[idxs][-n_test:]  



        



    

