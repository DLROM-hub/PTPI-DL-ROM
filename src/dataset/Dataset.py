################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Implementation of the Dataset class <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################


import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
import math
from collections.abc import Iterable

keras.utils.set_random_seed(1)
np.random.seed(1)



class Dataset(keras.utils.PyDataset):
    """ A generic dataset class. It retains the labeled supervised samples and
        the unlabeled physics-based samples.
    """

    def __init__(
        self,
        batched_data_ids : Iterable,
        phys_sampler : callable = None,
        sup_data : dict[np.array] = None,
        batch_size_phys = 1,
        batch_size_sup = 1,
        shuffle : bool = True
    ):
        """ 
        
        Args:
            batched_data_id (Iterable): an iterable of strings which represent
                                        the ids of the collected data.
            phys_sampler (callable): the sampler for unsupervised data.(defaults
                                     to None)
            sup_data (dict[np.array]): the supervised data (defaults to None).
            batch_size_phys: the batch size for the unlabeled dataset (defaults
                             to 1).
            batch_size_sup: the batch size for the labeled dataset (defaults to 
                            1).
            shuffle (bool): if True it shuffles the labeled data (defaults to
                            True).

        """
        self.has_sup_data = sup_data is not None
        self.has_phys_sampler = phys_sampler is not None
        if not self.has_sup_data and not self.has_phys_sampler:
            raise ValueError(
                'Both the supervised data and the physics sampler are not ' + \
                'available : cannot train without data!'
            )
        if self.has_sup_data:
            self.sup_data = sup_data
            self.batch_size_sup = batch_size_sup
        if self.has_phys_sampler:
            self.phys_sampler = phys_sampler
            self.batch_size_phys = batch_size_phys
        self.batched_data_ids = batched_data_ids
        self.set_batching_mechanism(batched_data_ids[0])
        self.shuffle = shuffle



    def set_batching_mechanism(
        self,
        batched_data_id : str
    ):
        """ Computes useful quantities for batching.

        Args:
            batched_data_id (str): id of one of batched data.
        """
        if self.has_sup_data:
            self.N_data_sup = self.sup_data[batched_data_id].shape[0]
            self.n_batches_sup = math.ceil(
                self.N_data_sup / self.batch_size_sup
            )
        if self.has_phys_sampler:
            self.phys_data = self.phys_sampler()
            self.N_data_phys = self.phys_data[batched_data_id].shape[0]
            self.n_batches_phys = math.ceil(
                self.N_data_phys / self.batch_size_phys
            )



    def __len__(
        self
    ) -> int:
        """ 

        Returns:
            The maximum number of batches in the dataset.
        """

        if not self.has_phys_sampler:
            return self.n_batches_sup
        elif not self.has_sup_data:
            return self.n_batches_phys
        else:
            return max(self.n_batches_sup, self.n_batches_phys)



    def __getitem__(
        self,
        idx : int
    ) -> dict:
        """ 
        
        Args:
            idx (int): The index.

        Returns:
            curr_data (dict): The sampled data.
        """

        # Initialize the sample data dictionary
        curr_data = dict()

        # Sample batch from supervised data
        if self.has_sup_data:
            idx_sup = idx % self.n_batches_sup
            low_sup = idx_sup * self.batch_size_sup
            high_sup = min(low_sup + self.batch_size_sup, self.N_data_sup)
            for id in self.sup_data.keys():
                if id in set(self.batched_data_ids):
                    curr_data[id+'_sup'] = self.sup_data[id][low_sup:high_sup]
                else:
                    curr_data[id+'_sup'] = self.sup_data[id]

        # Sample batch for the physics-based loss
        if self.has_phys_sampler:
            idx_phys = idx % self.n_batches_phys
            low_phys = idx_phys * self.batch_size_phys
            high_phys = min(low_phys + self.batch_size_phys, self.N_data_phys)
            for id in self.phys_data.keys():
                if id in set(self.batched_data_ids):
                    curr_data[id+'_phys'] = \
                        self.phys_data[id][low_phys:high_phys]
                else:
                    curr_data[id+'_phys'] = self.phys_data[id]
        
        return curr_data



    def on_epoch_end(
        self, epoch : int
    ):
        """ It is used to shuffle the data at the end of the epoch.

        Args:
            epoch (int): the current epoch.
        """

        if self.has_phys_sampler:
            if epoch % self.phys_sampler.n_resample:
                self.phys_data = self.phys_sampler()
            idxs = np.arange(self.N_data_phys)
            if self.shuffle:
                np.random.shuffle(idxs)
                for id in self.phys_data.keys():
                    if id in set(self.batched_data_ids):
                        self.phys_data[id] = self.phys_data[id][idxs]
        if self.has_sup_data:
            idxs = np.arange(self.N_data_sup)
            if self.shuffle:
                np.random.shuffle(idxs)
                for id in self.sup_data.keys():
                    if id in set(self.batched_data_ids):
                        self.sup_data[id] = self.sup_data[id][idxs]

