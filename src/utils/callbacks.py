################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Implementation of callbacks <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################

import os 
os.environ["KERAS_BACKEND"] = "jax"

import csv

import numpy as np
import keras

np.random.seed(1)
keras.utils.set_random_seed(1)





class LogCallback(keras.callbacks.Callback):
    """ It is used to log epochs info onto a csv file.
    """

    def __init__(self, filepath):
        """ 

        Args:
            filepath: the logfile's path.
        """
        super().__init__()
        self.filepath = filepath



    def on_epoch_end(self, epoch : int, logs = None):
        """ Writes info at the end of given epoch.
        
        Args:
            epoch (int): current epoch.
            logs: logging info (defaults to None).
        """
        if epoch == 0:
            with open(self.filepath, 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(logs.keys())
        with open(self.filepath, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(logs.values())





class SaveCallback(keras.callbacks.Callback):
    """ It is used to save the model during the training.
    """
            
    def __init__(self, checkpoint_filepath):
        """
        
        Args:
            checkpoint_filepath: the filepath for saving the model checkpoint.
        """
        super().__init__()
        self.saved_loss = np.inf
        self.checkpoint_filepath = checkpoint_filepath

    
    def on_epoch_end(self, epoch : int, logs = None):
        """ Saves the model checkpoint according the validation performance.

        Args: 
            epoch (int): the current epoch.
            logs: logging info (defaults to None).
        """
        last_val_loss = self.model.history.get_last_values()['val_loss']
        if  last_val_loss < self.saved_loss:
            self.saved_loss = np.copy(last_val_loss)
            self.model.save_weights(
                self.checkpoint_filepath, overwrite = True
            )





class LoadCallback(keras.callbacks.Callback):
    """ It is used to load the weights at the end of the training. If used in 
        combination with SaveCallback, it restores the best model weights.
    """
            
    def __init__(self, checkpoint_filepath):
        """
        
        Args:
            checkpoint_filepath: the filepath to load the model checkpoint from.
        """
        super().__init__()
        self.checkpoint_filepath = checkpoint_filepath

    
    def on_train_end(self, logs = None):
        """ Loads the model checkpoint.

        Args:
            logs: logging info (defaults to None).
        """
        self.model.load_weights(self.checkpoint_filepath, skip_mismatch = True)

        
