################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Implementation of the Checkpoint class <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################

import os 
os.environ["KERAS_BACKEND"] = "jax"





class Checkpoint:
    """ Checkpoints class to save the model with.
    """

    def __init__(
            self, 
            save_folder : str
        ):
        """ 

        Args:
            save_folder (str): the saving filepath.
        """
        self.checkpoint_folder = os.path.join(
            save_folder, "checkpoints"
        )
        if not os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)



    def get_weights_filepath(
            self, 
            name : str
        ) -> str:
        """ Gets the checkpoint filepath.

        Args: 
            name (str): The architecture name.

        Returns:
            checkpoint_filepath (str): The checkpoint filepath for the 
                                       architecture identified by "name".
        """
        checkpoint_filepath = os.path.join(
            self.checkpoint_folder,
            name + ".weights.h5"
        )
        return checkpoint_filepath
    


    def get_log_filepath(
            self, 
            name : str
        ) -> str:
        """ Gets the checkpoint filepath.

        Args: 
            name (str): The architecture name.

        Returns:
            checkpoint_filepath (str): The checkpoint filepath for the 
                                       architecture identified by "name".
        """
        checkpoint_filepath = os.path.join(
            self.checkpoint_folder,
            name + ".csv"
        )
        return checkpoint_filepath