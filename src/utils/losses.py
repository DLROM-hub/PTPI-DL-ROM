################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Implementation of keras-enabled losses <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################

import os 
import abc

os.environ["KERAS_BACKEND"] = "jax"

import keras

keras.utils.set_random_seed(1)






class LossFunction(keras.Model):
    """ Abstract class for constructing the loss function on top of a 
        keras.Model
    """

    def __init__(self):
        super().__init__()

    
    def check_architecture(self):
        """ Checks if there exists an underlying architecture

        Raises:
            RuntimeError: if no architecture is present.
        """
        has_weights = len(self.get_weights()) > 0
        if not has_weights:
            raise RuntimeError('This LossFunction instance is not provided with\
                               an underlying architecture')


    @abc.abstractmethod
    def call(self, data, cache, *args):
        """ Abstract call function to force reimplementation.
        """
        return





class L2LossFunction(LossFunction):
    """ Basic implementation of the L2 loss function with one input set and one
        output set.
    """
   
    def __init__(
        self, 
        model : keras.Model, 
        input_id : str,
        output_id : str
    ):
        super().__init__()
        self.model = model
        self.input_id = input_id + '_sup'
        self.output_id = output_id + '_sup'



    def call(self, data, cache = None):
        """

        Args:
            data: the available labeled data.
        
        Returns: 
            a dictionary containing the loss value.
        """
        output = self.model(data[self.input_id])
        l2loss = keras.ops.mean(
            keras.ops.sum((data[self.output_id] - output)**2, axis = 1)
        )
        return {'loss' : l2loss}
    


