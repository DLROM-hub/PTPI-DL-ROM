################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# loss_models for nonlinear parametrized PDEs in small data regimes
#
# -> Implementation of the loss_model class <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################


import os

os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
import numpy as np
import tqdm
import time

keras.utils.set_random_seed(1)
np.random.seed(1)

from . import utils
from . import dataset





class Trainer:
    """ A class that provides useful functionalities for training
    """

    def __init__(self, loss_model : utils.losses.LossFunction):
        """

        Args:
            loss_model (utils.LossFunction): a model to compute the loss with.
        """
        self.loss_model = loss_model



    def load(self, weights_filepath):
        """ Loads weights and stores them the in architecture that underlies 
            loss_model.
        
        Args:
            weights_filepath (str): the filepath to load from.
        """
        # Check existence of underlying architecture
        self.loss_model.check_architecture()
        
        # Load weights
        self.loss_model.load_weights(weights_filepath, skip_mismatch = True)



    def set_trainability(self, trainability : bool):
        """ Sets trainability of the architecture that underlies loss_model.
        
        Args:
            trainability (bool): trainability value to enforce.
        """
        # Check existence of underlying architecture
        self.loss_model.check_architecture()

        # Sets trainability
        for layer in self.loss_model.layers:
            layer.trainable = trainability
    


    def compile(self, optimizer : keras.optimizers.Optimizer):
        """ Compiles by setting and initializing variables and classes 
            pertaining to optimization.

        Args:
            optimizer (keras.optimizers.Optimizer): the optimizer.
        """

        # Initialize history object
        self.loss_model.history = utils.History()

        # Build optimizer variables
        self.optimizer = optimizer
        self.optimizer.build(self.loss_model.trainable_variables)


    
    def fit(
        self, 
        train_dataset : dataset.Dataset,
        val_dataset : dataset.Dataset,
        test_dataset : dataset.Dataset = None,
        epochs : int = 1, 
        callbacks : list = [],
        display_options : list[str] = ['train_loss', ],
        cache = None
    ) -> utils.History:
        """ Trains the architecture that underlies loss_model.

        Args:
            train_dataset (Dataset): the train dataset that may contain 
                                     both the physics sampler and the 
                                     supervised data.
            val_dataset (Dataset): the validation dataset that may contain 
                                   both the physics sampler and the 
                                   supervised data.
            test_dataset (Dataset): the test dataset that may contain 
                                    both the physics sampler and the 
                                    supervised data (defaults to None)
            epochs (int): the total number of epochs (defaults to 1).
            callbacks (list): callbacks to perform during training (defaults to
                              []).
            display_options (list[str]): what to display as postfix during 
                                         training (defaults to ('train_loss', ))
            cache: cached computations to enable faster training (defaults to 
                   None).

        Returns:
            The training history. 
        """


        @jax.jit
        def compute_loss(
            trainable_variables : list, 
            non_trainable_variables : list, 
            data : dict, 
            cache
        ): 
            """ Computes the loss via stateless call.
        
            Args: 
                trainable_variables (list): list of trainable variables.
                non_trainable_variables (list): list of non-trainable variables.
                data (dict) : collects a single batch of data.
                cache: cached computations to enable faster training.

            Returns:
                loss: the computed loss.
                aux_dict (dict): contains the metrics and the non-trainable
                                 variables .
            """

            (
                output_dict, 
                non_trainable_variables 
            ) = self.loss_model.stateless_call(
                trainable_variables, 
                non_trainable_variables, 
                data, 
                cache
            )
            
            loss = output_dict['loss']
            del output_dict['loss']
            aux_dict = {'non_trainable_variables' : non_trainable_variables, }
            aux_dict = {**output_dict, **aux_dict}

            return loss, aux_dict
        

        @jax.jit
        def train_step(
            state : tuple, 
            data : dict, 
            cache
        ):
            """ Performs a single train step.
        
            Args: 
                state (tuple): Represents the current training state.
                data (dict) : Collects a single batch of data.
                cache: Cached computations to enable faster training.

            Returns:
                output_dict (dict): stores loss and metrics.
                state (list): List of non-trainable variables.
            """

            # Extracts the current state
            (
                trainable_variables, 
                non_trainable_variables, 
                optimizer_variables
            ) = state

            # Computes training loss and its gradients
            grad_fn = jax.value_and_grad(
                compute_loss, has_aux = True
            )
            (loss, aux_dict), grads = grad_fn(
                trainable_variables, non_trainable_variables, data, cache
            )

            # Applies optimization step
            (
                trainable_variables, 
                optimizer_variables 
            ) = self.optimizer.stateless_apply(
                optimizer_variables, grads, trainable_variables
            )

            # Collects auxiliary data
            non_trainable_variables = aux_dict.get('non_trainable_variables')

            # Updates the state
            state = (
                trainable_variables,
                non_trainable_variables,
                optimizer_variables,
            )

            # Constructs output dict
            output_dict = {'loss' : loss,}
            del aux_dict['non_trainable_variables']
            output_dict = {**output_dict, **aux_dict}

            return output_dict, state
        

        @jax.jit
        def val_step(
            state : tuple, 
            data : dict, 
            cache
        ):
            """ Performs a single validation step.
        
            Args: 
                state (tuple): Represents the current training state.
                data (dict) : Collects a single batch of data.
                cache: Cached computations to enable faster training.

            Returns:
                output_dict (dict): stores loss and metrics.
                state (list): List of non-trainable variables.
            """
            
            # Extract the current state
            (
                trainable_variables, 
                non_trainable_variables, 
                optimizer_variables 
            ) = state

            # Computes the validation loss
            loss, aux_dict = compute_loss(
                trainable_variables, 
                non_trainable_variables, 
                data, 
                cache
            )

            # Collects metric value
            output_dict = {'loss' : loss,}
            del aux_dict['non_trainable_variables']
            output_dict = {**output_dict, **aux_dict}
            
            return output_dict, state
        

        # Check existence of underlying architecture
        self.loss_model.check_architecture()

        # Extract state variables
        trainable_variables = self.loss_model.trainable_variables
        non_trainable_variables = self.loss_model.non_trainable_variables
        optimizer_variables = self.optimizer.variables
        state = (
            trainable_variables, 
            non_trainable_variables, 
            optimizer_variables
        )

        # Explicit call to build the loss_model 
        # (necessary to employ stateless_call)
        curr_data_build = train_dataset[0]
        _ = self.loss_model(data = curr_data_build, cache = cache)
        del curr_data_build

        # Initialize callbacks
        callbacks = keras.callbacks.CallbackList(
            callbacks, 
            model = self.loss_model
        )
        callbacks.on_train_begin()
        
        # Initialize tqdm object and elapsed time variable
        epoch_range = tqdm.tqdm(range(epochs))
        t0 = time.time()


        # Training loop
        for epoch in epoch_range:

            # Initialization
            logs = self.loss_model.history.get_last_values()
            callbacks.on_epoch_begin(epoch, logs)

            def epoch_stepper(
                dataset : dataset.Dataset, 
                id : str, 
                step_fn : callable, 
                state,
                cache,
            ):
                """Performs a single step for the specified dataset.

                Args:
                    dataset (dataset.Dataset): the specified dataset.
                    id (str): the dataset identifier
                    step_fn (callable): the step function for the specified 
                                        dataset.
                    cache: the available cached computations.
                
                Returns:
                    The model state.
                """
                if dataset is None:
                    return 
                n_iter = len(dataset)
                for idx in range(n_iter):
                    curr_data = dataset[idx]
                    output_dict, state = step_fn(
                        state = state, 
                        data = curr_data,
                        cache = cache
                    )
                    if idx == 0:
                        self.loss_model.history.init_last(
                            in_dict = output_dict, id = id, coeff = 1 / n_iter
                        )
                    else:
                        self.loss_model.history.update_last(
                            in_dict = output_dict, id = id, coeff = 1 / n_iter
                        )

                dataset.on_epoch_end(epoch)
                return state
            

            # Calling epoch_stepper on each dataset
            state = epoch_stepper(
                train_dataset, 'train', train_step, state, cache
            )
            epoch_stepper(val_dataset, 'val', val_step, state, cache)
            epoch_stepper(test_dataset, 'test', val_step, state, cache)

            # Updating loss_model after epoch
            (
                trainable_variables, 
                non_trainable_variables, 
                optimizer_variables 
            ) = state

            # Saving weights of best loss_model
            for variable, value in zip(
                self.loss_model.trainable_variables, trainable_variables
            ):
                variable.assign(value)
            for variable, value in zip(
                self.loss_model.non_trainable_variables, non_trainable_variables
            ):
                variable.assign(value)

            # Saving elapsed time
            elapsed_time = time.time() - t0
            self.loss_model.history.store('elapsed_time', elapsed_time)

            # Get logs
            logs = self.loss_model.history.get_last_values()
            
            # Callbacks on epoch end
            callbacks.on_epoch_end(epoch, logs)

            # tqdm logging
            epoch_range.set_postfix(
                {key: "%.2e" % logs[key] for key in display_options}
            )

            
        # Callbacks on train end
        callbacks.on_train_end()

        return self.loss_model.history
    

  