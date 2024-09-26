################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Testing utilities for PTPI-DL-ROMs paper <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################


import os
import sys
import time

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np

sys.path.insert(0, '..')

from src import utils
from src import nns
from DataLoader import DataLoader


def test(model : nns.LowRankDecNetwork, data_loader : DataLoader):
    """ Tests full model performance.

    Args:
        model (nns.LowRankDecNetwork): the LowRankDecNetwork model.
        data_loader (DataLoader): The data_loader, which reads and 
                                  stores data.

    Returns:
        test_dict (dict): Contains the ground truth, the predicted solution,
                          and the error computations.
    """
    # Extacting inputs and targets from dataloader
    u_true = data_loader.data_model['test']['u_true']
    mu_test = data_loader.data_model['test']['mu']
    x_test = data_loader.data_model['test']['x']

    # First call to build and compile
    model.predict(mu = np.zeros_like(mu_test), x = x_test).block_until_ready()

    # Compute and register elapsed time
    t0 = time.perf_counter()
    u_pred = model.predict(mu = mu_test, x = x_test).block_until_ready()
    elapsed_time = time.perf_counter() - t0
    elapsed_time_per_instance = elapsed_time / mu_test.shape[0]
    print("Elapsed time per instance = %f s" % elapsed_time_per_instance)

    # Initialize quantities for error computation
    N_t_test = data_loader.N_t_test
    n_instances = data_loader.data_model['test']['mu'].shape[0] // N_t_test
    err_param = np.zeros((n_instances,))
    err_time = 0.0

    # Compute time-wise error
    for i in range(n_instances):
        curr_pred = u_pred[i*N_t_test : (i+1)*N_t_test]
        curr_true = u_true[i*N_t_test : (i+1)*N_t_test]
        num = np.linalg.norm(curr_true - curr_pred, 2, axis = 1)**2
        den = np.linalg.norm(curr_true, 2, axis = 1)**2 
        err_time += num / den
        err_param[i] = np.sqrt(np.sum(num) / np.sum(den))
    err_time = np.sqrt(err_time / n_instances)

    # Compute mean relative error
    mean_err = utils.metrics.mean_rel_err_metric(u_true, u_pred)
    print("Operator approximation: mean relative error = %f" % mean_err)
    
    # Return dictionary with all the error metrics and the predictions
    test_dict = dict()
    test_dict['u_pred'] = u_pred
    test_dict['u_true'] = u_true
    test_dict['mean_err'] = mean_err
    test_dict['err_time'] = err_time
    test_dict['err_param'] = err_param
    test_dict['elapsed_time_per_instance'] = elapsed_time_per_instance

    return test_dict





def test_trunk(trunk : keras.Model, data_loader : DataLoader):
    """ Tests trunk net interpolation performances w.r.t. the 2-norm and the
        Frobenius norm.
    
    Args:
        trunk (keras.Model): The trunk network.
        data_loader (DataLoader): The data_loader, which reads and 
                                  stores data.
    """
    

    # Extract variables and compute predictions
    V_true = data_loader.data_trunk['test']['V']
    V_pred = trunk(data_loader.data_trunk['test']['x'])

    # Compute approximation error
    num = (V_pred - V_true)
    den = V_true
    ords = ['fro', 2]
    norms = ['Frobenius', 'l2']
    err_string = ""
    err_rel = list()       
    for i in range(len(ords)):
        curr_err = np.mean(
            np.linalg.norm(num, ord = ords[i], axis = (0,2)) / \
            np.linalg.norm(den, ord = ords[i], axis = (0,2))
        )
        err_rel.append(curr_err)
        err_string += " [ " + str(norms[i]) + ": " + str(err_rel[i]) + " ]"
    print(
        'Trunk approximation of POD basis: mean relative error =', 
        err_string
    )





class TestCallback(keras.callbacks.Callback):
    """ It is used to log epochs info onto a csv file.
    """

    def __init__(self, data_loader : DataLoader, test_model : keras.Model):
        """ 

        Args:
            data_loader (DataLoader): stores all the data.
            test_model (keras.Model): the underlying model used for inference.
        """
        super().__init__()
        self.data_loader = data_loader
        self.test_model = test_model



    def on_epoch_end(self, epoch : int, logs = None):
        """ Writes info at the end of given epoch.
        
        Args:
            epoch (int): current epoch.
            logs: info at given epoch (defaults to None).
        """
        # Extracting inputs and targets from dataloader
        u_true = self.data_loader.data_model['test']['u_true']
        mu_test = self.data_loader.data_model['test']['mu']
        x_test = self.data_loader.data_model['test']['x']
        # Prediction
        u_pred = self.test_model.predict(mu = mu_test, x = x_test)
        # Mean relative error computation
        mean_err = utils.metrics.mean_rel_err_metric(u_true, u_pred)
        self.model.history.store('test_metric', mean_err)


        