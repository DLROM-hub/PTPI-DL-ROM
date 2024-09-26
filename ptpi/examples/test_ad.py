################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Testing automatic differentiation performance <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################

import sys
import os
import time

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_FLAGS"] = \
    '--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'

import jax
import keras
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.style.use('dark_background')

sys.path.insert(0, '..')
sys.path.insert(0, os.path.join('..', '..'))

import src.nns as nns

keras.backend.clear_session()
keras.utils.set_random_seed(1)
np.random.seed(1)



################################################################################
# CONFIGURATION: Experiment setting
################################################################################

# Setting the results directory
results_dir = os.path.join('..', '..', 'results', 'test_ad')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
results_data_file = os.path.join(results_dir, "test_ad.npy")


# Setting hyperparameters
input_size = 3
output_size = 10
batch_size = 100



################################################################################
# CONFIGURATION: Auxiliary functions
################################################################################

# Defining function to compute the gradient with
def get_grad(nn, x, order : int):
    assert order == 1 or order == 2 or order == 3
    n_spatial_dims = x.shape[1]
    grads = list()
    grads2 = list()
    for ix in range(n_spatial_dims):
        get_scalar_output = lambda x, y, z: nn(
            jnp.concatenate((x[:,None],y[:,None],z[:,None]), axis = 1)
        )[0]
        if order == 1:
            grad_op = jax.vmap(
                jax.jacobian(get_scalar_output, argnums = ix)
            )
            grads +=  (
                grad_op(x[:,0][:,None], x[:,1][:,None], x[:,2][:,None])[:,:,0], 
            )
        if order == 2:
            curr_grads2 = list()
            for ix2 in range(n_spatial_dims):
                grad2_op = jax.vmap(
                    jax.jacobian(
                        jax.jacobian(get_scalar_output, argnums = ix), 
                        argnums = ix2
                    )
                )
                curr_grads2 += (
                    grad2_op(
                        x[:,0][:,None], x[:,1][:,None], x[:,2][:,None]
                    )[:,:,0,0],
                )
            grads2 += (curr_grads2, )
    if order == 1:
        return grads
    else:
        return grads, grads2
    


# Defining the function to create the neural network with
def create_dense_nn(width, depth):
    # Define trunk components
    dense_nn = nns.DenseNetwork(
        width = width, 
        depth = depth, 
        output_dim = output_size,
        activation = 'elu'
    )
    reshape_nn = keras.layers.Reshape(target_shape = (output_size, 1))

    # Build trunk network
    nn_arch = [dense_nn, reshape_nn]
    dense_nn = nns.ops.hstack_nns(
        input_dims = (input_size, ),
        blocks = nn_arch
    )

    return dense_nn



# Definition of the function which runs AD for the first and second derivative
def run_ad(dense_nn, x):
    t1 = time.perf_counter()
    out1 = get_grad(dense_nn, x, order = 1)
    time_d1 = time.perf_counter() - t1
    t2 = time.perf_counter() 
    out2 = get_grad(dense_nn, x, order = 2)
    time_d2 = time.perf_counter() - t2
    return time_d1, time_d2



################################################################################
# CONFIGURATION: Functions for the numerical experiments.
################################################################################

def test_depth(
    runs : int = 10, 
    width : int = 10, 
    depths = range(3,11)
):
    """ Function to test the AD computation time w.r.t. the nn depth.

    Args:
        runs (int): the number of runs to repeat the experiment (defaults to 
                    10).
        width (int): fixed value of the neural network width.
        depths : range of depths to analyze.

    Returns:
        depths: the range of depths.
        elapsed_time_d1: the elapsed time for the computation of the 1st der. 
        elapsed_time_d2: the elapsed time for the computation of the 2nd der. 
    """
    elapsed_time_d1 = np.zeros((runs, len(list(depths))))
    elapsed_time_d2 = np.zeros((runs, len(list(depths))))

    for d, depth in enumerate(depths):
        
        x = np.random.rand(batch_size,input_size)
        
        dense_nn = create_dense_nn(width, depth)
    
        out1 = get_grad(dense_nn, x, order = 1)
        out2 = get_grad(dense_nn, x, order = 2)

        for run in range(runs):
            time_d1, time_d2 = run_ad(dense_nn, x)
            elapsed_time_d1[run,d] = time_d1
            elapsed_time_d2[run,d] = time_d2
        
    return depths, elapsed_time_d1, elapsed_time_d2



def test_width(
    runs : int = 10, 
    depth : int = 10, 
    widths = range(5,30,3)
):
    """ Function to test the AD computation time w.r.t. the nn width.

    Args:
        runs (int): the number of runs to repeat the experiment (defaults to 
                    10).
        depth (int): fixed value of the neural network depth.
        widths : range of widths to analyze.

    Returns:
        widths: the range of widths.
        elapsed_time_d1: the elapsed time for the computation of the 1st der. 
        elapsed_time_d2: the elapsed time for the computation of the 2nd der. 
    """
    elapsed_time_w1 = np.zeros((runs, len(list(widths))))
    elapsed_time_w2 = np.zeros((runs, len(list(widths))))

    for w, width in enumerate(widths):
        
        x = np.random.rand(batch_size,input_size)
        
        dense_nn = create_dense_nn(width, depth)
    
        out1 = get_grad(dense_nn, x, order = 1)
        out2 = get_grad(dense_nn, x, order = 2)

        for run in range(runs):
            time_w1, time_w2 = run_ad(dense_nn, x)
            elapsed_time_w1[run,w] = time_w1
            elapsed_time_w2[run,w] = time_w2
        
    return widths, elapsed_time_w1, elapsed_time_w2
        


################################################################################
# NUMERICAL EXPERIMENTS 
################################################################################

# Experiments 
depths, elapsed_time_d1, elapsed_time_d2 = test_depth()
widths, elapsed_time_w1, elapsed_time_w2 = test_width()

# Saving and loading
try:
    results_data = dict()
    results_data['depths'] = depths
    results_data['elapsed_time_d1'] = elapsed_time_d1
    results_data['elapsed_time_d2'] = elapsed_time_d2
    results_data['widths'] = widths
    results_data['elapsed_time_w1'] = elapsed_time_w1
    results_data['elapsed_time_w2'] = elapsed_time_w2
    np.save(results_data_file, results_data)
except:
    results_data = np.load(
        results_data_file, 
        allow_pickle = True
    )[()]
    depths = results_data['depths']
    elapsed_time_d1 =results_data['elapsed_time_d1']
    elapsed_time_d2 = results_data['elapsed_time_d2']
    widths = results_data['widths']
    elapsed_time_w1 = results_data['elapsed_time_w1']
    elapsed_time_w2 = results_data['elapsed_time_w2'] 


plot_info = zip(
    [depths, widths], 
    [elapsed_time_d1,elapsed_time_w1], 
    [elapsed_time_d2,elapsed_time_w2],
    [('g', 'c'), ('r', 'orange')],
    ['depth', 'width']
)



################################################################################
# POSTPROCESSING 
################################################################################

def postprocess(filepath, plot_info):
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (10,4))
    axs[0].set_ylabel('Elapsed time [s]', fontsize = 14.5)

    for idx_plot, (x, y1, y2, styles, xlabel_txt) in enumerate(plot_info):
        axs[idx_plot].plot(
            x, 
            np.mean(y1, axis = 0), 
            styles[0], linewidth = 3, 
            label = '$1^{st}$ der.'
        )
        axs[idx_plot].fill_between(
            x, 
            np.min(y1, axis = 0),
            np.max(y1, axis = 0),
            color = styles[0], 
            alpha = 0.3,
            edgecolor = None
        )
        axs[idx_plot].set_xlabel(xlabel_txt, fontsize = 14.5)
        axs[idx_plot].plot(
            x, 
            np.mean(y2, axis = 0), 
            styles[1], 
            linewidth = 3, 
            label = '$2^{nd}$ der.'
        )
        axs[idx_plot].fill_between(
            x, 
            np.min(y2, axis = 0),
            np.max(y2, axis = 0),
            color = styles[1], 
            alpha = 0.3,
            edgecolor = None
        )
        axs[idx_plot].legend(
            bbox_to_anchor = (0, 1.02, 1, 0.2), 
            loc = "lower left",
            mode = "expand", 
            borderaxespad = 0, 
            ncol = 3,
            fontsize = 14
        )
        xticks_curr = [np.min(x), int(np.floor(np.mean(x))), np.max(x)]
        yticks_curr = [
            np.min(y1), 
            (np.min(y1)+np.max(y2))/2, 
            np.max(y2)
        ]
        yticks_labels = ['{:.2e}'.format(y) for y in yticks_curr]
        axs[idx_plot].set_xticks(xticks_curr)
        axs[idx_plot].set_xticklabels(labels = xticks_curr, fontsize = 13)
        axs[idx_plot].xaxis.set_minor_locator(
            ticker.LinearLocator(int((np.max(x) - np.min(x))) + 1)
        )
        axs[idx_plot].set_yticks(yticks_curr)
        axs[idx_plot].set_yticklabels(yticks_labels, fontsize = 13)
        axs[idx_plot].yaxis.set_minor_locator(
            ticker.LinearLocator(
                int((np.max(y2) - np.min(y1)) / (np.min(y1)) * 2) + 1
            )
        )
        axs[idx_plot].set_xlim([np.min(x), np.max(x)])

    plt.tight_layout()
    plt.savefig(filepath)


# Call plot functions for paper
plt.style.use('default')
plot_info_paper = zip(
    [depths, widths], 
    [elapsed_time_d1,elapsed_time_w1], 
    [elapsed_time_d2,elapsed_time_w2],
    [('b', 'c'), ('r', 'orange')],
    ['depth', 'width']
)
postprocess(
    filepath = os.path.join(results_dir, 'test_ad.jpg'),
    plot_info = plot_info_paper
)

# Call plot functions for slides
plt.style.use('dark_background')
plot_info_slides = zip(
    [depths, widths], 
    [elapsed_time_d1,elapsed_time_w1], 
    [elapsed_time_d2,elapsed_time_w2],
    [('#32CD32', 'c'), ('r', 'orange')],
    ['depth', 'width']
)
postprocess(
    filepath = os.path.join(results_dir, 'test_ad_slides.jpg'),
    plot_info = plot_info_slides
)