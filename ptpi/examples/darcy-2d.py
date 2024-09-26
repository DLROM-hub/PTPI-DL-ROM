################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Implementation of the Darcy 2D test case <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################

import sys
import os
import csv

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_FLAGS"] = \
    '--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.10'

import jax
import keras
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.patches import Rectangle

sys.path.insert(0, '..')
sys.path.insert(0, os.path.join('..', '..'))

from src.dataset import *
import src.nns as nns
import src.utils as utils
import src.Trainer as Trainer
from ptpi import *


keras.backend.clear_session()
keras.utils.set_random_seed(1)
np.random.seed(1)


################################################################################
# CONFIGURATION: Definition of hyperaparameters and directories
################################################################################

# Define problem hyperparameters
pod_dim = 30
latent_dim = 7
c = 3
d = 2
p = 2
N_t_sup = 1
N_t_test = 1

# Set up directories
parent_dir =  os.path.join('..', '..', 'data_pi', 'darcy-2d')
results_dir = os.path.join('..', '..', 'results', 'darcy-2d')
slides_dir = os.path.join('..', '..', 'results', 'darcy-2d', 'slides')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(slides_dir):
    os.makedirs(slides_dir)
checkpoint = utils.Checkpoint(save_folder = results_dir)

# Collect data filenames
filenames = dict()
filenames['mu'] = os.path.join(parent_dir, 'params_train_darcy.mat')
filenames['u_true'] = os.path.join(parent_dir, 'S_train_darcy.mat')
filenames['x'] = os.path.join(parent_dir, 'loc_points_train_darcy.mat')
filenames_test = dict()
filenames_test['mu'] = os.path.join(parent_dir, 'params_test_darcy.mat')
filenames_test['u_true'] = os.path.join(parent_dir, 'S_test_darcy.mat')
filenames_test['x'] = os.path.join(parent_dir, 'loc_points_test_darcy.mat')

# Train or test
train_or_test = sys.argv[1]
assert (train_or_test == 'train' or train_or_test == 'test')


################################################################################
# CONFIGURATION: Function to construct the neural network architectures
################################################################################

def get_architectures():
    # Define trunk components
    fourier_encoding_dim = 100
    fourier_feature_layer = nns.FourierFeatureLayer(
        output_dim = fourier_encoding_dim,
        freq_magnitude = 50
    )
    dense_trunk = nns.DenseNetwork(
        width = 100, 
        depth = 7, 
        output_dim = pod_dim * c,
        activation = 'silu',
        kernel_initializer = 'he_uniform'
    )
    reshape_trunk = keras.layers.Reshape(target_shape = (pod_dim, c))

    # Build trunk network
    trunk_arch = [fourier_feature_layer, dense_trunk, reshape_trunk]
    trunk = nns.ops.hstack_nns(
        input_dims = (d, ),
        blocks = trunk_arch
    )

    # Define branch components
    reduced_network = nns.DenseNetwork(
        width = 50, 
        depth = 7, 
        output_dim = latent_dim,
        activation = 'elu',
        kernel_initializer = 'he_uniform'
    )
    reshape_encoder = keras.layers.Reshape(target_shape = (pod_dim * c, ))
    encoder_network = nns.DenseNetwork(
        width = 50, 
        depth = 5, 
        output_dim = latent_dim,
        activation = 'elu',
        kernel_initializer = 'he_uniform'
    )
    encoder_block = [reshape_encoder, encoder_network]
    encoder_network = nns.ops.hstack_nns(
        input_dims = (pod_dim, c),
        blocks = encoder_block
    )
    decoder_network = nns.DenseNetwork(
        width = 70, 
        depth = 7, 
        output_dim = pod_dim * c,
        activation = 'elu',
        kernel_initializer = 'he_uniform'
    )
    reshape_decoder = keras.layers.Reshape(target_shape = (pod_dim, c))
    decoder_block = [decoder_network, reshape_decoder]
    decoder_network = nns.ops.hstack_nns(
        input_dims = (latent_dim, ),
        blocks = decoder_block
    )

    # Build the DL-ROM enhanced branch network
    branch_arch = dict()
    branch_arch['reduced_network'] = reduced_network
    branch_arch['encoder_network'] = encoder_network
    branch_arch['decoder_network'] = decoder_network
    branch, branch_test = nns.dlroms.create_dlrom(
        input_dims = (p, ),
        outputs_dims = (pod_dim, c),
        architecture = branch_arch,
    )

    return trunk, branch, branch_test


################################################################################
# CONFIGURATION: Definition of the function to compute gradients with
################################################################################

def get_grad(trunk, x):
    n_spatial_dims = x.shape[1]
    grads = list()
    for ix in range(n_spatial_dims):
        get_scalar_output = lambda x, y: trunk(
            jnp.concatenate((x[:,None],y[:,None]), axis = 1)
        )[0]
        grad_op = jax.vmap(jax.jacrev(get_scalar_output, argnums = ix))
        grads +=  (grad_op(x[:,0][:,None], x[:,1][:,None])[:,:,:,0], )
    return grads


################################################################################
# CONFIGURATION: Definition of the physical model
################################################################################

class Darcy(utils.losses.LossFunction):

    def __init__(self, branch, trunk, pod, loss_weights):
        super(Darcy, self).__init__()
        self.trunk = trunk
        self.branch = branch
        self.N = self.trunk.output.shape[-1]
        self.kappa_minus_one_fn = lambda mu, x: (
            (jnp.exp(jnp.sin(mu[0]*jnp.pi*x[:,0]) + jnp.sin(mu[1]*jnp.pi*x[:,1])))
        ).astype('float32')
        self.kappa_minus_one_fn = jax.vmap(self.kappa_minus_one_fn, in_axes = (0,None))
        self.pod = pod
        self.loss_weights = loss_weights

        
    def call(self, data, cache):

        # Extract data
        mu_phys = data.get('mu_phys')
        x_phys = data.get('x_phys')
        x_sup = data.get('x_sup')
        mu_sup = data.get('mu_sup')
        u_sup_true = data.get('u_true_sup')

        # Generate auxiliary data
        if cache == None:
            if self.loss_weights['omega_pde'] + self.loss_weights['omega_bc'] > 0:
                x_pde = x_phys['x_pde']
                x_bc_p = x_phys['x_bc_p']
                x_bc_left = x_phys['x_bc_left']
                x_bc_right = x_phys['x_bc_right']
                t_sup = self.trunk(x_sup)
                t_bc_left = self.trunk(x_bc_left)
                t_bc_right = self.trunk(x_bc_right)
                t_bc_p = self.trunk(x_bc_p)
                t_pde = self.trunk(x_pde)
                grads = get_grad(trunk = self.trunk, x = x_pde)
                dt_dx, dt_dy = grads
            else:
                t_sup = self.trunk(x_sup)
        else:
            t_sup = cache.get('basis')
            t_bc_p = cache.get('basis_bc_p')
            t_bc_left = cache.get('basis_bc_left')
            t_bc_right = cache.get('basis_bc_right')
            x_pde = cache.get('x_pde')
            grads = cache.get('grads')
            t_pde = cache.get('basis_pde')
            if grads is not None:
                dt_dx, dt_dy = grads

        # Supervised loss
        u_sup_true_proj = self.pod.project(u_sup_true)
        b_sup, reduced_network_repr, latent_repr = \
            self.branch((mu_sup, u_sup_true_proj))
        l_latent = keras.ops.mean((reduced_network_repr - latent_repr)**2)
        u_sup = keras.ops.einsum('bjc,ijc->bic', b_sup, t_sup)
        l_sup = keras.ops.mean(keras.ops.sum((u_sup - u_sup_true)**2, axis = 1))
        loss =  self.loss_weights['omega_sup'] * l_sup 

        # Eventually compute physics-based loss
        if self.loss_weights['omega_pde'] + self.loss_weights['omega_bc'] > 0:
            kappa_minus_one = self.kappa_minus_one_fn(mu_phys, x_pde)
            dummy = jnp.zeros(shape = (mu_phys.shape[0], u_sup_true_proj.shape[1], u_sup_true_proj.shape[2]))
            b_phys, _, _ = self.branch((mu_phys, dummy))
            U = keras.ops.einsum('bjc,ijc->bic', b_phys, t_pde)
            sigmax = U[:,:,0]
            sigmay = U[:,:,1]
            p = U[:,:,2]
            dU_dx = keras.ops.einsum('bjc,ijc->bic', b_phys, dt_dx)
            dsigmax_dx = dU_dx[:,:,0]
            dsigmay_dx = dU_dx[:,:,1]
            dp_dx = dU_dx[:,:,2]
            dU_dy = keras.ops.einsum('bjc,ijc->bic', b_phys, dt_dy)
            dsigmax_dy = dU_dy[:,:,0]
            dsigmay_dy = dU_dy[:,:,1]
            dp_dy = dU_dy[:,:,2]
            res_pde_1 = (kappa_minus_one * sigmax - dp_dx)**2
            res_pde_2 = (kappa_minus_one * sigmay - dp_dy)**2
            res_pde_3 = (dsigmax_dx + dsigmay_dy + 1.0)**2
            res_pde = res_pde_1 + res_pde_2 + res_pde_3
            l_pde = keras.ops.mean(res_pde)
            p_bc = keras.ops.einsum('bjc,ijc->bic', b_phys, t_bc_p)[:,:,2]
            sigmax_left_bc = keras.ops.einsum('bjc,ijc->bic', b_phys, t_bc_left)[:,:,0]
            sigmax_right_bc = keras.ops.einsum('bjc,ijc->bic', b_phys, t_bc_right)[:,:,0]
            res_p = keras.ops.mean(p_bc**2)
            res_left = keras.ops.mean((sigmax_left_bc)**2)
            res_right = keras.ops.mean((sigmax_right_bc)**2)
            l_bc = res_p + res_left + res_right
            loss += self.loss_weights['omega_pde'] * l_pde + \
                self.loss_weights['omega_bc'] * l_bc
        
        # Eventually adding latent loss
        if self.loss_weights['omega_latent'] > 0:
            loss += self.loss_weights['omega_latent'] * l_latent 
        
        metric = utils.metrics.mean_rel_err_metric(u_sup_true, u_sup)

        return {'loss' : loss, 'metric' : metric}


################################################################################
# CONFIGURATION: Dataset creation
################################################################################
    
# Train and test data
data_loader = DataLoader(alpha_train_branch = 0.8)
data_loader.read(
    filenames, 
    filenames_test, 
    N_t_sup = N_t_sup, 
    N_t_test = N_t_test,
    n_channels = c,
)

# Get POD-based data
pod = utils.POD(
    pod_dim = pod_dim,
    n_channels = c,
    rpod = True
)
data_loader.process_trunk_data(pod)

# Define callable sampler for the physics-based loss
class PhysicsSampler:

    def __init__(self, n_resample):
        self.N_s = 100
        self.mu_1_min = 0.9
        self.mu_1_max = 2.1
        self.mu_2_min = 0.9
        self.mu_2_max = 3.1
        self.n_resample = n_resample

    def __call__(self):
        data = dict()
        data['x'] = dict()
        mu_1 = self.mu_1_min + (self.mu_1_max - self.mu_1_min) * \
            np.random.rand(self.N_s,1)
        mu_2 = self.mu_2_min + (self.mu_2_max - self.mu_2_min) * \
            np.random.rand(self.N_s,1)
        mu = np.concatenate((mu_1, mu_2), axis = 1)
        data['mu'] = mu
        data['x']['x_pde'] = np.random.rand(300,2) * np.array([3.0,1.0])
        x_sup = data_loader.data_model['train']['x']
        left = (x_sup[:,0] == 0).astype('bool')
        right = (x_sup[:,0] == 3).astype('bool')
        up_down = ((x_sup[:,1] == 0) + (x_sup[:,1] == 1)).astype('bool')
        data['x']['x_bc_p'] = x_sup[up_down]
        data['x']['x_bc_left'] = x_sup[left]
        data['x']['x_bc_right'] = x_sup[right]
        return data

phys_sampler = PhysicsSampler(n_resample = 5)

# Define batch dimension
batch_size_sup = 10
batch_size_phys = 10

# Define entire datasets (supervised + physics)
dataset_train_whole = Dataset(
    batched_data_ids = ['mu', 'u_true'],
    phys_sampler = phys_sampler,
    sup_data = data_loader.data_model['train'],
    batch_size_sup = batch_size_sup,
    batch_size_phys = batch_size_phys
)
dataset_val_whole = Dataset(
    batched_data_ids = ['mu', 'u_true'],
    phys_sampler = phys_sampler,
    sup_data = data_loader.data_model['val'],
    batch_size_sup = batch_size_sup,
    batch_size_phys = batch_size_phys
)

# Define dataset for the data driven optimization (no physics-sampler)
dataset_train_dd = Dataset(
    batched_data_ids = ['mu', 'u_true'],
    phys_sampler = None,
    sup_data = data_loader.data_model['train'],
    batch_size_sup = batch_size_sup,
    batch_size_phys = batch_size_phys
)
dataset_val_dd = Dataset(
    batched_data_ids = ['mu', 'u_true'],
    phys_sampler = None,
    sup_data = data_loader.data_model['val'],
    batch_size_sup = batch_size_sup,
    batch_size_phys = batch_size_phys
)

# Define test dataset for evaluating the metric online
dataset_test = Dataset(
    batched_data_ids = ['mu', 'u_true'],
    phys_sampler = phys_sampler,
    sup_data = data_loader.data_model['test'],
    batch_size_sup = batch_size_sup,
    batch_size_phys = batch_size_phys,
    shuffle = False
)

# Define the dataset for trunk net training
dataset_train_trunk = Dataset(
    batched_data_ids = ['x', 'V'],
    sup_data = data_loader.data_trunk['train'],
    batch_size_sup = 10
)
dataset_val_trunk = Dataset(
    batched_data_ids = ['x', 'V'],
    sup_data = data_loader.data_trunk['val'],
    batch_size_sup = 10
)


################################################################################
# CONFIGURATION: optimization and testing
################################################################################

# Set optimizer
optimizer_dd = keras.optimizers.Adam(learning_rate = 1e-4)
optimizer_ptpi_trunk = keras.optimizers.Adam(learning_rate = 1e-3)
optimizer_ptpi_lowcost = keras.optimizers.Adam(learning_rate = 1e-3)
optimizer_ptpi_finetuning = keras.optimizers.Adam(learning_rate = 3e-4)

# Physics-informed POD DeepONets (vanilla pre-training) optimizers
optimizer_pipoddon_trunk = keras.optimizers.Adam(learning_rate = 1e-3)
optimizer_pipoddon_dd = keras.optimizers.Adam(learning_rate = 1e-3)
optimizer_pipoddon_pi = keras.optimizers.Adam(learning_rate = 3e-4)


################################################################################
# TRAINING: Data-driven approach
################################################################################

print('\n---------------------------------')
print('Data-driven approach (POD-DL-ROM)')
print('---------------------------------')

# Construct the architectures
trunk_dd, branch_dd, branch_test_dd = get_architectures()

# Model configuration
loss_weights = dict()
loss_weights['omega_sup'] = 0.5
loss_weights['omega_latent'] = 0.5
loss_weights['omega_pde'] = 0.0
loss_weights['omega_bc'] = 0.0
trainer_dd = Trainer(
    loss_model = Darcy(branch_dd, trunk_dd, pod, loss_weights)
)
trainer_dd.compile(optimizer = optimizer_dd)

# Cache the POD basis
cache_dd = dict()
cache_dd['basis'] = pod.subspaces

# Create callbacks
log_filepath = checkpoint.get_log_filepath(name = 'dd')
weights_filepath = checkpoint.get_weights_filepath(name = 'dd')
log_callback = utils.callbacks.LogCallback(log_filepath)
save_callback = utils.callbacks.SaveCallback(weights_filepath)
load_callback = utils.callbacks.LoadCallback(weights_filepath)


# Training
print('\n -> Train branch with supervised data...')
if train_or_test == 'train':
    history_dd = trainer_dd.fit(
        train_dataset = dataset_train_dd,
        val_dataset = dataset_val_dd,
        epochs = 1500, 
        callbacks = [log_callback, save_callback, load_callback],
        cache = cache_dd,
        display_options = ('train_loss', 'val_loss', 'val_metric')
    )
else:
    print('Using saved weights')
    trainer_dd.load(weights_filepath)

# Testing
test_model_dd = nns.LowRankDecNetwork(
    branch = branch_test_dd,
    trunk = pod
)
test_dict_dd = test_utils.test(
    model = test_model_dd,
    data_loader = data_loader
)

# Safe deletion
del trainer_dd, trunk_dd, branch_dd, branch_test_dd


################################################################################
# TRAINING: PI-POD-DeepONets approach
################################################################################

# Print info
print('\n---------------------------------------')
print('Physics-Informed POD DeepONets approach')
print('---------------------------------------')

# Construct the architectures
trunk_pipoddon, branch_pipoddon, branch_test_pipoddon = get_architectures()


# PRE-TRAINING #################################################################

# Trunk network
trainer_trunk_pipoddon = Trainer(
    loss_model = utils.losses.L2LossFunction(
        model = trunk_pipoddon,
        input_id = 'x',
        output_id = 'V'
    )
)
trainer_trunk_pipoddon.compile(optimizer = optimizer_pipoddon_trunk)

# Create callbacks
log_filepath = checkpoint.get_log_filepath(name = 'trunk-pipoddon')
weights_filepath = checkpoint.get_weights_filepath(name = 'trunk-pipoddon')
log_callback = utils.callbacks.LogCallback(log_filepath)
save_callback = utils.callbacks.SaveCallback(weights_filepath)
load_callback = utils.callbacks.LoadCallback(weights_filepath)

# Training
if train_or_test == 'train':
    print('\n -> Pre-training trunk to approximate POD basis...')
    history_trunk_pipoddon = trainer_trunk_pipoddon.fit(
        train_dataset = dataset_train_trunk,
        val_dataset = dataset_val_trunk,
        epochs = 500, 
        callbacks = [log_callback, save_callback, load_callback],
        display_options = ('train_loss', 'val_loss')
    )
else:
    print('Using saved weights')
    trainer_trunk_pipoddon.load(weights_filepath)
test_utils.test_trunk(trunk_pipoddon, data_loader)



# FINE-TUNING ##################################################################

# Model configuration
loss_weights = dict()
loss_weights['omega_sup'] = 0.5
loss_weights['omega_latent'] = 0.5
loss_weights['omega_pde'] = 50.0
loss_weights['omega_bc'] = 100.0
trainer_pipoddon = Trainer(
    loss_model = Darcy(
        branch_pipoddon, trunk_pipoddon, pod, loss_weights
    )
)
trainer_pipoddon.compile(optimizer = optimizer_pipoddon_pi)

# Create callbacks
log_filepath = checkpoint.get_log_filepath(name = 'pipoddon')
weights_filepath = checkpoint.get_weights_filepath(name = 'pipoddon')
log_callback = utils.callbacks.LogCallback(log_filepath)
save_callback = utils.callbacks.SaveCallback(weights_filepath)
load_callback = utils.callbacks.LoadCallback(weights_filepath)


# Training
print('\n -> Fine-tuning (branch + trunk) with (data + physics)...')
if train_or_test == 'train':
    history_pipoddon = trainer_pipoddon.fit(
        train_dataset = dataset_train_whole,
        val_dataset = dataset_val_whole,
        test_dataset = dataset_test,
        epochs = 2000, 
        callbacks = [log_callback, save_callback, load_callback],
        display_options = ('train_loss', 'val_loss', 'test_metric')
    )
else:
    trainer_pipoddon.load(weights_filepath)
    print('Using saved weights')

# Testing
test_model_pipoddon = nns.LowRankDecNetwork(
    branch = branch_test_pipoddon,
    trunk = trunk_pipoddon
)
test_dict_pipoddon = test_utils.test(
    model = test_model_pipoddon,
    data_loader = data_loader
)

# Safe deletion
del trainer_pipoddon, trainer_trunk_pipoddon
del trunk_pipoddon, branch_pipoddon, branch_test_pipoddon



################################################################################
# TRAINING: PTPI-DL-ROM approach (low-cost training + fine-tuning)
################################################################################

print('\n--------------------')
print('PTPI-DL-ROM approach')
print('--------------------')

# Construct the architectures
trunk_ptpi, branch_ptpi, branch_test_ptpi = get_architectures()


# TRUNK NETWORK PRE-TRAINING ###################################################

# Trunk network
trainer_trunk_ptpi = Trainer(
    utils.losses.L2LossFunction(
        model = trunk_ptpi,
        input_id = 'x',
        output_id = 'V'
    )
)
trainer_trunk_ptpi.compile(optimizer = optimizer_ptpi_trunk)

# Create callbacks
log_filepath = checkpoint.get_log_filepath(name = 'trunk-ptpi')
weights_filepath = checkpoint.get_weights_filepath(name = 'trunk-ptpi')
log_callback = utils.callbacks.LogCallback(log_filepath)
save_callback = utils.callbacks.SaveCallback(weights_filepath)
load_callback = utils.callbacks.LoadCallback(weights_filepath)

# Training
print('\n -> Pre-training trunk to approximate POD basis...')
if train_or_test == 'train':
    history_trunk_ptpi = trainer_trunk_ptpi.fit(
        train_dataset = dataset_train_trunk,
        val_dataset = dataset_val_trunk,
        epochs = 300, 
        callbacks = [log_callback, save_callback, load_callback],
        display_options = ('train_loss', 'val_loss')
    )
else:
    print('Using saved weights')
    trainer_trunk_ptpi.load(weights_filepath)
test_utils.test_trunk(trunk_ptpi, data_loader)


# BRANCH NETWORK PRE-TRAINING ##################################################

# Declare trunk not trainable
trainer_trunk_ptpi.set_trainability(trainability = False)

# Model configuration
loss_weights = dict()
loss_weights['omega_sup'] = 0.5
loss_weights['omega_latent'] = 0.5
loss_weights['omega_pde'] = 50.0
loss_weights['omega_bc'] = 100.0
trainer_ptpi = Trainer(
    loss_model = Darcy(
        branch_ptpi, trunk_ptpi, pod, loss_weights
    )
)
trainer_ptpi.compile(optimizer = optimizer_ptpi_lowcost)

# Caching computations 
def get_cache():
    cache = dict()
    x_sup = data_loader.data_model['train']['x']
    left = (x_sup[:,0] == 0).astype('bool')
    right = (x_sup[:,0] == 3).astype('bool')
    up_down = ((x_sup[:,1] == 0) + (x_sup[:,1] == 1)).astype('bool')
    cache['basis'] = trunk_ptpi(x_sup)
    cache['x_bc_p'] = x_sup[up_down]
    cache['basis_bc_p'] = trunk_ptpi(x_sup[up_down])
    cache['basis_bc_left'] = trunk_ptpi(x_sup[left])
    cache['basis_bc_right'] = trunk_ptpi(x_sup[right])
    cache['x_pde'] = x_sup[~(left * right * up_down)]
    cache['basis_pde'] = trunk_ptpi(x_sup[~(left * right * up_down)])
    grads = get_grad(
        trunk = trunk_ptpi,
        x = x_sup[~(left * right * up_down)]
    )
    cache['grads'] = grads

    return cache

cache_ptpi = get_cache()


# Create callbacks
log_filepath = checkpoint.get_log_filepath(name = 'ptpi-lowcost')
weights_filepath = checkpoint.get_weights_filepath(name = 'ptpi-lowcost')
log_callback = utils.callbacks.LogCallback(log_filepath)
save_callback = utils.callbacks.SaveCallback(weights_filepath)
load_callback = utils.callbacks.LoadCallback(weights_filepath)

# Training
print('\n -> Pre-training branch with (data + physics)...')
if train_or_test == 'train':
    history_ptpi = trainer_ptpi.fit(
        train_dataset = dataset_train_whole,
        val_dataset = dataset_val_whole,
        test_dataset = dataset_test,
        epochs = 200, 
        callbacks = [log_callback, save_callback, load_callback],
        cache = cache_ptpi,
        display_options = ('train_loss', 'val_loss', 'test_metric')
    )
else:
    trainer_ptpi.load(weights_filepath)
    print('Using saved weights')

# Testing
test_model_ptpi = nns.LowRankDecNetwork(
    branch = branch_test_ptpi,
    trunk = trunk_ptpi
)
test_dict_ptpi_lowcost = test_utils.test(
    model = test_model_ptpi,
    data_loader = data_loader
)


# FINE-TUNING ##################################################################

# Declare trunk trainable
trainer_trunk_ptpi.set_trainability(trainability = True)

# Compile the model
trainer_ptpi.compile(optimizer = optimizer_ptpi_finetuning)

# Create callbacks
log_filepath = checkpoint.get_log_filepath(name = 'ptpi-finetuned')
weights_filepath = checkpoint.get_weights_filepath(name = 'ptpi-finetuned')
log_callback = utils.callbacks.LogCallback(log_filepath)
save_callback = utils.callbacks.SaveCallback(weights_filepath)
load_callback = utils.callbacks.LoadCallback(weights_filepath)

# Training
print('\n -> Fine-tuning (branch + trunk) with (data + physics)...')
if train_or_test == 'train':
    history_ptpi = trainer_ptpi.fit(
        train_dataset = dataset_train_whole,
        val_dataset = dataset_val_whole,
        test_dataset = dataset_test,
        epochs = 2000, 
        callbacks = [log_callback, save_callback, load_callback],
        display_options = ('train_loss', 'val_loss', 'test_metric')
    )
else:
    print('Using saved weights')
    trainer_ptpi.load(weights_filepath)

# Testing
test_dict_ptpi_finetuned = test_utils.test(
    model = test_model_ptpi,
    data_loader = data_loader
)

# Safe deletion
del trainer_ptpi, trunk_ptpi, branch_ptpi, branch_test_ptpi



################################################################################
# POSTPROCESSING: Comparison between the employed approaches
################################################################################

def postprocess_comparison(dir, cmap, mu_instances):
    """ Compares the Data-driven and Physics-Informed approaches.

    Args:
        dir (str): the save folder
        cmap: the colormap
        mu_instances (list[int]): Parameter instances' index.
    """

    # Looping over the channels
    for k in range(c+1):

        # Creating the figure
        fig, axs = plt.subplots(
            nrows = len(mu_instances), ncols = 4, figsize = (17,19)
        )

        # Looping over the instances
        for i in range(len(mu_instances)):

            # Gather quantities to plot
            instance_idx = mu_instances[i]
            x = data_loader.data_model['test']['x']
            mu_instance = data_loader.data_model['test']['mu'][instance_idx]
            if k < 3:
                true_instance = data_loader.data_model['test']['u_true'][:,:,k][
                    instance_idx]
                pred_instance_dd = test_dict_dd['u_pred'][:,:,k][instance_idx]
                pred_instance_lowcost = test_dict_ptpi_lowcost['u_pred'][:,:,k][
                    instance_idx]
                pred_instance_finetuned = test_dict_ptpi_finetuned['u_pred'][:,:,k][
                    instance_idx]
            else:
                def get_magnitude(item : np.array):
                    """ The function to compute magnitude with.

                    Args: 
                        item (np.array): Item to extract the channels from.
                    """
                    u = item[:,0][:,None]
                    v = item[:,1][:,None]
                    item = np.linalg.norm(
                        np.concatenate((u, v), axis = 1),
                        axis = 1
                    )
                    return item
                true_instance = get_magnitude(
                    data_loader.data_model['test']['u_true'][:,:,:2][
                        instance_idx])
                pred_instance_dd = get_magnitude(
                    test_dict_dd['u_pred'][:,:,:2][instance_idx]
                )
                pred_instance_lowcost = get_magnitude(
                    test_dict_ptpi_lowcost['u_pred'][:,:,:2][instance_idx]
                )
                pred_instance_finetuned = get_magnitude(
                    test_dict_ptpi_finetuned['u_pred'][:,:,:2][instance_idx]
                )
            
            # Interpolating on a grid
            x, y = [x[:,i] for i in range(x.shape[1])]
            x_grid, y_grid = np.mgrid[0:3:1000j, 0:1:1000j]
            true_grid = griddata(
                (x, y), true_instance, (x_grid, y_grid)
            ).T
            pred_dd_grid= griddata(
                (x, y), pred_instance_dd, (x_grid, y_grid)
            ).T
            pred_lowcost_grid = griddata(
                (x, y), pred_instance_lowcost, (x_grid, y_grid)
            ).T
            pred_finetuned_grid = griddata(
                (x, y), pred_instance_finetuned, (x_grid, y_grid)
            ).T

            # Visualizing
            im_true = axs[i,0].imshow(
                true_grid.T, 
                extent = (0,1,0,3), 
                origin='lower',
                cmap = cmap 
            )
            im_dd = axs[i,1].imshow(
                pred_dd_grid.T, 
                extent = (0,1,0,3), 
                origin='lower',
                cmap = cmap,
                vmin = np.min(true_grid),
                vmax = np.max(true_grid)
            )
            im_lowcost = axs[i,2].imshow(
                pred_lowcost_grid.T, 
                extent = (0,1,0,3), 
                origin='lower',
                cmap = cmap,
                vmin = np.min(true_grid),
                vmax = np.max(true_grid)
            )
            im_finetuned = axs[i,3].imshow(
                pred_finetuned_grid.T, 
                extent = (0,1,0,3), 
                origin='lower',
                cmap = cmap,  
                vmin = np.min(true_grid),
                vmax = np.max(true_grid)
            )

            # Adjusting the plot, the ticks and the colorbar
            for j in range(4):
                axs[i,j].axes.get_xaxis().set_ticks([])
                axs[i,j].axes.get_yaxis().set_ticks([])
            plt.subplots_adjust(
                left = 0.04, 
                right = 0.91, 
                hspace = 0.15, 
                wspace = 0.2, 
                top = 0.93, 
                bottom = 0.02
            )
            cbar = utils.plot_utils.im_colorbar(im_finetuned, spacing = 0.03)
            cbar.ax.tick_params(labelsize = 20)

            # Inserting row titles
            rounded_mu = np.round(mu_instance, decimals = 2)
            row_title = r'${\bf{\mu}}$' + " = (" + str(rounded_mu[0]) + \
                ',' + str(rounded_mu[1]) + ')'
            axs[i,0].set_ylabel(
                row_title, 
                rotation = 'vertical', 
                fontsize = 25,
                labelpad = 20
            )

        # Inserting column titles
        col_title = 'Exact' % mu_instance[-1]
        axs[0,0].set_title(col_title, fontsize = 25, pad = 25)
        axs[0,1].set_title(
            'POD-DL-ROM\n(data-driven)', fontsize = 25, pad = 25
        )
        axs[0,2].set_title(
            'PTPI-DL-ROM\n(after pre-training)',fontsize = 25, pad = 25
        )
        axs[0,3].set_title(
            'PTPI-DL-ROM\n(fine-tuned)',fontsize = 25, pad = 25
        )
        plt.savefig(os.path.join(
            dir, 'instance_darcy_comparison_channel_' + str(k) + '.png')
        )


# Call postprocessing function
plt.style.use('default')
postprocess_comparison(results_dir, 'jet', mu_instances = (1,31,63))
plt.style.use('dark_background')
postprocess_comparison(slides_dir, 'gist_rainbow', mu_instances = (1,31,63))


################################################################################
# POSTPROCESSING: Error analysis
################################################################################


def error_analysis(dir : str, info_colors : dict):
    """ Plots the log10 of the relative error in the parameter space.

    Args:
        dir (str): the save directory
        info_colors (dict): info about coloring pattern
    """

    # Creating the figure
    fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (14.0, 6.0))
    images = list()

    # Getting unique values of the parameters
    train_mu_unique = np.unique(
        data_loader.data_model['train']['mu'], 
        axis = 0
    )
    test_mu_unique = np.unique(
        data_loader.data_model['test']['mu'], 
        axis = 0
    )


    # Compute minimum and maximum values to set the rectangle dimensions
    min_train = np.min(train_mu_unique, axis = 0)
    max_train = np.max(train_mu_unique, axis = 0)
    min_test = np.min(test_mu_unique, axis = 0)
    max_test = np.max(test_mu_unique, axis = 0)
    width_train, height_train = max_train - min_train
    width_test, height_test = max_test - min_test

    # Prepare iterables for the for loop
    enum = enumerate([test_dict_dd, test_dict_ptpi_lowcost, test_dict_ptpi_finetuned])
    titles = [
        'POD-DL-ROM\n(data-driven)', 
        'PTPI-DL-ROM\n(after pre-training)',
        'PTPI-DL-ROM\n(fine-tuned)' 
    ]

    # Set color limits
    colorlim_min = np.min(np.log10(test_dict_ptpi_finetuned['err_param']))
    colorlim_max = np.max(np.log10(test_dict_dd['err_param']))

    # Iteration over the paradigms
    for (count, item) in enum:


        # Plot the various parameter spaces 
        rectangle_test = Rectangle(
            min_test, 
            width_test, 
            height_test, 
            color = info_colors[0], 
            alpha = 1.0,
            fill = False,
            linewidth = 3,
            linestyle = '--',
            label = '$\mathcal{P}_{test}$'
        )
        rectangle_train = Rectangle(
            min_train, 
            width_train, 
            height_train, 
            color = info_colors[1], 
            alpha = 1.0,
            fill = False,
            linewidth = 3,
            linestyle = '--',
            label = '$\mathcal{P}_{sup}$'
        )
        rectangle_phys = Rectangle(
            (phys_sampler.mu_1_min, phys_sampler.mu_2_min), 
            phys_sampler.mu_1_max - phys_sampler.mu_1_min, 
            phys_sampler.mu_2_max - phys_sampler.mu_2_min, 
            color = info_colors[2], 
            alpha = 1.0,
            fill = False,
            linewidth = 3,
            linestyle = '--',
            label = '$\mathcal{P}_{res}$'
        )
        axs[count].add_patch(rectangle_test)
        axs[count].add_patch(rectangle_train)
        axs[count].add_patch(rectangle_phys)

        # Interpolate and plot the log10 of the relative error
        mu_1_grid, mu_2_grid = np.mgrid[
            min_test[0]:max_test[0]:1000j, min_test[1]:max_test[1]:1000j
        ]
        err_grid = griddata(
            (test_mu_unique[:,0],  test_mu_unique[:,1]), 
            np.log10(item['err_param']), 
            (mu_1_grid, mu_2_grid)
            ).T
        im_err = axs[count].imshow(
            err_grid, 
            extent = (min_test[0], max_test[0], min_test[1], max_test[1]), 
            origin='lower',
            cmap = 'coolwarm'
        )
        images.append(im_err)

        # Set current image title
        title = titles[count]
        axs[count].set_title(title, fontsize = 18, pad = 25)
        axs[count].set_xlabel('$\mu_1$', fontsize = 25)
        axs[count].set_ylabel('$\mu_2$', fontsize = 25)
        axs[count].axis('equal')
        images[count].set_clim(colorlim_min, colorlim_max)

        # Control ticks to be displayed
        xticks_err = np.round(
            np.linspace(min_test[0], max_test[0], 4), decimals = 2
        )
        yticks_err = np.round(
            np.linspace(min_test[1], max_test[1], 4), decimals = 2
        )
        axs[count].set_xticks(xticks_err)
        axs[count].set_yticks(yticks_err)
        axs[count].set_xticklabels(
            labels = xticks_err,
            fontsize = 17
        )
        axs[count].set_yticklabels(
            labels = yticks_err, 
            fontsize = 17
        )

        # Adjust the plot
        plt.subplots_adjust(
            left = 0.1, 
            right = 0.82, 
            hspace = 0.2, 
            wspace = 0.5, 
            top = 0.85,
            bottom = 0.15
        )
    
    # Set legend and colorbar
    plt.legend(bbox_to_anchor=(1.04, -0.20), loc='lower left', fontsize = 25)
    cbar = utils.plot_utils.im_colorbar(
        im = images[0], 
        im_cax = images[2], 
        spacing = 0.05, 
        start_bottom = 0.19
    )
    cbar.ax.tick_params(labelsize = 18)

    # Save the figure
    plt.savefig(os.path.join(dir, 'errors_darcy.png'))


# Call error analysis function
info_colors = ['cyan', 'fuchsia', 'lime']
plt.style.use('dark_background')
error_analysis(slides_dir, info_colors)

info_colors = ['blue', 'fuchsia', 'lime']
plt.style.use('default')
error_analysis(results_dir, info_colors)


################################################################################
# POSTPROCESSING: Training efficiency
################################################################################

def training_efficiency():
    
    def readfile(architecture_name):
        filepath = checkpoint.get_log_filepath(name = architecture_name)
        csvfile = open(filepath, newline='')
        reader = csv.reader(csvfile, delimiter=',')
        lines = [rows for rows in reader]
        keys = lines[0]
        all_values = np.array(lines[1:]).astype('float32')
        info = {keys[idx] : all_values[:,idx] for idx in range(len(keys))}
        return info
    
    trunk_pipoddon_info = readfile(architecture_name = 'trunk-pipoddon')
    pipoddon_info = readfile(architecture_name = 'pipoddon')
    trunk_ptpi_info = readfile(architecture_name = 'trunk-ptpi')
    ptpi_lowcost_info = readfile(architecture_name = 'ptpi-lowcost')
    ptpi_finetuned_info = readfile(architecture_name = 'ptpi-finetuned')

    elapsed_time_pretraining_pipoddon = trunk_pipoddon_info['elapsed_time'][-1]
    elapsed_time_pretraining_ptpi = trunk_ptpi_info['elapsed_time'][-1] + \
        ptpi_lowcost_info['elapsed_time'][-1]
    time_finetuning_pipoddon = elapsed_time_pretraining_pipoddon + \
        pipoddon_info['elapsed_time']
    time_finetuning_ptpi = elapsed_time_pretraining_ptpi + \
            ptpi_finetuned_info['elapsed_time']
    
    fig, ax = plt.subplots(figsize = (12,7))
    handles_l1 = []
    handles_l2 = []
    labels_paradigms = []
    labels_endtimes = []

    # Relative error plots
    plt_pipoddon, = plt.semilogy(
        time_finetuning_pipoddon, 
        pipoddon_info['test_metric'],
        'darkred',
        linewidth = 2
    )
    plt_ptpi, = plt.semilogy(
        time_finetuning_ptpi, 
        ptpi_finetuned_info['test_metric'],
        'b',
        linewidth = 2
    )
    handles_l1.append(plt_pipoddon)
    handles_l1.append(plt_ptpi)

    # End pre-training line
    plt_time_pipoddon = plt.axvline(
        x = time_finetuning_pipoddon[0], 
        color = 'darkred', 
        linestyle = ':', 
        linewidth = 2
    )
    plt_time_ptpi = plt.axvline(
        x = time_finetuning_ptpi[0], 
        color = 'b', 
        linestyle = ':', 
        linewidth = 2
    )
    handles_l2.append(plt_time_pipoddon)
    handles_l2.append(plt_time_ptpi)

    # Start finetuning point
    plt.scatter(
        time_finetuning_pipoddon[0],
        pipoddon_info['test_metric'][0],
        marker = "D",
        c = "k",
        s = 200
    )
    plt.scatter(
        time_finetuning_ptpi[0],
        ptpi_finetuned_info['test_metric'][0],
        marker = "D",
        c = "k",
        s = 200
    )
    
    # Legends
    labels_paradigms.append('PI-POD-DeepONets')
    labels_endtimes.append(
        'PI-POD-DeepONets \n pre-training endtime'
    )
    labels_paradigms.append('PTPI-DL-ROM')
    labels_endtimes.append('PTPI-DL-ROM \npre-training endtime')
    l1 = plt.legend(
        handles = handles_l1,
        labels = labels_paradigms,
        bbox_to_anchor = (0, 1.02, 1, 0.2), 
        loc = "lower left",
        mode = "expand", 
        borderaxespad = 0, 
        ncol = 4, 
        fontsize = 18
    ) 
    l2 = plt.legend(
        handles = handles_l2,
        labels = labels_endtimes,
        bbox_to_anchor = (0, -0.36, 1, 0.2), 
        loc = "lower left",
        mode = "expand", 
        borderaxespad = 0, 
        ncol = 3, 
        fontsize = 18
    )  
    plt.subplots_adjust(
        left = 0.08, 
        right = 0.98, 
        hspace = 0.2, 
        wspace = 0.5, 
        top = 0.90,
        bottom = 0.26
    )
    ax.add_artist(l1)
    ax.add_artist(l2)
    
    
    # Save plot
    plt.xlabel('Elapsed time [$s$]', fontsize = 17)
    xlim_max = max(time_finetuning_pipoddon[-1], time_finetuning_ptpi[-1])
    plt.xlim([0, 1.1 * xlim_max])
    plt.ylabel('Test relative error $\mathcal{E}$', fontsize = 17)
    plt.gca().axes.tick_params(labelsize = 14) 
    plt.savefig(os.path.join(results_dir, 'training_efficiency.png'))



training_efficiency()