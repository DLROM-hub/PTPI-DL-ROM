################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Implementation of the ADR 3D test case <-
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
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.40'

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
pod_dim = 15
latent_dim = 8
c = 1
d = 3
p = 3
N_t_sup = 70
N_t_test = 70

# Set up directories
parent_dir =  os.path.join('..', '..', 'data_pi', 'adr-3d')
results_dir = os.path.join('..', '..', 'results', 'adr-3d')
slides_dir = os.path.join('..', '..', 'results', 'adr-3d', 'slides')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(slides_dir):
    os.makedirs(slides_dir)
checkpoint = utils.Checkpoint(save_folder = results_dir)

# Collect data filenames
filenames = dict()
filenames['mu'] = os.path.join(parent_dir, 'params_train_adr.mat')
filenames['u_true'] = os.path.join(parent_dir, 'S_U_train_adr.mat')
filenames['x'] = os.path.join(parent_dir, 'loc_points.mat')
filenames_test = dict()
filenames_test['mu'] = os.path.join(parent_dir, 'params_test_adr.mat')
filenames_test['u_true'] = os.path.join(parent_dir, 'S_U_test_adr.mat')
filenames_test['x'] = os.path.join(parent_dir, 'loc_points.mat')

# Train or test
train_or_test = sys.argv[1]
assert (train_or_test == 'train' or train_or_test == 'test')


################################################################################
# CONFIGURATION: Data and sampling
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
        self.mu_1_min = 0.4
        self.mu_1_max = 2.1
        self.mu_2_min = 0.4
        self.mu_2_max = 2.1
        self.n_resample = n_resample

    def __call__(self):
        data = dict()
        mu_1 = self.mu_1_min + (self.mu_1_max - self.mu_1_min) * \
            np.random.rand(self.N_s,1)
        mu_2 = self.mu_2_min + (self.mu_2_max - self.mu_2_min) * \
            np.random.rand(self.N_s,1)
        time = 7.1 * np.pi * np.random.rand(self.N_s,1)
        data['mu'] = np.concatenate((mu_1, mu_2, time), axis = 1)
        data['x'] = dict()
        data['x']['x_pde'] = np.random.rand(300,3)
        x_sup = data_loader.data_model['train']['x']
        x = x_sup[:,0]
        y = x_sup[:,1]
        z = x_sup[:,2]
        is_at_boundary = (x == 0) + (x == 1) + (y == 0) + (y == 1) + \
              (z == 0) + (z == 1)
        is_at_boundary = is_at_boundary.astype('bool')
        data['x']['x_bc'] = x_sup[is_at_boundary]
        return data

phys_sampler = PhysicsSampler(n_resample = 5)

# Define batch dimension
batch_size_sup = 50
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
    batch_size_sup = 100
)
dataset_val_trunk = Dataset(
    batched_data_ids = ['x', 'V'],
    sup_data = data_loader.data_trunk['val'],
    batch_size_sup = 100
)



################################################################################
# CONFIGURATION: Function to construct the neural network architectures
################################################################################

def get_architectures():
    # Define trunk components
    dense_trunk = nns.DenseNetwork(
        width = 100, 
        depth = 7, 
        output_dim = pod_dim * c,
        activation = jnp.sin,
        kernel_initializer = 'glorot_uniform'
    )
    reshape_trunk = keras.layers.Reshape(target_shape = (pod_dim, c))

    # Build trunk network
    trunk_arch = [dense_trunk, reshape_trunk]
    trunk = nns.ops.hstack_nns(
        input_dims = (d, ),
        blocks = trunk_arch
    )

    # Define branch components
    reduced_network = nns.DenseNetwork(
        width = 50, 
        depth = 5, 
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
        width = 50, 
        depth = 5, 
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
        architecture = branch_arch
    )

    return trunk, branch, branch_test


################################################################################
# CONFIGURATION: Definition of the function to compute gradients with
################################################################################

def get_grad(trunk, x):
    n_spatial_dims = x.shape[1]
    grads = list()
    lapl_t = 0.0
    for ix in range(n_spatial_dims):
        get_scalar_output = lambda x, y, z: trunk(
            jnp.concatenate((x[:,None],y[:,None],z[:,None]), axis = 1)
        )[0]
        grad_op = jax.vmap(jax.jacrev(get_scalar_output, argnums = ix))
        grads +=  (
            grad_op(x[:,0][:,None], x[:,1][:,None], x[:,2][:,None])[:,:,0], 
        )
        lapl_op = jax.vmap(
            jax.jacfwd(
                jax.jacrev(get_scalar_output, argnums = ix), 
                argnums = ix
            )
        )
        lapl_t += lapl_op(
            x[:,0][:,None], x[:,1][:,None], x[:,2][:,None]
        )[:,:,0,0]
    return lapl_t, grads


################################################################################
# CONFIGURATION: Definition of the physical model
################################################################################

class Adr(utils.losses.LossFunction):

    def __init__(self, branch, trunk, pod, loss_weights):
        super(Adr, self).__init__()
        self.trunk = trunk
        self.branch = branch
        self.pod = pod
        self.loss_weights = loss_weights
        self.forcing_fn = lambda params, x: \
            jnp.exp(3 * x[:,0] * x[:,1] * x[:,2]) * \
            jnp.sin(jnp.pi * params[0] * x[:,0]) * \
            jnp.sin(jnp.pi * params[1] * x[:,1])
        self.forcing_fn = jax.vmap(self.forcing_fn, in_axes = (0,None))

        
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
                x_pde = x_phys.get('x_pde')
                x_bc = x_phys.get('x_bc')
                t_sup = self.trunk(x_sup)
                t_bc = self.trunk(x_bc)
                t_pde = self.trunk(x_pde)
                lapl_t, grads = get_grad(trunk = self.trunk, x = x_pde)
                dt_dx, _, _ = grads
            else: 
                t_sup = self.trunk(x_sup)
        else:
            lapl_t = cache.get('lapl_t')
            t_sup = cache.get('basis')
            t_bc = cache.get('basis_bc')
            x_pde = cache.get('x_pde')
            dt_dx = cache.get('dt_dx')
            t_pde = cache.get('basis_pde')

        # Supervised loss
        u_sup_true_proj = self.pod.project(u_sup_true)
        b_sup, reduced_network_repr, latent_repr = \
            self.branch((mu_sup, u_sup_true_proj))
        u_sup = keras.ops.einsum('bjc,ijc->bic', b_sup, t_sup)
        l_sup = keras.ops.mean(keras.ops.sum((u_sup - u_sup_true)**2, axis = 1))
        loss =  self.loss_weights['omega_sup'] * l_sup 

        # Eventually compute physics-based loss
        if self.loss_weights['omega_pde'] + self.loss_weights['omega_bc'] > 0:
            dummy = nns.ops.get_dlrom_dummy(
                branch = self.branch, 
                mu_phys = mu_phys
            )
            b_phys, _, _ = self.branch((mu_phys, dummy))
            b_prime = nns.ops.get_time_derivative(
                branch = self.branch,
                mu_phys = mu_phys,
                tau = 0.001
            )
            u_prime = keras.ops.einsum('bjc,ijc->bic', b_prime, t_pde)
            forcing = self.forcing_fn(mu_phys, x_pde)[:,:,None]
            lapl_u = keras.ops.einsum('bjc,ijc->bic', b_phys, lapl_t)
            du_dx =  keras.ops.einsum('bjc,ijc->bic', b_phys, dt_dx)
            u = keras.ops.einsum('bjc,ijc->bic', b_phys, t_pde)
            time = mu_phys[:,2][:,None,None]
            a = jnp.log(0.1 * time)
            res_pde = u_prime - 0.05 * lapl_u + 0.05 * u + a * du_dx - forcing
            l_pde = keras.ops.mean(res_pde**2)
            u_bc = keras.ops.einsum('bjc,ijc->bic', b_phys, t_bc)
            l_bc = keras.ops.mean(u_bc**2)
            loss += self.loss_weights['omega_pde'] * l_pde + \
                self.loss_weights['omega_bc'] * l_bc
        
        # Eventually adding latent loss
        if self.loss_weights['omega_latent'] > 0:
            l_latent = keras.ops.mean((reduced_network_repr - latent_repr)**2)
            loss += self.loss_weights['omega_latent'] * l_latent 

        metric = utils.metrics.mean_rel_err_metric(u_sup_true, u_sup)

        return {'loss' : loss, 'metric' : metric}




################################################################################
# CONFIGURATION: optimization and testing
################################################################################

# Data driven optimizer
optimizer_dd = keras.optimizers.Adam(learning_rate = 1e-3)

# Physics-informed (no pre-training) optimizers
optimizer_nopt = keras.optimizers.Adam(learning_rate = 1e-4)

# Physics-informed (vanilla pre-training) optimizers
optimizer_vanpt_dd = keras.optimizers.Adam(learning_rate = 1e-3)
optimizer_vanpt_pi = keras.optimizers.Adam(learning_rate = 1e-4)

# Physics-informed POD DeepONets (vanilla pre-training) optimizers
optimizer_pipoddon_trunk = keras.optimizers.Adam(learning_rate = 1e-3)
optimizer_pipoddon_dd = keras.optimizers.Adam(learning_rate = 1e-3)
optimizer_pipoddon_pi = keras.optimizers.Adam(learning_rate = 1e-4)

# PTPI-DL-ROMs optimizers
optimizer_ptpi_trunk = keras.optimizers.Adam(learning_rate = 1e-3)
optimizer_ptpi_lowcost = keras.optimizers.Adam(learning_rate = 1e-3)
optimizer_ptpi_finetuning = keras.optimizers.Adam(learning_rate = 1e-4)


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
    loss_model = Adr(branch_dd, trunk_dd, pod, loss_weights)
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
        epochs = 2000, 
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
# TRAINING: Non-pre-trained Physics-Informed approach
################################################################################

# Print info
print('\n-----------------------------------------')
print('Non-pre-trained Physics-Informed approach')
print('-----------------------------------------')

# Construct the architectures
trunk_nopt, branch_nopt, branch_test_nopt = get_architectures()

# Model configuration
loss_weights = dict()
loss_weights['omega_sup'] = 0.5
loss_weights['omega_latent'] = 0.5
loss_weights['omega_pde'] = 0.5
loss_weights['omega_bc'] = 50.0
trainer_nopt = Trainer(
    loss_model = Adr(
        branch_nopt, trunk_nopt, pod, loss_weights
    )
)
trainer_nopt.compile(optimizer = optimizer_nopt)

# Create callbacks
log_filepath = checkpoint.get_log_filepath(name = 'nopt')
weights_filepath = checkpoint.get_weights_filepath(name = 'nopt')
log_callback = utils.callbacks.LogCallback(log_filepath)
save_callback = utils.callbacks.SaveCallback(weights_filepath)
load_callback = utils.callbacks.LoadCallback(weights_filepath)


# Training
print('\n -> Fine-tuning (branch + trunk) with (data + physics)...')
if train_or_test == 'train':
    history_nopt = trainer_nopt.fit(
        train_dataset = dataset_train_whole,
        val_dataset = dataset_val_whole,
        test_dataset = dataset_test,
        epochs = 7000, 
        callbacks = [log_callback, save_callback, load_callback],
        display_options = ('train_loss', 'val_loss', 'test_metric')
    )
else:
    trainer_nopt.load(weights_filepath)
    print('Using saved weights')

# Testing
test_model_nopt = nns.LowRankDecNetwork(
    branch = branch_test_nopt,
    trunk = trunk_nopt
)
test_dict_nopt = test_utils.test(
    model = test_model_nopt,
    data_loader = data_loader
)

# Safe deletion
del trainer_nopt
del trunk_nopt, branch_nopt, branch_test_nopt


################################################################################
# TRAINING: Vanilla Pre-trained Physics-Informed approach
################################################################################

# Print info
print('\n---------------------------------------------')
print('Vanilla Pre-trained Physics-Informed approach')
print('---------------------------------------------')

# Construct the architectures
trunk_vanpt, branch_vanpt, branch_test_vanpt = get_architectures()

# PRE-TRAINING #################################################################

# Initial model configuration
loss_weights = dict()
loss_weights['omega_sup'] = 0.5
loss_weights['omega_latent'] = 0.5
loss_weights['omega_pde'] = 0.0
loss_weights['omega_bc'] = 0.0
trainer_vanpt = Trainer(
    loss_model = Adr(branch_vanpt, trunk_vanpt, pod, loss_weights)
)
trainer_vanpt.compile(optimizer = optimizer_vanpt_dd)

# Create callbacks
log_filepath = checkpoint.get_log_filepath(name = 'vanpt-pretraining')
weights_filepath = checkpoint.get_weights_filepath(name = 'vanpt-pretraining')
log_callback = utils.callbacks.LogCallback(log_filepath)
save_callback = utils.callbacks.SaveCallback(weights_filepath)
load_callback = utils.callbacks.LoadCallback(weights_filepath)

# Training
print('\n -> Pre-training (branch + trunk) with supervised data...')
if train_or_test == 'train':
    history_vanpt = trainer_vanpt.fit(
        train_dataset = dataset_train_dd,
        val_dataset = dataset_val_dd,
        epochs = 3000, 
        callbacks = [log_callback, save_callback, load_callback],
        cache = None,
        display_options = ('train_loss', 'val_loss', 'val_metric')
    )
else:
    print('Using saved weights')
    trainer_vanpt.load(weights_filepath)

# FINE-TUNING ##################################################################

# Model configuration
loss_weights = dict()
loss_weights['omega_sup'] = 0.5
loss_weights['omega_latent'] = 0.5
loss_weights['omega_pde'] = 0.5
loss_weights['omega_bc'] = 50.0
trainer_vanpt = Trainer(
    loss_model = Adr(
        branch_vanpt, trunk_vanpt, pod, loss_weights
    )
)
trainer_vanpt.compile(optimizer = optimizer_vanpt_pi)

# Create callbacks
log_filepath = checkpoint.get_log_filepath(name = 'vanpt-finetuning')
weights_filepath = checkpoint.get_weights_filepath(name = 'vanpt-finetuning')
log_callback = utils.callbacks.LogCallback(log_filepath)
save_callback = utils.callbacks.SaveCallback(weights_filepath)
load_callback = utils.callbacks.LoadCallback(weights_filepath)

# Training
print('\n -> Fine-tuning (branch + trunk) with (data + physics)...')
if train_or_test == 'train':
    history_vanpt = trainer_vanpt.fit(
        train_dataset = dataset_train_whole,
        val_dataset = dataset_val_whole,
        test_dataset = dataset_test,
        epochs = 4000, 
        callbacks = [log_callback, save_callback, load_callback],
        display_options = ('train_loss', 'val_loss', 'test_metric')
    )
else:
    trainer_vanpt.load(weights_filepath)
    print('Using saved weights')

# Testing
test_model_vanpt = nns.LowRankDecNetwork(
    branch = branch_test_vanpt,
    trunk = trunk_vanpt
)
test_dict_vanpt = test_utils.test(
    model = test_model_vanpt,
    data_loader = data_loader
)

# Safe deletion
del trainer_vanpt
del trunk_vanpt, branch_vanpt, branch_test_vanpt


################################################################################
# TRAINING: Physics-Informed POD DeepONets approach
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
        epochs = 1000, 
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
loss_weights['omega_pde'] = 0.5
loss_weights['omega_bc'] = 50.0
trainer_pipoddon = Trainer(
    loss_model = Adr(
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
        epochs = 6000, 
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
# TRAINING: PTPI approach 
# (trunk training + low-cost training + fine-tuning)
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
        epochs = 1000, 
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
loss_weights['omega_pde'] = 0.5
loss_weights['omega_bc'] = 50.0
trainer_ptpi = Trainer(
    loss_model = Adr(
        branch_ptpi, trunk_ptpi, pod, loss_weights
    )
)
trainer_ptpi.compile(optimizer = optimizer_ptpi_lowcost)

# Caching computations 
def get_cache():
    cache = dict()
    x_sup = data_loader.data_model['train']['x']
    predicate = (x_sup[:,0] == 0) + (x_sup[:,0] == 1) + \
        (x_sup[:,1] == 0) + (x_sup[:,1] == 1) + \
        (x_sup[:,2] == 0) + (x_sup[:,2] == 1)
    predicate = predicate.astype('bool')
    cache['basis'] = trunk_ptpi(x_sup)
    cache['basis_bc'] = trunk_ptpi(x_sup[predicate])
    cache['x_pde'] = x_sup[~predicate]
    cache['basis_pde'] = trunk_ptpi(x_sup[~predicate])
    lapl_t, grads = get_grad(
        trunk = trunk_ptpi,
        x = x_sup[~predicate]
    )
    cache['lapl_t'] = lapl_t
    cache['dt_dx'] = grads[0]

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
        epochs = 2000, 
        callbacks = [log_callback, save_callback, load_callback],
        cache = cache_ptpi,
        display_options = ('train_loss', 'val_loss')
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
        epochs = 4000, 
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
# POSTPROCESSING: Visual comparison between the approaches
################################################################################

def postprocess_comparison(
        dir : str,
        cmap,
        mu_instances : "list[int]", 
        time_idx : int):
    """ Compares the Data-driven and Physics-Informed approaches.

    Args:
        dir (str): the save directiory
        cmap: the colormap
        mu_instances (list[int]): Parameter instances' index.
        time_idx (int): Time instance index.
    """

    assert len(mu_instances) == 3

    # Create figure
    fig, axs = plt.subplots(
        nrows = len(mu_instances), ncols = 4, figsize = (14,11)
    )

    # Loop over the instances
    for i in range(len(mu_instances)):

        # Gather the quantities to plot
        instance_idx = mu_instances[i]
        loc = data_loader.data_model['test']['x']
        params_instance = data_loader.data_model['test']['mu'][
            instance_idx*N_t_test : (instance_idx+1)*N_t_test][time_idx]
        true_instance = data_loader.data_model['test']['u_true'][:,:,0][
            instance_idx*N_t_test : (instance_idx+1)*N_t_test][time_idx]
        pred_instance_dd = test_dict_dd['u_pred'][:,:,0][
            instance_idx*N_t_test : (instance_idx+1)*N_t_test][time_idx]
        pred_instance_lowcost = test_dict_ptpi_lowcost['u_pred'][:,:,0][
            instance_idx*N_t_test : (instance_idx+1)*N_t_test][time_idx]
        pred_instance_finetuned = test_dict_ptpi_finetuned['u_pred'][:,:,0][
            instance_idx*N_t_test : (instance_idx+1)*N_t_test][time_idx]
        
        # Create the grid
        x, y, z = [loc[:,i] for i in range(loc.shape[1])]
        x_grid, y_grid = np.mgrid[0:1:1000j, 0:1:1000j]
        z_grid = 0.5

        # Interpolate on the grid
        true_grid = griddata(
            (x, y, z), true_instance, (x_grid, y_grid, z_grid)
        ).T
        pred_dd_grid= griddata(
            (x, y, z), pred_instance_dd, (x_grid, y_grid, z_grid)
        ).T
        pred_lowcost_grid = griddata(
            (x, y, z), pred_instance_lowcost, (x_grid, y_grid, z_grid)
        ).T
        pred_finetuned_grid = griddata(
            (x, y, z), pred_instance_finetuned, (x_grid, y_grid, z_grid)
        ).T

        # Plot
        im_true = axs[i,0].imshow(
            true_grid, 
            extent = (0,1,0,1), 
            origin='lower',
            cmap = cmap
        )
        im_dd = axs[i,1].imshow(
            pred_dd_grid, 
            extent = (0,1,0,1), 
            origin='lower',
            cmap = cmap,
            vmin = np.min(true_grid),
            vmax = np.max(true_grid)
        )
        im_lowcost = axs[i,2].imshow(
            pred_lowcost_grid, 
            extent = (0,1,0,1), 
            origin='lower',
            cmap = cmap,
            vmin = np.min(true_grid),
            vmax = np.max(true_grid)
        )
        im_finetuned = axs[i,3].imshow(
            pred_finetuned_grid, 
            extent = (0,1,0,1), 
            origin='lower',
            cmap = cmap,  
            vmin = np.min(true_grid),
            vmax = np.max(true_grid)
        )

        # Adjust plot
        for j in range(4):
            axs[i,j].axes.get_xaxis().set_ticks([])
            axs[i,j].axes.get_yaxis().set_ticks([])
        plt.subplots_adjust(
            left = 0.06, 
            right = 0.9, 
            hspace = 0.2, 
            wspace = 0.3, 
            bottom = 0.01,
            top = 0.94
        )

        # Add colorbar
        cbar = utils.plot_utils.im_colorbar(im_finetuned, spacing = 0.03)
        cbar.ax.tick_params(labelsize = 16)
        rounded_params = np.round(params_instance[:-1], decimals = 2)

        # Set instance title
        row_title = r'${\bf{\mu}}$' + " = (" + str(rounded_params[0]) + \
            ',' + str(rounded_params[1]) + ')'
        axs[i,0].set_ylabel(
            row_title, 
            rotation = 'vertical', 
            fontsize = 18,
            labelpad = 20
        )

    # Set paradigms' titles
    col_title = 'Exact ($t = %1.2f$)' % params_instance[-1]
    axs[0,0].set_title(col_title, fontsize = 18, pad = 25)
    axs[0,1].set_title('POD-DL-ROM\n(data-driven)', fontsize = 18, pad = 25)
    axs[0,2].set_title(
        'PTPI-DL-ROM\n(after pre-training)',fontsize = 16, pad = 25
    )
    axs[0,3].set_title('PTPI-DL-ROM\n(fine-tuned)',fontsize = 18, pad = 25)

    # Saves the figure
    plt.savefig(os.path.join(dir, 'instance_adr3d_comparison.png'))


# Call postprocessing function
plt.style.use('dark_background')
postprocess_comparison(
    dir = slides_dir,
    cmap = 'gist_rainbow',
    mu_instances = (1,7,63), 
    time_idx = 69
)
plt.style.use('default')
postprocess_comparison(
    dir = results_dir,
    cmap = 'jet',
    mu_instances = (1,7,63), 
    time_idx = 69
)

################################################################################
# POSTPROCESSING: Error analysis
################################################################################

def error_analysis(dir : str):
    """ Plots the log10 of the relative error in the parameter space.

    Args:
        dir (str): the save directiory
    """

    # Create figure
    fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (14.0, 5.0))
    images = list()

    # Get unique values of the parameters
    train_params_unique = np.unique(
        data_loader.data_model['train']['mu'][:,:-1], 
        axis = 0
    )
    test_params_unique = np.unique(
        data_loader.data_model['test']['mu'][:,:-1], 
        axis = 0
    )

    # Compute minimum and maximum values to set the rectangle dimensions
    min_train = np.min(train_params_unique, axis = 0)
    max_train = np.max(train_params_unique, axis = 0)
    min_test = np.min(test_params_unique, axis = 0)
    max_test = np.max(test_params_unique, axis = 0)
    width_train, height_train = max_train - min_train
    width_test, height_test = max_test - min_test

    # Prepare iterables for the for loop
    enum = enumerate(
        [test_dict_dd, test_dict_ptpi_lowcost, test_dict_ptpi_finetuned]
    )
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
            color = 'blue', 
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
            color = 'fuchsia', 
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
            color = 'lime', 
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
            (test_params_unique[:,0],  test_params_unique[:,1]), 
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
        axs[count].set_xlabel('$\mu_1$', fontsize = 20)
        axs[count].set_ylabel('$\mu_2$', fontsize = 20)
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
            fontsize = 14
        )
        axs[count].set_yticklabels(
            labels = yticks_err, 
            fontsize = 14
        )

        # Adjust the plot
        plt.subplots_adjust(
            left = 0.1, 
            right = 0.82, 
            hspace = 0.2, 
            wspace = 0.5, 
            top = 0.78,
            bottom = 0.15
        )
    
    # Set legend and colorbar
    plt.legend(bbox_to_anchor=(1.04, -0.15), loc='lower left', fontsize = 18)
    cbar = utils.plot_utils.im_colorbar(
        im = images[0], 
        im_cax = images[2], 
        spacing = 0.05, 
        start_bottom = 0.25
    )
    cbar.ax.tick_params(labelsize = 16)

    # Saves the figure
    plt.savefig(os.path.join(dir, 'errors_adr.png'))


# Call error analysis function
plt.style.use('dark_background')
error_analysis(dir = slides_dir)
plt.style.use('default')
error_analysis(dir = results_dir)



################################################################################
# POSTPROCESSING: Efficiency of PTPI training
################################################################################

def plot_training_efficiency(
        dir : str,
        info_colors : dict
    ):
    """ Plots the gradient descent

    Args:
        dir (str): the save directory
        info_colors (dict): the colorcode
        training_efficiency_data (dict): data about the gradient descent of 4
                                   approaches (nopt, vanpt, pipoddon, ptpi)
    """
    def readfile(architecture_name):
        filepath = checkpoint.get_log_filepath(name = architecture_name)
        csvfile = open(filepath, newline='')
        reader = csv.reader(csvfile, delimiter=',')
        lines = [rows for rows in reader]
        keys = lines[0]
        all_values = np.array(lines[1:]).astype('float32')
        info = {keys[idx] : all_values[:,idx] for idx in range(len(keys))}
        return info
    
    nopt_info = readfile(architecture_name = 'nopt')
    vanpt_pretraining_info = readfile(architecture_name = 'vanpt-pretraining')
    vanpt_finetuning_info = readfile(architecture_name = 'vanpt-finetuning')
    trunk_pipoddon_info = readfile(architecture_name = 'trunk-pipoddon')
    pipoddon_info = readfile(architecture_name = 'pipoddon')
    trunk_ptpi_info = readfile(architecture_name = 'trunk-ptpi')
    ptpi_lowcost_info = readfile(architecture_name = 'ptpi-lowcost')
    ptpi_finetuned_info = readfile(architecture_name = 'ptpi-finetuned')

    elapsed_time_pretraining_vanpt = vanpt_pretraining_info['elapsed_time'][-1]
    elapsed_time_pretraining_pipoddon = trunk_pipoddon_info['elapsed_time'][-1]
    elapsed_time_pretraining_ptpi = trunk_ptpi_info['elapsed_time'][-1] + \
        ptpi_lowcost_info['elapsed_time'][-1]
    time_nopt = nopt_info['elapsed_time']
    time_finetuning_vanpt = elapsed_time_pretraining_vanpt + \
        vanpt_finetuning_info['elapsed_time']
    time_finetuning_pipoddon = elapsed_time_pretraining_pipoddon + \
        pipoddon_info['elapsed_time']
    time_finetuning_ptpi = elapsed_time_pretraining_ptpi + \
            ptpi_finetuned_info['elapsed_time']
    
    fig, ax = plt.subplots(figsize = (15,8))

    # Creates legend handles
    handles_l1 = []
    handles_l2 = []

    # Creates legend labels
    labels_paradigms = []
    labels_endtimes = []

    # Visualization for PTPI
    if info_colors['ptpi'] is not None:
        plt_finetuned, = plt.semilogy(
            time_finetuning_ptpi, 
            ptpi_finetuned_info['test_metric'],
            color = info_colors['ptpi'],
            linewidth = 2,
        )
        plt_time_ptpi = plt.axvline(
            x = elapsed_time_pretraining_ptpi, 
            color = info_colors['ptpi'], 
            linestyle = ':', 
            linewidth = 2
        )
        plt.scatter(
            time_finetuning_ptpi[0],
            ptpi_finetuned_info['test_metric'][0],
            marker = "D",
            c = "k",
            s = 200,
            zorder = 100
        )
        handles_l1.append(plt_finetuned)
        handles_l2.append(plt_time_ptpi)
        labels_paradigms.append('PTPI-DL-ROM')
        labels_endtimes.append('PTPI-DL-ROM \npre-training endtime')
    
    # Visualization for "no pretraining"
    if info_colors['nopt'] is not None:
        plt_nopt, = plt.semilogy(
            time_nopt, 
            nopt_info['test_metric'],
            color = info_colors['nopt'],
            linewidth = 2
        )
        plt.scatter(
            time_nopt[0],
            nopt_info['test_metric'][0],
            marker = "D",
            c = "k",
            s = 200, 
            zorder = 100
        )
        handles_l1.append(plt_nopt)
        labels_paradigms.append('w/o pre-training')

    # Visualization for "vanilla pre-training"
    if info_colors['vanpt'] is not None:
        plt_vanpt, = plt.semilogy(
            time_finetuning_vanpt, 
            vanpt_finetuning_info['test_metric'],
            color = info_colors['vanpt'],
            linewidth = 2
        )
        plt_time_vanpt = plt.axvline(
            x = elapsed_time_pretraining_vanpt, 
            color = info_colors['vanpt'], 
            linestyle = ':', 
            linewidth = 2
        )
        plt.scatter(
            time_finetuning_vanpt[0],
            vanpt_finetuning_info['test_metric'][0],
            marker = "D",
            c = "k",
            s = 200,
            zorder = 100
        )
        handles_l1.append(plt_vanpt)
        handles_l2.append(plt_time_vanpt)
        labels_paradigms.append('w/ vanilla pre-training' )
        labels_endtimes.append('vanilla \npre-training endtime')
    
    # Visualization for "Physics-informed POD DeepONets"
    if info_colors['pipoddon'] is not None:
        plt_pipoddon, = plt.semilogy(
            time_finetuning_pipoddon, 
            pipoddon_info['test_metric'],
            color = info_colors['pipoddon'],
            linewidth = 2
        )
        plt_time_pipoddon = plt.axvline(
            x = elapsed_time_pretraining_pipoddon, 
            color = info_colors['pipoddon'], 
            linestyle = ':', 
            linewidth = 2
        )
        plt.scatter(
            time_finetuning_pipoddon[0],
            pipoddon_info['test_metric'][0],
            marker = "D",
            c = "k",
            s = 200,
            zorder = 100
        )
        handles_l1.append(plt_pipoddon)
        handles_l2.append(plt_time_pipoddon)
        labels_paradigms.append('PI-POD-DeepONets' )
        labels_endtimes.append(
            'PI-POD-DeepONets \n pre-training endtime'
        )

    # Labels and legends
    plt.xlabel('Elapsed time [$s$]', fontsize = 20)
    plt.ylabel('Test relative error $\mathcal{E}$', fontsize = 20)
    if len(handles_l1) != 0:
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
    if len(handles_l2) != 0:
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

    # Adjust the plot
    plt.subplots_adjust(
        left = 0.08, 
        right = 0.98, 
        hspace = 0.2, 
        wspace = 0.5, 
        top = 0.90,
        bottom = 0.26
    )
    plt.gca().axes.tick_params(labelsize = 18) 
    if len(handles_l1) != 0:
        ax.add_artist(l1)
    if len(handles_l2) != 0:
        ax.add_artist(l2)

    # Saves the plot  
    savestring = "" 
    for key in info_colors.keys():
        savestring += str(int(info_colors[key] is not None))
    plt.savefig(os.path.join(dir, 'training_efficiency' + savestring + '.png'))


# Call the function to plot training efficiency graphs with
plt.style.use('dark_background')
plot_training_efficiency(
    dir = slides_dir,
    info_colors = {
        'ptpi' : None, 'nopt' : 'magenta', 'vanpt' : None, 'pipoddon' : None
    }
)
plot_training_efficiency(
    dir = slides_dir,
    info_colors = {
        'ptpi' : None, 'nopt' : 'magenta', 'vanpt' : 'orange', 'pipoddon' : None
    }
)
plot_training_efficiency(
    dir = slides_dir,
    info_colors = {
        'ptpi' : 'cyan', 'nopt' : 'magenta','vanpt' : 'orange','pipoddon' : None
    }
)
plt.style.use('default')
plot_training_efficiency(
    dir = results_dir,
    info_colors = {
        'ptpi' : 'b', 'nopt' : 'orange', 'vanpt' : 'lime','pipoddon' : 'darkred'
    }
)


