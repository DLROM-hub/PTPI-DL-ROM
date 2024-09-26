################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Implementation of the NS BFS test case <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################

import sys
import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_FLAGS"] = \
    '--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.05'

import jax
import keras
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
from scipy.stats import qmc

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

# Define hyperparameters
pod_dim = 20
latent_dim = 5
c = 3
d = 2
p = 2
N_t_sup = 20    
N_t_test = 80

# Set up directories
parent_dir =  os.path.join('..', '..', 'data_pi', 'ns-bfs')
results_dir = os.path.join('..', '..', 'results', 'ns-bfs')
slides_dir = os.path.join('..', '..', 'results', 'ns-bfs', 'slides')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(slides_dir):
    os.makedirs(slides_dir)
checkpoint = utils.Checkpoint(save_folder = results_dir)

# Collect data filenames
filenames = dict()
filenames['mu'] = os.path.join(parent_dir, 'params_train_NS.mat')
filenames['u_true'] = os.path.join(parent_dir, 'S_train_NS.mat')
filenames['x'] = os.path.join(parent_dir, 'loc_points_train_NS.mat')
filenames_test = dict()
filenames_test['mu'] = os.path.join(parent_dir, 'params_test_NS.mat')
filenames_test['u_true'] = os.path.join(parent_dir, 'S_test_NS.mat')
filenames_test['x'] = os.path.join(parent_dir, 'loc_points_test_NS.mat')

# Train or test
train_or_test = sys.argv[1]
assert (train_or_test == 'train' or train_or_test == 'test')


################################################################################
# CONFIGURATION: Data and sampling
################################################################################
    
# Train and test data
data_loader = DataLoader(
    alpha_train_branch = 0.8
)
data_loader.read(
    filenames, 
    filenames_test, 
    N_t_sup = N_t_sup, 
    N_t_test = N_t_test,
    n_channels = c
)

# Get POD-based data
pod = utils.POD(
    pod_dim = pod_dim,
    n_channels = c,
    rpod = True
)
data_loader.process_trunk_data(pod)

# Define sampler for physics-based loss
class PhysicsSampler:

    def __init__(self, n_resample):
        self.N_s = 100
        self.n_resample = n_resample

    def __call__(self):
        data = dict()
        sampler = qmc.LatinHypercube(d = 2)
        sample = sampler.random(n = self.N_s) 
        sample = sample * np.array([300.,4.1]) + np.array([200.,0.])
        data['mu'] = sample.astype('float32')
        data['x'] = dict()
        all_x_pde = data_loader.data_model['train']['x'].astype('float32')[251:]
        idxs = np.arange(all_x_pde.shape[0])
        np.random.shuffle(idxs)
        N_pde = 500
        data['x']['x_pde'] = np.concatenate(
            (
                all_x_pde[idxs[:N_pde]], 
                np.random.rand(100,2) * np.array([1.0, 0.5])+ np.array([1.0,0]),
                np.random.rand(200,2) * np.array([1.0, 0.5]) + np.array([0.0,0.5])
            ), 
            axis = 0
        )
        x_bc = data_loader.data_model['train']['x'].astype('float32')[:251]
        predicate = x_bc[:,0] < 4 - 1e-6
        data['x']['x_bc_dir'] = x_bc[predicate]
        data['x']['x_bc_neu'] = x_bc[~predicate] 
        return data


phys_sampler = PhysicsSampler(n_resample = 10)

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
    fourier_encoding_dim = 100
    fourier_feature_layer = nns.FourierFeatureLayer(
        output_dim = fourier_encoding_dim
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
        width = 100, 
        depth = 5, 
        output_dim = latent_dim,
        activation = 'elu'
    )
    reshape_encoder = keras.layers.Reshape(target_shape = (pod_dim * c, ))
    encoder_network = nns.DenseNetwork(
        width = 50, 
        depth = 5, 
        output_dim = latent_dim,
        activation = 'elu'
    )
    encoder_block = [reshape_encoder, encoder_network]
    encoder_network = nns.ops.hstack_nns(
        input_dims = (pod_dim, c),
        blocks = encoder_block
    )
    decoder_network = nns.DenseNetwork(
        width = 100, 
        depth = 10, 
        output_dim = pod_dim * c,
        activation = 'elu'
    )
    reshape_decoder = keras.layers.Reshape(target_shape = (pod_dim, c))
    decoder_block = [decoder_network, reshape_decoder]
    decoder_network = nns.ops.hstack_nns(
        input_dims = (latent_dim, ),
        blocks = decoder_block
    )

    # Initialize the normalizer for the DLROM
    normalizer_dlrom = utils.Normalizer(
        input_train = data_loader.data_model['train']['mu'],
        target_train = pod.project(data_loader.data_model['train']['u_true'])
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
        normalizer = normalizer_dlrom
    )

    return trunk, branch, branch_test


################################################################################
# CONFIGURATION: Definition of the function to compute gradients with
################################################################################

def get_grad(trunk, x):
    n_spatial_dims = x.shape[1]
    grads = list()
    lapl_t = list()
    for ix in range(n_spatial_dims):
        get_scalar_output = lambda x, y: trunk(
            jnp.concatenate((x[:,None],y[:,None]), axis = 1)
        )[0]
        grad_op = jax.vmap(jax.jacrev(get_scalar_output, argnums = ix))
        grads +=  (grad_op(x[:,0][:,None], x[:,1][:,None])[:,:,:,0], )
        lapl_op = jax.vmap(
            jax.jacfwd(
                jax.jacrev(get_scalar_output, argnums = ix), argnums = ix
            )
        )
        lapl_t += (lapl_op(x[:,0][:,None], x[:,1][:,None])[:,:,:,0,0], )
    return lapl_t, grads


################################################################################
# CONFIGURATION: Definition of the physical model
################################################################################

class NavierStokes(utils.losses.LossFunction):

    def __init__(self, branch, trunk, pod, loss_weights):
        super(NavierStokes, self).__init__()
        self.trunk = trunk
        self.branch = branch
        self.pod = pod
        self.loss_weights = loss_weights
        time_fn = lambda t: (1 - jnp.exp(-3 * t))
        self.bc_fn = lambda params, x: 2 * (x[:,1] - 0.5) * (1.0 - x[:,1]) * \
            (x[:,0] < 1e-6) * time_fn(params[1])
        self.bc_fn = jax.vmap(self.bc_fn, in_axes = (0,None))

        
    def call(self, data, cache):

        # Extract data
        mu_phys = data.get('mu_phys')
        x_phys = data.get('x_phys')
        x_sup = data.get('x_sup')
        mu_sup = data.get('mu_sup')
        u_sup_true = data.get('u_true_sup')

        # Generate auxiliary data
        if cache == None:
            x_pde = x_phys['x_pde']
            x_bc_dir = x_phys['x_bc_dir']
            x_bc_neu = x_phys['x_bc_neu']
            t_sup = self.trunk(x_sup)
            t_bc_dir = self.trunk(x_bc_dir)
            t_bc_neu = self.trunk(x_bc_neu)
            t_pde = self.trunk(x_pde)
            lapl_t, grads = get_grad(trunk = self.trunk, x = x_pde)
            _, grads_neu = get_grad(trunk = self.trunk, x = x_bc_neu)
            dt_dx, dt_dy = grads
            dt_dxx, dt_dyy = lapl_t
            dt_dx_neu = grads_neu[0]
        else:
            x_bc_dir = cache.get('x_bc_dir')
            x_bc_neu = cache.get('x_bc_neu')
            t_sup = cache.get('t_sup')
            t_bc_dir = cache.get('t_bc_dir')
            t_bc_neu = cache.get('t_bc_neu')
            x_pde = cache.get('x_pde')
            dt_dx = cache.get('dt_dx')
            dt_dy = cache.get('dt_dy')
            dt_dxx = cache.get('dt_dxx')
            dt_dyy = cache.get('dt_dyy')
            t_pde = cache.get('t_pde')
            dt_dx_neu = cache.get('dt_dx_neu')

        # Supervised loss
        u_sup_true_proj = self.pod.project(u_sup_true)
        b_sup, reduced_network_repr, latent_repr = self.branch(
            (mu_sup, u_sup_true_proj)
        )
        res_latent = (reduced_network_repr - latent_repr)**2
        l_latent = self.loss_weights['omega_latent'] * \
            keras.ops.mean(res_latent)
        u_sup = keras.ops.einsum('bjc,ijc->bic', b_sup, t_sup)
        res_sup = keras.ops.sum((u_sup - u_sup_true)**2, axis = 1)
        l_sup = keras.ops.mean(res_sup, axis = 0)
        l_sup = self.loss_weights['omega_sup_u'] * l_sup[0] + \
            self.loss_weights['omega_sup_v'] * l_sup[1] + \
            self.loss_weights['omega_sup_p'] * l_sup[2]
            
        # Eventually compute physics-based loss
        physics_loss_weights_sum = self.loss_weights['omega_pde_u'] + \
            self.loss_weights['omega_pde_v'] + \
            self.loss_weights['omega_pde_p'] + \
            self.loss_weights['omega_bc_u_dir'] + \
            self.loss_weights['omega_bc_v_dir'] + \
            self.loss_weights['omega_bc_u_neu'] + \
            self.loss_weights['omega_bc_v_neu'] 
        
        if physics_loss_weights_sum > 0:
            # Compute physics-based loss wrt internal dofs
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
            U_prime = keras.ops.einsum('bjc,ijc->bic', b_prime, t_pde)
            U = keras.ops.einsum('bjc,ijc->bic', b_phys, t_pde)
            dU_dx = keras.ops.einsum('bjc,ijc->bic', b_phys, dt_dx)
            dU_dy = keras.ops.einsum('bjc,ijc->bic', b_phys, dt_dy)
            dU_dxx = keras.ops.einsum('bjc,ijc->bic', b_phys, dt_dxx)
            dU_dyy = keras.ops.einsum('bjc,ijc->bic', b_phys, dt_dyy)
            u_prime = U_prime[:,:,0]
            v_prime = U_prime[:,:,1]
            u = U[:,:,0]
            v = U[:,:,1]
            du_dx = dU_dx[:,:,0]
            dv_dx = dU_dx[:,:,1]
            dp_dx = dU_dx[:,:,2]
            du_dy = dU_dy[:,:,0]
            dv_dy = dU_dy[:,:,1]
            dp_dy = dU_dy[:,:,2]
            du_dxx = dU_dxx[:,:,0]
            dv_dxx = dU_dxx[:,:,1]
            du_dyy = dU_dyy[:,:,0]
            dv_dyy = dU_dyy[:,:,1]
            Re = mu_phys[:,0][:,None]
            res_pde_u =  u_prime + u * du_dx + v * du_dy + dp_dx - \
                1 / Re * (du_dxx + du_dyy)
            res_pde_v = v_prime + u * dv_dx + v * dv_dy + dp_dy - \
                1 / Re * (dv_dxx + dv_dyy)
            res_pde_p = du_dx + dv_dy
            res_pde = self.loss_weights['omega_pde_u'] * res_pde_u**2 + \
                self.loss_weights['omega_pde_v'] * res_pde_v**2 + \
                self.loss_weights['omega_pde_p'] * res_pde_p**2
            l_pde = keras.ops.mean(res_pde)

            # Compute physics-based loss wrt bcs
            U_bc_dir = keras.ops.einsum('bjc,ijc->bic', b_phys, t_bc_dir)
            U_bc_neu = keras.ops.einsum('bjc,ijc->bic', b_phys, t_bc_neu)
            dU_dx_neu = keras.ops.einsum('bjc,ijc->bic', b_phys, dt_dx_neu)
            u_bc_dir = U_bc_dir[:,:,0]
            v_bc_dir = U_bc_dir[:,:,1]
            p_bc_neu = U_bc_neu[:,:,2]
            du_dx_neu = dU_dx_neu[:,:,0]
            u_true_dir = self.bc_fn(mu_phys, x_bc_dir)
            res_dir = self.loss_weights['omega_bc_u_dir'] * \
                (u_bc_dir - u_true_dir)**2 + \
                self.loss_weights['omega_bc_v_dir'] * v_bc_dir**2
            res_neu = self.loss_weights['omega_bc_u_neu'] * \
                (du_dx_neu - p_bc_neu)**2 + \
                self.loss_weights['omega_bc_v_neu'] * p_bc_neu**2
            l_bc = keras.ops.mean(res_dir) + keras.ops.mean(res_neu)

        loss = l_sup 
        
        if physics_loss_weights_sum > 0:
            loss += l_pde + l_bc

        metric = utils.metrics.mean_rel_err_metric(u_sup_true, u_sup)

        return {'loss' : loss, 'metric' : metric}


################################################################################
# CONFIGURATION: Optimization and testing
################################################################################

# Set optimizer
optimizer_dd = keras.optimizers.Adam(learning_rate = 3e-4)
optimizer_ptpi_trunk = keras.optimizers.Adam(learning_rate = 3e-4)
optimizer_ptpi_lowcost = keras.optimizers.Adam(learning_rate = 3e-4)
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
loss_weights = dict()
loss_weights['omega_sup_u'] = 0.5 
loss_weights['omega_sup_v'] = 0.5
loss_weights['omega_sup_p'] = 0.5
loss_weights['omega_pde_u'] = 0
loss_weights['omega_pde_v'] = 0 
loss_weights['omega_pde_p'] = 0 
loss_weights['omega_bc_u_dir'] = 0
loss_weights['omega_bc_v_dir'] = 0
loss_weights['omega_bc_u_neu'] = 0 
loss_weights['omega_bc_v_neu'] = 0
loss_weights['omega_latent'] = 0.5
trainer_dd = Trainer(
    loss_model = NavierStokes(branch_dd, trunk_dd, pod, loss_weights)
)
trainer_dd.compile(optimizer = optimizer_dd)

# Cache the POD basis
cache_dd = dict()
cache_dd['t_sup'] = pod.subspaces

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
        epochs = 5000, 
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
# TRAINING: Pre-trained Physics-Informed approach 
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
        epochs = 3000, 
        callbacks = [log_callback, save_callback, load_callback],
        display_options = ('train_loss', 'val_loss')
    )
else:
    print('Using saved weights')
    trainer_trunk_ptpi.load(weights_filepath)
test_utils.test_trunk(trunk_ptpi, data_loader)

# BRANCH NETWORK PRE-TRAINING ##############################################

# Declare trunk not trainable
trainer_trunk_ptpi.set_trainability(trainability = False)

# Model configuration
loss_weights = dict()
loss_weights['omega_sup_u'] = 0.5 
loss_weights['omega_sup_v'] = 0.5
loss_weights['omega_sup_p'] = 0.5
loss_weights['omega_pde_u'] = 10
loss_weights['omega_pde_v'] = 10
loss_weights['omega_pde_p'] = 10 
loss_weights['omega_bc_u_dir'] = 100
loss_weights['omega_bc_v_dir'] = 1000
loss_weights['omega_bc_u_neu'] = 1 
loss_weights['omega_bc_v_neu'] = 1
loss_weights['omega_latent'] = 0.5
trainer_ptpi = Trainer(
    loss_model = NavierStokes(branch_ptpi, trunk_ptpi, pod, loss_weights)
)
trainer_ptpi.compile(optimizer = optimizer_ptpi_lowcost)


# Caching computations and reinstantiate model with cache
def get_cache():
    cache = dict()
    x_sup = data_loader.data_model['train']['x']
    x_bc = x_sup[:251]
    predicate = x_bc[:,0] < 4 - 1e-6
    x_pde = x_sup[251:]
    cache['x_bc_dir'] = x_bc[predicate]
    cache['x_bc_neu'] = x_bc[~predicate]
    cache['t_sup'] = trunk_ptpi(x_sup)
    cache['t_bc_dir'] = trunk_ptpi(x_bc[predicate])
    cache['t_bc_neu'] = trunk_ptpi(x_bc[~predicate])
    cache['x_pde'] = x_pde
    cache['t_pde'] = trunk_ptpi(x_pde)
    lapl_t, grads = get_grad(
        trunk = trunk_ptpi,
        x = x_pde
    )
    _, grads_neu = get_grad(
        trunk = trunk_ptpi,
        x = cache['x_bc_neu']
    )
    cache['dt_dx'] = grads[0]
    cache['dt_dy'] = grads[1]
    cache['dt_dxx'] = lapl_t[0]
    cache['dt_dyy'] = lapl_t[1]
    cache['dt_dx_neu'] = grads_neu[0]
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
        epochs = 5000, 
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
        epochs = 5000, 
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
# POSTPROCESSING: Plot time-wise instances
################################################################################

def postprocessing(
        dir : str, 
        time_idxs : "list[int]", 
        instance_idx : int, 
        cmap):
    """ Postprocessing comparison between physics-informed and data-driven.

    Args:
        dir (str): the save directory
        time_idxs (list[idxs]): The time indices.
        instance_idx (int): The parameter instance index.
        cmap: the colormap
    """

    assert len(time_idxs) == 3

    # Creating strings for identifying channels (and velocity magnitude)
    which_variable = ('vel_magn', 'pres', 'vel_x', 'vel_y')

    # Iterating over the created list of strings
    for variable in which_variable:

        # Generating a figure
        fig, axs = plt.subplots(
            nrows = len(time_idxs), ncols = 3, figsize = (17,7)
        )

        # Looping over the time indexes
        for i in range(len(time_idxs)):

            # Gather the quantities to plot
            loc = data_loader.data_model['test']['x']
            time_idx = time_idxs[i]
            params_instance = data_loader.data_model['test']['mu'][
                instance_idx*N_t_test : (instance_idx+1)*N_t_test][time_idx]
            true_instance = data_loader.data_model['test']['u_true'][
                instance_idx*N_t_test : (instance_idx+1)*N_t_test][time_idx]
            pred_dd = test_dict_dd['u_pred'][
                instance_idx*N_t_test : (instance_idx+1)*N_t_test][time_idx]
            pred_finetuned = test_dict_ptpi_finetuned['u_pred'][
                instance_idx*N_t_test : (instance_idx+1)*N_t_test][time_idx]
            
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
            
            # Selecting current variable of interest
            if variable == 'vel_magn':
                true_instance = get_magnitude(true_instance)
                pred_dd = get_magnitude(pred_dd)
                pred_finetuned = get_magnitude(pred_finetuned)
            elif variable == 'vel_x':
                true_instance = true_instance[:,0]
                pred_dd = pred_dd[:,0]
                pred_finetuned = pred_finetuned[:,0]
            elif variable == 'vel_y':
                true_instance = true_instance[:,1]
                pred_dd = pred_dd[:,1]
                pred_finetuned = pred_finetuned[:,1]
            else:
                true_instance = true_instance[:,2]
                pred_dd = pred_dd[:,2]
                pred_finetuned = pred_finetuned[:,2]

            # Creating a grid and interpolating
            x, y = [loc[:,i] for i in range(loc.shape[1])]
            x_grid, y_grid = np.mgrid[0:4:1000j, 0:1:3000j]
            true_grid = griddata(
                (x, y), true_instance, (x_grid, y_grid), method = 'linear'
            ).T
            pred_dd_grid = griddata(
                (x, y), pred_dd, (x_grid, y_grid), method = 'linear'
            ).T
            pred_finetuned_grid = griddata(
                (x, y), pred_finetuned, (x_grid, y_grid), method = 'linear'
            ).T
            grids = (
                true_grid, pred_dd_grid, pred_finetuned_grid
            )
            for item in grids:
                rectangle_hor = (x_grid < 1) * (y_grid < 0.5)
                is_nan = rectangle_hor.T 
                item[is_nan] = np.nan
            
            # Visualize the plots
            lims = dict(
                extent = (0,4,0,1), 
                origin = 'lower', 
                cmap = cmap,
                vmin = np.min(true_instance),
                vmax = np.max(true_instance)
            )
            im_true = axs[i,0].imshow(true_grid, **lims)
            im_dd = axs[i,1].imshow(pred_dd_grid, **lims)
            im_finetuned = axs[i,2].imshow(pred_finetuned_grid, **lims)

            # Adjust the plot and the colorbar
            plt.subplots_adjust(
                left = 0.05, right = 0.92, hspace = 0.4, wspace = 0.1
            )
            cbar = utils.plot_utils.im_colorbar(im_finetuned)
            cbar.ax.tick_params(labelsize = 17)

            # Work out ticks
            for j in range(3):
                axs[i,j].spines['top'].set_visible(False)
                axs[i,j].spines['right'].set_visible(False)
                axs[i,j].spines['bottom'].set_visible(False)
                axs[i,j].spines['left'].set_visible(False)
                axs[i,j].axes.get_xaxis().set_ticks([])
                axs[i,j].axes.get_yaxis().set_ticks([])
            
            # Set row title
            row_title = 't = %1.2f' % params_instance[-1]
            axs[i,0].set_ylabel(
                row_title, 
                rotation = 'vertical', 
                fontsize = 23, 
                labelpad = 20
            )

        # Set column title
        rounded_param = np.round(params_instance[0], decimals = 2)
        col_title = r'Exact ($Re$' + " = " + str(rounded_param) + ')'
        axs[0,0].set_title(col_title, fontsize = 23, pad = 25)
        axs[0,1].set_title(
            'POD-DL-ROM\n(data-driven)', fontsize = 23, pad = 25
        )
        axs[0,2].set_title(
            'PTPI-DL-ROM\n(fine-tuned)', fontsize = 23, pad = 25
        )

        # Saving the figure
        plt.savefig(
            os.path.join(
                dir, variable + '_instance_ns_elbow_comparison.png'
            )
        )

# Call the post processing function
plt.style.use('dark_background')
postprocessing(
    dir = slides_dir, 
    time_idxs = (11,40,70), 
    instance_idx = 2, 
    cmap = 'gist_rainbow'
)
plt.style.use('default')
postprocessing(
    results_dir, 
    time_idxs = (11,40,70),
    instance_idx = 2, 
    cmap = 'jet'
)

################################################################################
# POSTPROCESSING: Plot time-wise errors (Data-driven vs ptpi)
################################################################################

def error_analysis():
    """ Compares the time-wise error between Data-driven POD-DL-ROM
        and PTPI-DL-ROM paradigms.
    """

    # Getting unique time instances
    train_time_unique = np.unique(data_loader.data_model['train']['mu'][:,-1])
    test_time_unique = np.unique(data_loader.data_model['test']['mu'][:,-1])

    # Getting time-wise errors
    err_time_dd = np.mean(test_dict_dd['err_time'], axis = 1)
    err_time_lowcost = np.mean(test_dict_ptpi_lowcost['err_time'], axis = 1)
    err_time_finetuned = np.mean(test_dict_ptpi_finetuned['err_time'], axis = 1)

    # Creating the figure
    plt.figure(figsize = (15,7))

    # Plotting the time-wise errors
    plt.semilogy(
        test_time_unique, 
        err_time_dd, 
        label = 'POD-DL-ROM\n(data-driven)',
        color = 'b',
        linewidth = 3
    )
    plt.semilogy(
        test_time_unique, 
        err_time_lowcost,
        label = 'PTPI-DL-ROM\n(after pre-training)',
        color = 'r',
        linewidth = 3
    )
    plt.semilogy(
        test_time_unique, 
        err_time_finetuned,
        label = 'PTPI-DL-ROM\n(fine-tuned)',
        color = 'g', 
        linewidth = 3
    )

    # Displaying the supervised train time domain
    plt.axvspan(
        np.min(train_time_unique), 
        np.max(train_time_unique), 
        color ='r', 
        alpha = 0.1, 
        label = '$\mathcal{T}_{sup}$'
    )

    # Work out labels, legend and layout
    plt.ylabel('$E(t,N_{data})$', fontsize = 23)
    plt.xlabel('$t$', fontsize = 23)
    plt.legend(
        bbox_to_anchor = (0, 1.02, 1, 0.2), 
        loc =   "lower left",
        mode = "expand", 
        borderaxespad = 0, 
        ncol = 4, 
        fontsize = 23
    )
    plt.gca().axes.tick_params(labelsize = 18) 
    plt.tight_layout()

    # Save the figure
    plt.savefig(
        os.path.join(
            results_dir, 'error_analysis_ns.png'
        )
    )

# Call error analysis function
error_analysis()

################################################################################
# POSTPROCESSING: Movies for slides
################################################################################

def make_movie_error(idx, fps = 5, bitrate = 30):
    # Getting unique time instances
    train_time_unique = np.unique(data_loader.data_model['train']['mu'][:,-1])
    test_time_unique = np.unique(data_loader.data_model['test']['mu'][:,-1])

    # Getting time-wise errors
    def get_err(test_dict, idx):
        u_pred = test_dict['u_pred'][idx*N_t_test : (idx+1)*N_t_test]
        u_true = test_dict['u_true'][idx*N_t_test : (idx+1)*N_t_test]
        num = np.mean(np.linalg.norm(u_pred - u_true, axis = 1), axis = 1)
        den = np.mean(np.linalg.norm(u_true, axis = 1), axis = 1)
        return num / den

    err_time_dd = get_err(test_dict_dd, idx)
    err_time_finetuned = get_err(test_dict_ptpi_finetuned, idx)

    # Constructing figure
    fig = plt.figure(figsize = (12, 5))
    axs = fig.add_subplot()

    def update(frame):
        plt.cla()
        # Plotting the time-wise errors
        axs.semilogy(
            test_time_unique[:frame], 
            err_time_dd[:frame], 
            label = 'POD-DL-ROM\n(data-driven)',
            color = 'cyan',
            linewidth = 3
        )
        axs.semilogy(
            test_time_unique[:frame], 
            err_time_finetuned[:frame],
            label = 'PTPI-DL-ROM\n(fine-tuned)',
            color = 'magenta', 
            linewidth = 3
        )

        # Displaying the supervised train time domain
        axs.axvspan(
            np.min(train_time_unique), 
            np.max(train_time_unique), 
            color ='w', 
            alpha = 0.2, 
            label = '$\mathcal{T}_{sup}$'
        )

        # Work out legend 
        axs.legend(
            bbox_to_anchor = (0, 1.02, 1, 0.2), 
            loc =   "lower left",
            mode = "expand", 
            borderaxespad = 0, 
            ncol = 4, 
            fontsize = 23
        )
        plt.gca().axes.tick_params(labelsize = 18) 
        plt.xlim([np.min(test_time_unique), np.max(test_time_unique)])
        plt.ylim([np.min(err_time_finetuned), np.max(err_time_dd)])
        axs.axvline(test_time_unique[frame])
        axs.scatter(
            [test_time_unique[frame], test_time_unique[frame]],
            [err_time_dd[frame], err_time_finetuned[frame]],
            c = 'white',
            s = 100
        )

        # Work out labels and layout
        axs.set_ylabel('$E(t)$', fontsize = 23)
        axs.set_xlabel('$t$', fontsize = 23)
        plt.subplots_adjust(
            left = 0.15, 
            right = 0.9, 
            hspace = 0.4, 
            wspace = 0.1, 
            top = 0.8, 
            bottom = 0.2
        )

        
    anim = FuncAnimation(fig=fig, func=update, frames=N_t_test)
    anim.save(
        filename = os.path.join(
            slides_dir, 'error_ns_sample' + str(idx)+ '.gif'), 
        writer='pillow', 
        fps = fps, 
        bitrate = bitrate
    )



def make_movie_instance(idx, fps = 5, bitrate = 30):

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

    # Get instances on a grid
    def get_variables(frame):

        # Get instances
        loc = data_loader.data_model['test']['x']
        params_instance = data_loader.data_model['test']['mu'][
            idx*N_t_test : (idx+1)*N_t_test][frame]
        true_instance = data_loader.data_model['test']['u_true'][
            idx*N_t_test : (idx+1)*N_t_test][frame]
        pred_dd = test_dict_dd['u_pred'][
            idx*N_t_test : (idx+1)*N_t_test][frame]
        pred_finetuned = test_dict_ptpi_finetuned['u_pred'][
            idx*N_t_test : (idx+1)*N_t_test][frame]
        
        # Compute variables
        true_instance = [get_magnitude(true_instance), true_instance[:,2]]
        pred_dd = [get_magnitude(pred_dd), pred_dd[:,2]]
        pred_finetuned = [get_magnitude(pred_finetuned), pred_finetuned[:,2]]

        return loc, params_instance, true_instance, pred_dd, pred_finetuned


    # Constructing figure
    gs = gridspec.GridSpec(2, 3, width_ratios = [0.333, 0.333, 0.334,])
    fig = plt.figure(figsize = (13, 5))
    axs = list()
    cbar_axs = list()
    for i in range(2):
        for j in range(3):
            axs.append(fig.add_subplot(gs[3*i+j]))

    # get colorbars
    targets = np.array([get_variables(frame)[2] for frame in range(N_t_test)])
    targets_max = np.max(targets, axis = (0,2))
    targets_min = np.min(targets, axis = (0,2))
    targets_max[1] = targets_max[1] / 18
    x_grid, y_grid = np.mgrid[0:4:100j, 0:1:100j]

    def update(frame):
        print(frame)
        # Get variables
        loc, params_instance, true_instance, pred_dd, pred_finetuned = \
            get_variables(frame)

        # Creating a grid 
        x, y = [loc[:,i] for i in range(loc.shape[1])]

        # Interpolating on  a grid
        for i in range(len(true_instance)):
            true_grid = griddata(
                (x, y), true_instance[i], (x_grid, y_grid), method = 'linear'
            ).T
            pred_dd_grid = griddata(
                (x, y), pred_dd[i], (x_grid, y_grid), method = 'linear'
            ).T
            pred_finetuned_grid = griddata(
                (x, y), pred_finetuned[i], (x_grid, y_grid), method = 'linear'
            ).T
            grids = (
                true_grid, pred_dd_grid, pred_finetuned_grid
            )
            for item in grids:
                rectangle_hor = (x_grid < 1) * (y_grid < 0.5)
                is_nan = rectangle_hor.T 
                item[is_nan] = np.nan

             # Visualize the plots
            lims = dict(
                extent = (0,4,0,1), 
                origin = 'lower', 
                cmap = 'gist_rainbow',
                vmin = targets_min[i],
                vmax = targets_max[i]
            )
            im_true = axs[3*i].imshow(true_grid, **lims)
            im_dd = axs[3*i+1].imshow(pred_dd_grid, **lims)
            im_finetuned = axs[3*i+2].imshow(pred_finetuned_grid, **lims)

            # Adjust the plot and the colorbar
            plt.subplots_adjust(
                left = 0.05, right = 0.92, hspace = 0.4, wspace = 0.1
            )
            if frame == 0:
                cbar = utils.plot_utils.im_colorbar(im_true, im_finetuned)
                cbar.ax.tick_params(labelsize = 17)

            # Work out ticks
            for j in range(3):
                axs[3*i+j].spines['top'].set_visible(False)
                axs[3*i+j].spines['right'].set_visible(False)
                axs[3*i+j].spines['bottom'].set_visible(False)
                axs[3*i+j].spines['left'].set_visible(False)
                axs[3*i+j].axes.get_xaxis().set_ticks([])
                axs[3*i+j].axes.get_yaxis().set_ticks([])

        # Set column title
        rounded_param = np.round(params_instance[0], decimals = 2)
        col_title = r'Exact ($Re$' + " = " + str(rounded_param) + ')'
        axs[0].set_title(col_title, fontsize = 23, pad = 25)
        axs[1].set_title(
            'POD-DL-ROM\n(data-driven)', fontsize = 23, pad = 25
        )
        axs[2].set_title(
            'PTPI-DL-ROM\n(fine-tuned)', fontsize = 23, pad = 25
        )

        return axs

    anim = FuncAnimation(fig=fig, func=update, frames=N_t_test)
    anim.save(
        filename = os.path.join(slides_dir, 'ns_sample' + str(idx)+ '.gif'), 
        writer='pillow', 
        fps = fps, 
        bitrate = bitrate
    )


# Call the make_movie functions
plt.style.use('dark_background')
make_movie_error(idx = 2, fps = 5, bitrate = 30)
make_movie_instance(idx = 2, fps = 5, bitrate = 30)
