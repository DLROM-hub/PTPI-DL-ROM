################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Implementation of the Eikonal test case <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################

import sys
import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_FLAGS"] = \
    '--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'

import jax
import keras
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
pod_dim = 2
latent_dim = 2
c = 1
d = 2
p = 1
N_t_sup = 1
N_t_test = 1

# Set up directories
parent_dir =  os.path.join('..', '..', 'data_pi', 'eikonal')
results_dir = os.path.join('..', '..', 'results', 'eikonal')
slides_dir = os.path.join('..', '..', 'results', 'eikonal', 'slides')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(slides_dir):
    os.makedirs(slides_dir)
checkpoint = utils.Checkpoint(save_folder = results_dir)

# Collect data filenames
filenames = dict()
filenames['mu'] = os.path.join(parent_dir, 'params_train_eikonal.mat')
filenames['u_true'] = os.path.join(parent_dir, 'S_U_train_eikonal.mat')
filenames['x'] = os.path.join(parent_dir, 'loc_points.mat')
filenames_test = dict()
filenames_test['mu'] = os.path.join(parent_dir, 'params_test_eikonal.mat')
filenames_test['u_true'] = os.path.join(parent_dir, 'S_U_test_eikonal.mat')
filenames_test['x'] = os.path.join(parent_dir, 'loc_points.mat')

# Train or test
train_or_test = sys.argv[1]
assert (train_or_test == 'train' or train_or_test == 'test')


################################################################################
# CONFIGURATION: Function to construct the neural network architectures
################################################################################

def get_architectures():
    # Define trunk components
    dense_trunk = nns.DenseNetwork(
        width = 50, 
        depth = 5, 
        output_dim = pod_dim * c,
        activation = 'elu',
        kernel_initializer = 'he_uniform'
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
        grads +=  (
            grad_op(x[:,0][:,None], x[:,1][:,None])[:,:,0], 
        )
    return grads


################################################################################
# CONFIGURATION: Definition of the physical model
################################################################################

class Eikonal(utils.losses.LossFunction):

    def __init__(self, branch, trunk, pod, loss_weights):
        super(Eikonal, self).__init__()
        self.trunk = trunk
        self.branch = branch
        self.theta = np.linspace(0, 2 * np.pi, 100)
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
            x_pde = x_phys.get('x_pde')
            t_sup = self.trunk(x_sup)
            grads = get_grad(trunk = self.trunk, x = x_pde)
            dt_dx, dt_dy = grads
        else:
            t_sup = cache.get('basis')
            x_pde = cache.get('x_pde')
            dt_dx = cache.get('dt_dx')
            dt_dy = cache.get('dt_dy')

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
            dummy = nns.ops.get_dlrom_dummy(
                branch = self.branch, 
                mu_phys = mu_phys
            )
            u_sup_true_proj = self.pod.project(u_sup_true)
            b_phys, _, _ = self.branch((mu_phys, dummy))
            x_bc = jnp.array(
                [mu_phys * jnp.cos(self.theta), 
                mu_phys * jnp.sin(self.theta)]
            )
            x_bc = x_bc.transpose((1,2,0))
            t_bc = jnp.array(
                [self.trunk(x_bc[i]) for i in range(mu_phys.shape[0])]
            )
            du_dx = keras.ops.einsum('bjc,ijc->bic', b_phys, dt_dx)
            du_dy = keras.ops.einsum('bjc,ijc->bic', b_phys, dt_dy)
            u_bc = keras.ops.einsum('bjc,bijc->bic', b_phys, t_bc)
            res_pde = ((du_dx**2 + du_dy**2) - 1.0)**2
            l_pde = keras.ops.mean(res_pde)
            l_bc = keras.ops.mean(u_bc**2)
            loss += self.loss_weights['omega_pde'] * l_pde + \
                self.loss_weights['omega_bc'] * l_bc
        
        # Eventually adding latent loss
        if self.loss_weights['omega_latent'] > 0:
            loss += self.loss_weights['omega_latent'] * l_latent 
        
        metric = utils.metrics.mean_rel_err_metric(u_sup_true, u_sup)

        return {'loss' : loss, 'metric' : metric}


################################################################################
# ERROR ANALYSIS: Comparison between the proposed Physics-Informed approach and 
# a Data-Driven approach in a small data framework though an ablation study
################################################################################

# Indicating which data we load in each iteration of the training set
instances_selections = [
    [0,1,2,39,40],
]
for i in range(3):
    remaining_samples = set(range(41)) - set(instances_selections[i])
    remaining_samples = np.array(list(remaining_samples))
    idxs = np.arange(len(remaining_samples))
    np.random.shuffle(idxs)
    selected_samples = list(
        remaining_samples[idxs[:int(5 * (2**(i+1) - 2**(i)))]]
    ) + instances_selections[i]
    instances_selections.append(selected_samples)


# Initialize objects for the error analysis
params_selection = list()
errors_dd = list()
errors_ptpi = list()

# Main loop
for i in range(len(instances_selections)):

    # --------------------------------------------------------------------------
    # CONFIGURATION: DATA AND SAMPLING
    # --------------------------------------------------------------------------
   
    # Train and test data
    data_loader = DataLoader(
        alpha_train_branch = 0.8
    )
    data_loader.read(
        filenames, 
        filenames_test, 
        N_t_sup = N_t_sup, 
        N_t_test = N_t_test,
        which_instances = instances_selections[i]
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
            self.N_s = 1000
            self.mu_1_min = 0.1
            self.mu_1_max = 1.1
            self.n_resample = n_resample

        def __call__(self):
            data = dict()
            data['mu'] = self.mu_1_min + (self.mu_1_max - self.mu_1_min) * \
                np.random.rand(self.N_s,1)
            data['x'] = dict()
            data['x']['x_pde'] = 2 * np.random.rand(1000,2) - 1
            return data

    phys_sampler = PhysicsSampler(n_resample = 5)

    # Define batch dimension
    batch_size_sup = 1
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

    # Append current training and validation parameters to allow for final 
    # visualization
    train_val_params_unique = np.unique(
        np.vstack(
            (
                data_loader.data_model['train']['mu'], 
                data_loader.data_model['val']['mu']
            )
        )
    )   
    params_selection.append(train_val_params_unique)


    #---------------------------------------------------------------------------
    # CONFIGURATION: optimization 
    #---------------------------------------------------------------------------

    # Set optimizer
    optimizer_dd = keras.optimizers.Adam(learning_rate = 1e-3)
    optimizer_ptpi_trunk = keras.optimizers.Adam(learning_rate = 1e-3)
    optimizer_ptpi_lowcost = keras.optimizers.Adam(learning_rate = 3e-4)
    optimizer_ptpi_finetuning = keras.optimizers.Adam(learning_rate = 1e-4)


    #---------------------------------------------------------------------------
    # TRAINING: Data-driven approach
    #---------------------------------------------------------------------------

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
        loss_model = Eikonal(branch_dd, trunk_dd, pod, loss_weights)
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
            epochs = 600, 
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
    
    # Saving errors to allow for visualization
    errors_dd.append(test_dict_dd['err_param'])

    # Safe deletion
    del trainer_dd, trunk_dd, branch_dd, branch_test_dd

    #---------------------------------------------------------------------------
    # TRAINING: Pre-Trained Physics-Informed approach 
    # (trunk training + low-cost training + fine-tuning)
    #---------------------------------------------------------------------------

    print('\n--------------------')
    print('PTPI-DL-ROM approach')
    print('--------------------')

    # Construct the architectures
    trunk_ptpi, branch_ptpi, branch_test_ptpi = get_architectures()

    # TRUNK NETWORK PRE-TRAINING ###############################################

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
    loss_weights['omega_sup'] = 0.5
    loss_weights['omega_latent'] = 0.5
    loss_weights['omega_pde'] = 0.5
    loss_weights['omega_bc'] = 100.0
    trainer_ptpi = Trainer(
        loss_model = Eikonal(
            branch_ptpi, trunk_ptpi, pod, loss_weights
        )
    )
    trainer_ptpi.compile(optimizer = optimizer_ptpi_lowcost)

    # Caching computations 
    def get_cache():
        cache = dict()
        x_sup = data_loader.data_model['train']['x']
        cache['x_pde'] = x_sup
        cache['basis'] = trunk_ptpi(x_sup)
        grads = get_grad(
            trunk = trunk_ptpi,
            x = x_sup
        )
        cache['dt_dx'] = grads[0]
        cache['dt_dy'] = grads[1]

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
            epochs = 1000, 
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

    # FINE-TUNING ##############################################################

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
            epochs = 500, 
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

    # Saving errors to allow for visualization
    errors_ptpi.append(test_dict_ptpi_finetuned['err_param'])

    #---------------------------------------------------------------------------
    # POST-PROCESSING: Comparison between DD and PI simulations 
    #---------------------------------------------------------------------------

    # Preparing lists, gathering info
    paradigms = ('dd', 'lowcost', 'finetuned')
    titles = [
        'POD-DL-ROM\n(data-driven)', 
        'PTPI-DL-ROM\n(after pre-training)',
        'PTPI-DL-ROM\n(fine-tuned)' 
    ]
    predictions = (
        test_dict_dd['u_pred'],
        test_dict_ptpi_lowcost['u_pred'],
        test_dict_ptpi_finetuned['u_pred'], 
    )

    # Selecting the instance to visualize
    instance_idx = -1

    def visualize_instances(dir, cmap):
        fig, axs = plt.subplots(nrows = 3, ncols = 3, figsize = (14,12.5))
        params_instance = data_loader.data_model['test']['mu'][instance_idx]
        true_instance = data_loader.data_model['test']['u_true'][instance_idx]
        N_h = int(np.sqrt(true_instance.shape[0]))
        options = dict(
            origin = 'lower', 
            extent = (0,1,0,1),
            interpolation = 'bilinear',
            cmap = cmap
        )
        im_true = axs[0,0].imshow(
            true_instance.reshape((N_h, N_h)), 
            **options
        )
        axs[0,0].set_title(
            'Exact ($\mu = %1.2f$)' % params_instance[0],
            fontsize = 18,
            pad = 17
        )
        plt.subplots_adjust(
            left = 0.03, 
            right = 0.88, 
            hspace = 0.2, 
            wspace = 0.7, 
            top = 0.93,
            bottom = 0.02
        )
        cbar_true = utils.plot_utils.im_colorbar(im_true, spacing = 0.03)
        cbar_true.ax.tick_params(labelsize = 16)
        enum = enumerate(zip(paradigms, titles, predictions))
        for count, (model_name, title, pred_test) in enum:
            pred_instance = pred_test[instance_idx]
            err_instance = np.abs(true_instance - pred_instance)
            N_h = int(np.sqrt(true_instance.shape[0]))
            im_pred = axs[count,1].imshow(
                pred_instance.reshape((N_h, N_h)), 
                **options,
                vmin = np.min(true_instance),
                vmax = np.max(true_instance)
            )
            im_err = axs[count,2].imshow(
                err_instance.reshape((N_h, N_h)), 
                **options
            )
            cbar_pred = utils.plot_utils.im_colorbar(im_pred, spacing = 0.03)
            cbar_err = utils.plot_utils.im_colorbar(im_err, spacing = 0.03)
            for item in (cbar_pred, cbar_err):
                item.ax.tick_params(labelsize = 17)      
            axs[count,1].set_title(
                title,
                fontsize = 18, 
                pad = 17
            )
            axs[count,2].set_title(
                'Absolute error',
                fontsize = 18, 
                pad = 17
            )
        for j in range(1,3):
            axs[j,0].spines['top'].set_visible(False)
            axs[j,0].spines['right'].set_visible(False)
            axs[j,0].spines['bottom'].set_visible(False)
            axs[j,0].spines['left'].set_visible(False)
        for j in range(3):
            for k in range(3):
                axs[j,k].axes.get_xaxis().set_ticks([])
                axs[j,k].axes.get_yaxis().set_ticks([])  
        plt.savefig(
            os.path.join(
                dir,
                'eikonal_all_paradigms' + str(i + 1) + '.png'
            )
        )
    
    plt.style.use('dark_background')
    cmap_slides = LinearSegmentedColormap.from_list('my_gradient', (
        (0.000, (1.000, 1.000, 1.000)),
        (0.500, (1.000, 0.000, 0.050)),
        (1.000, (0.300, 0.000, 0.050))
        )
    )
    visualize_instances(dir = slides_dir, cmap = 'gist_rainbow')
    plt.style.use('default')
    visualize_instances(dir = results_dir, cmap = 'jet')

################################################################################
# POSTPROCESSING: Final error analysis for the ablation study
################################################################################

def postprocess_error_analysis(dir, info_plot):
    """ Plots the results of the ablation study
    """

    test_params_unique = np.unique(data_loader.data_model['test']['mu'])
    #---------------------------------------------------------------------------
    # Scatterplot of selected parameter instances for the ablation study
    #---------------------------------------------------------------------------

    def visualize_parameter_instances(enable_ablation = True):

        # Initialize figure
        plt.figure(figsize = (15,3))

        # Plot supervised parameter space
        plt.axvspan(
            np.min(train_val_params_unique), 
            np.max(train_val_params_unique), 
            color = info_plot['parameter_space_color'], 
            alpha = info_plot['parameter_space_alpha'], 
            label = '$\mathcal{P}_{sup}$'
        )

        # Plot training instances
        if enable_ablation:
            for i in range(len(instances_selections)):
                if i == 0:
                    to_plot = params_selection[i]
                else:
                    to_plot = list(set(params_selection[i]) - \
                                set(params_selection[i-1]))
                plt.scatter(
                    to_plot,
                    0.05 * np.ones(len(to_plot)), 
                    color = info_plot['color_codes'][i],
                    marker = 'x', 
                    label = '$N_{data}$ = ' + str(len(instances_selections[i])),
                    s = 100, 
                    linewidths = 3
                )
        else:
             plt.scatter(
                params_selection[-1],
                0.05 * np.ones(len(params_selection[-1])), 
                color = 'cyan',
                marker = 'x',
                label = 'train data',
                s = 100, 
                linewidths = 3
            )
        
        # Plot testing instances
        plt.scatter(
            test_params_unique,
            - 0.05 * np.ones_like(test_params_unique), 
            color = info_plot['test_data_color'],
            marker = 'o', 
            label = 'test data'
        )

        # Adjust the plots and prints legend
        plt.xlim(0,1)
        plt.ylim(-0.1,0.1)
        plt.xlabel('$\mu$', fontsize = 24)
        plt.subplots_adjust(
            left = 0.03, 
            right = 0.97, 
            hspace = 0.4, 
            wspace = 0.5, 
            top = 0.75,
            bottom = 0.25
        )
        plt.gca().axes.yaxis.set_ticks([])
        plt.gca().axes.tick_params(labelsize = 20)  
        plt.legend(
            bbox_to_anchor = (0, 1.02, 1, 0.2), 
            loc =   "lower left",
            mode = "expand", 
            borderaxespad = 0, 
            ncol = 6, 
            fontsize = 20
        )

        if enable_ablation:
            plt.savefig(os.path.join(dir, 'instances-selection-eikonal.png'))
        else:
            plt.savefig(os.path.join(dir, 'instances-eikonal-no-abl.png'))

    # Call ablation study function
    visualize_parameter_instances()
    visualize_parameter_instances(enable_ablation = False)


    #---------------------------------------------------------------------------
    # Relative error plot for the ablation study
    #---------------------------------------------------------------------------

    def ablation_study(enable_ptpi = True):
        # Initialize the figure
        plt.figure(figsize = (15,8))

        # Plot supervised parameter space
        plt.axvspan(
            np.min(train_val_params_unique), 
            np.max(train_val_params_unique), 
            color = info_plot['parameter_space_color'], 
            alpha = info_plot['parameter_space_alpha'], 
            label = '$\mathcal{P}_{sup}$'
        )

        # Relative error plots
        for i in range(len(instances_selections)):
            if enable_ptpi:
                plt.semilogy(test_params_unique, 
                    errors_ptpi[i], 
                    color = info_plot['color_codes'][i], 
                    label = '$N_{data} = $' + str(len(instances_selections[i])),
                    linewidth = 3
                )
                plt.semilogy(test_params_unique, 
                    errors_dd[i], 
                    color = info_plot['color_codes'][i], 
                    linestyle = '--',
                    linewidth = info_plot['error_dd_linewidth']
                )
            else:
                plt.semilogy(test_params_unique, 
                    errors_dd[i], 
                    color = info_plot['color_codes'][i], 
                    label = '$N_{data} = $' + str(len(instances_selections[i])),
                    linestyle = '--',
                    linewidth = info_plot['error_dd_linewidth']
                )
        # Adjust plot and print legend
        plt.xlim(0,1)
        plt.xlabel('$\mu$', fontsize = 24)
        plt.ylabel('$e(\mu,N_{data})$', fontsize = 24)
        plt.legend(
            bbox_to_anchor = (0, 1.02, 1, 0.2), 
            loc =   "lower left",
            mode = "expand", 
            borderaxespad = 0, 
            ncol = 5, 
            fontsize = 20
        )
        plt.subplots_adjust(
            left = 0.1, 
            right = 0.9, 
            hspace = 0.4, 
            wspace = 0.5, 
            top = 0.85,
            bottom = 0.15
        )
        plt.gca().axes.tick_params(labelsize = 20) 

        # Save the figure
        if enable_ptpi:
            plt.savefig(os.path.join(dir, 'errors-eikonal.png'))
        else:
            plt.savefig(os.path.join(dir, 'errors-eikonal-only-dd.png'))
        
    # Call ablation study function
    ablation_study()
    ablation_study(enable_ptpi = False)



# Call the postprocess function
plt.style.use('dark_background')
info_plot = dict()
info_plot['parameter_space_color'] = 'white'
info_plot['parameter_space_alpha'] = 0.3
info_plot['test_data_color'] = 'white'
info_plot['error_dd_linewidth'] = 1
info_plot['color_codes'] = [
    [0.0,1.0,0.0], [0.0,1.0,1.0], [1.0,0.0,1.0], 'orange'
]
postprocess_error_analysis(
    dir = slides_dir,
    info_plot = info_plot
)

plt.style.use('default')
info_plot = dict()
info_plot['parameter_space_color'] = 'r'
info_plot['parameter_space_alpha'] = 0.1
info_plot['test_data_color'] = 'gray'
info_plot['error_dd_linewidth'] = 3
info_plot['color_codes'] = [
    [0.0,1.0,0.0], [0.0,1.0,1.0], [1.0,0.0,1.0], [0.0,0.0,0.0]
]
postprocess_error_analysis(
    dir = results_dir,
    info_plot = info_plot
)


