import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from sklearn import preprocessing
from sklearn.linear_model import Ridge
import sys
from pathlib import Path
import numpy as onp
import jax
import jax.numpy as jnp
import joblib
import sys
import time
from scipy.stats import mannwhitneyu, spearmanr, rankdata

from soromox.rendering import MatplotlibRenderer, OpenCVPlanarRenderer, Open3DRenderer, RendererColorConfig, BackboneColorConfig, ViserRenderer

curr_folder = Path(__file__).parent # current folder
sys.path.append(str(curr_folder.parent)) # scripts folder
main_folder = curr_folder.parent.parent # main folder "codes"
plots_folder = main_folder/'plots and videos' # folder for plots and videos
dataset_folder = main_folder/'datasets' # folder with the dataset
data_folder = main_folder/'saved data' # folder for saving data
from utilis import *

# Jax settings
jax.config.update("jax_enable_x64", True)  # double precision
jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)
seed = 123
key = jax.random.key(seed)

# Plots settings
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset':   'cm',
})


### PRC accuracy vs reservoir dimension (sMNIST)
if False:
    # Data
    ny = np.arange(6, 16, 3)

    acc_ron = np.array([65.88, 68.66, 70.51, 78.57])
    acc_untrained = np.array([43.28, 42.07, 52.63, 44.08])
    acc_part_trained = np.array([59.99, 61.86, 62.68, 60.61])
    acc_trained = np.array([64.03, 66.78, 66.18, 71.30])
        
    stddev_untrained = np.array([7.32, 8.89, 5.43, 12.79])
    stddev_part_trained = np.array([0.83, 1.06, 2.39, 12.16])
    stddev_trained = np.array([1.32, 1.82, 2.41, 0.63])

    # Plot
    fig = plt.figure(figsize=(4.1, 3))

    plt.plot(ny, acc_ron, label='Virtual RON', color="#7E7E7E", linestyle='--', linewidth='1.8', marker='^', markersize='8') # RON

    plt.plot(ny, acc_untrained, label='Unoptimized', color="#49C5FF", linestyle='-', linewidth='1.8', marker='s', markersize='8') # unoptimized
    plt.fill_between(ny, acc_untrained-stddev_untrained, acc_untrained+stddev_untrained, color ="#49C5FF", alpha=0.15)

    plt.plot(ny, acc_part_trained, label='Partially pretrained', color="#FF9924", linestyle='-', linewidth='1.8', marker='D', markersize='8') # partially trained
    plt.fill_between(ny, acc_part_trained-stddev_part_trained, acc_part_trained+stddev_part_trained, color ="#FF9924", alpha=0.15)

    plt.plot(ny, acc_trained, label='Pretrained (ours)', color="#10C200FF", linestyle='-', linewidth='1.8', marker='o', markersize='8') # partially trained
    plt.fill_between(ny, acc_trained-stddev_trained, acc_trained+stddev_trained, color ="#10C200FF", alpha=0.15)

    plt.grid(True, alpha=0.6)
    plt.xlabel(r'$n_y$', fontsize=13)
    plt.ylabel(r'accuracy [%]', fontsize=13)
    plt.title(r'sMNIST ($\uparrow$)', fontsize=14)
    plt.xticks(np.arange(6, 16), fontsize=13)
    plt.yticks(fontsize=13)
    #plt.yticks(np.arange(30, 81, 10))
    plt.xlim(5.5, 15.5)
    #plt.ylim(30, 98)
    #plt.legend(ncol=2, fontsize=9, loc='upper left')

    fig.tight_layout()
    #fig.savefig("smnist_scalability.pdf", bbox_inches="tight")
    #plt.show()


### PRC accuracy vs reservoir dimension (ADIAC)
if False:
    # Data
    ny = np.arange(6, 16, 3)

    acc_ron = np.array([46.80, 53.20, 57.03, 62.15])
    acc_untrained = np.array([30.54, 38.57, 40.30, 45.42])
    acc_part_trained = np.array([39.35, 37.08, 32.92, 32.26])
    acc_trained = np.array([44.58, 51.01, 50.18, 53.10])
        
    stddev_untrained = np.array([3.22, 3.20, 7.30, 16.85])
    stddev_part_trained = np.array([6.41, 16.21, 11.68, 11.18])
    stddev_trained = np.array([1.03, 3.75, 0.68, 2.45])

    # Plot
    fig = plt.figure(figsize=(4.1, 3))

    plt.plot(ny, acc_ron, label='Virtual RON', color="#7E7E7E", linestyle='--', linewidth='1.8', marker='^', markersize='8') # RON

    plt.plot(ny, acc_untrained, label='Unoptimized', color="#49C5FF", linestyle='-', linewidth='1.8', marker='s', markersize='8') # unoptimized
    plt.fill_between(ny, acc_untrained-stddev_untrained, acc_untrained+stddev_untrained, color ="#49C5FF", alpha=0.15)

    plt.plot(ny, acc_part_trained, label='Partially pretrained', color="#FF9924", linestyle='-', linewidth='1.8', marker='D', markersize='8') # partially trained
    plt.fill_between(ny, acc_part_trained-stddev_part_trained, acc_part_trained+stddev_part_trained, color ="#FF9924", alpha=0.15)

    plt.plot(ny, acc_trained, label='Pretrained (ours)', color="#10C200FF", linestyle='-', linewidth='1.8', marker='o', markersize='8') # pretrained
    plt.fill_between(ny, acc_trained-stddev_trained, acc_trained+stddev_trained, color ="#10C200FF", alpha=0.15)

    plt.grid(True, alpha=0.6)
    plt.xlabel(r'$n_y$', fontsize=13)
    plt.ylabel(r'accuracy [%]', fontsize=13)
    plt.title(r'ADIAC ($\uparrow$)', fontsize=14)
    plt.xticks(np.arange(6, 16), fontsize=13)
    plt.yticks(fontsize=13)
    #plt.yticks(np.arange(30, 81, 10))
    plt.xlim(5.5, 15.5)
    #plt.ylim(18, 80)
    #plt.legend(ncol=2, fontsize=9, loc='upper left')

    fig.tight_layout()
    #fig.savefig("smnist_scalability.pdf", bbox_inches="tight")
    #plt.show()


### PRC accuracy vs reservoir dimension (Mackey-Glass)
if False:
    # Data
    ny = np.arange(6, 16, 3)

    acc_ron = np.array([0.513, 0.432, 0.358, 0.309])
    acc_untrained = np.array([0.717, 0.744, 0.697, 0.730])
    acc_part_trained = np.array([0.597, 0.582, 0.621, 0.485])
    acc_trained = np.array([0.525, 0.505, 0.435, 0.424])
        
    stddev_untrained = np.array([0.137, 0.065, 0.139, 0.058])
    stddev_part_trained = np.array([0.132, 0.028, 0.020, 0.011])
    stddev_trained = np.array([0.003, 0.016, 0.014, 0.006])

    # Plot
    fig = plt.figure(figsize=(4.1, 3))

    plt.plot(ny, acc_ron, label='Virtual RON', color="#7E7E7E", linestyle='--', linewidth='1.8', marker='^', markersize='8') # RON

    plt.plot(ny, acc_untrained, label='Unoptimized', color="#49C5FF", linestyle='-', linewidth='1.8', marker='s', markersize='8') # unoptimized
    plt.fill_between(ny, acc_untrained-stddev_untrained, acc_untrained+stddev_untrained, color ="#49C5FF", alpha=0.15)

    plt.plot(ny, acc_part_trained, label='Partially pretrained', color="#FF9924", linestyle='-', linewidth='1.8', marker='D', markersize='8') # partially trained
    plt.fill_between(ny, acc_part_trained-stddev_part_trained, acc_part_trained+stddev_part_trained, color ="#FF9924", alpha=0.15)

    plt.plot(ny, acc_trained, label='Pretrained (ours)', color="#10C200FF", linestyle='-', linewidth='1.8', marker='o', markersize='8') # pretrained
    plt.fill_between(ny, acc_trained-stddev_trained, acc_trained+stddev_trained, color ="#10C200FF", alpha=0.15)

    plt.grid(True, alpha=0.6)
    plt.xlabel(r'$n_y$', fontsize=13)
    plt.ylabel(r'NRMSE [–]', fontsize=13)
    plt.title(r'Mackey-Glass ($\downarrow$)', fontsize=14)
    plt.xticks(np.arange(6, 16), fontsize=13)
    plt.yticks(fontsize=13)
    #plt.yticks(np.arange(0.2, 0.9, 0.1))
    plt.xlim(5.5, 15.5)
    #plt.ylim(0.08, 0.88)
    #plt.legend(ncol=2, fontsize=9)

    fig.tight_layout()
    #fig.savefig("mg_scalability.pdf", bbox_inches="tight")
    #plt.show()


### PRC accuracy vs reservoir dimension (Lorenz96)
if False:
    # Data
    ny = np.arange(6, 16, 3)

    acc_ron = np.array([0.564, 0.515, 0.417, 0.385])
    acc_untrained = np.array([0.700, 0.685, 0.519, 0.680])
    acc_part_trained = np.array([0.662, 0.630, 0.506, 0.575])
    acc_trained = np.array([0.567, 0.522, 0.463, 0.455])
        
    stddev_untrained = np.array([0.104, 0.181, 0.042, 0.211])
    stddev_part_trained = np.array([0.106, 0.118, 0.021, 0.181])
    stddev_trained = np.array([0.002, 0.001, 0.011, 0.013])

    # Plot
    fig = plt.figure(figsize=(4.1, 3))

    plt.plot(ny, acc_ron, label='Virtual RON', color="#7E7E7E", linestyle='--', linewidth='1.8', marker='^', markersize='8') # RON

    plt.plot(ny, acc_untrained, label='Unoptimized', color="#49C5FF", linestyle='-', linewidth='1.8', marker='s', markersize='8') # unoptimized
    plt.fill_between(ny, acc_untrained-stddev_untrained, acc_untrained+stddev_untrained, color ="#49C5FF", alpha=0.15)

    plt.plot(ny, acc_part_trained, label='Partially pretrained', color="#FF9924", linestyle='-', linewidth='1.8', marker='D', markersize='8') # partially trained
    plt.fill_between(ny, acc_part_trained-stddev_part_trained, acc_part_trained+stddev_part_trained, color ="#FF9924", alpha=0.15)

    plt.plot(ny, acc_trained, label='Pretrained (ours)', color="#10C200FF", linestyle='-', linewidth='1.8', marker='o', markersize='8') # prerained
    plt.fill_between(ny, acc_trained-stddev_trained, acc_trained+stddev_trained, color ="#10C200FF", alpha=0.15)

    plt.grid(True, alpha=0.6)
    plt.xlabel(r'$n_y$', fontsize=13)
    plt.ylabel(r'NRMSE [–]', fontsize=13)
    plt.title(r'Lorenz96 ($\downarrow$)', fontsize=14)
    plt.xticks(np.arange(6, 16), fontsize=13)
    plt.yticks(fontsize=13)
    #plt.yticks(np.arange(0.2, 0.9, 0.1))
    plt.xlim(5.5, 15.5)
    #plt.ylim(0.18, 0.9)
    #plt.legend(ncol=2, fontsize=9, loc='lower left')

    fig.tight_layout()
    #fig.savefig("mg_scalability.pdf", bbox_inches="tight")
    #plt.show()

    # Legend in a separate figure
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()

    fig_leg = plt.figure(figsize=(9, 0.5))
    fig_leg.legend(handles, labels, loc='center', ncol=4, frameon=True, fontsize=9)
    plt.show()


### Inference animations (Mackey-Glass)
if False:
    # =====================================================
    # Script settings
    # =====================================================

    # General
    dt_u = 0.15 # time step for the input u. (in the RON paper dt = 0.17 s)
    Nw = 200 # washout steps for the Mackey-Glass task
    Nl = 84 # prediction lag for the Mackey-Glass task

    # Output layer (scaler + predictor)
    experiment_name = data_folder/'more_soft_robots_reservoir'/'MG'/'N12'/'a_run3' # name of the experiment to save/load
    train = False # if True, perform training (output layer). Otherwise, test saved 'experiment_name' model

    # Reservoir (robots + map + controller)
    load_model_path = data_folder/'more_soft_robots_optimization'/'MG/N12/default_run3' # choose the reservoir to load (robots + map + controller)
    map_type = 'linear' # 'linear', 'encoder-decoder', 'bijective', 'none'
    controller_type = 'fb+ff' # if 'unique': Tau = Tau_tot(Z,u). If 'fb+ff': Tau = Tau_fb(Z) + Tau_ff(u). If 'ff': Tau = Tau_ff(u) (randomly initialized tanh(V*u+d)) !!! If 'unique', the controller tau_tot is defined in fb_controller_type
    fb_controller_type = 'mlp' # 'linear_simple', 'linear_complete', 'tanh_simple', 'tanh_complete', 'mlp'
    ff_controller_type = 'mlp' # 'linear', 'tanh', 'mlp'
    robots_type = 'saved' # 'saved' (those in 'load_model_path'), 'random' (randomly sampled), 'default'


    # =====================================================
    # Datasets
    # =====================================================

    # Load Mackey-Glass dataset
    (
        (train_dataset, train_target),
        (valid_dataset, valid_target),
        (test_dataset, test_target),
    ) = load_mackey_glass_data(csvfolder=dataset_folder/'MG', lag=Nl, washout=Nw, train_portion=0.2, val_portion=0.6)

    # Convert to jax
    train_dataset = jnp.array([train_dataset], dtype=jnp.float64).T # sequence from k=0 to k=N-Nl-1. Shape (N-Nl, 1)
    valid_dataset = jnp.array([valid_dataset], dtype=jnp.float64).T # sequence from k=0 to k=N-Nl-1. Shape (N-Nl, 1)
    test_dataset = jnp.array([test_dataset], dtype=jnp.float64).T # sequence from k=0 to k=N-Nl-1. Shape (N-Nl, 1)

    N_train = len(train_dataset) # N for the train set sequence
    N_test = len(test_dataset) # N for the test set sequence


    # =====================================================
    # Define the reservoir
    # =====================================================

    # Define robots system
    data_robot_load = onp.load(load_model_path/'optimal_data_robot.npz')

    L = jnp.array(data_robot_load['L'], dtype=jnp.float64)
    D = jnp.array(data_robot_load['D'], dtype=jnp.float64)
    r = jnp.array(data_robot_load['r'], dtype=jnp.float64)
    rho = jnp.array(data_robot_load['rho'], dtype=jnp.float64)
    E = jnp.array(data_robot_load['E'], dtype=jnp.float64)
    G = jnp.array(data_robot_load['G'], dtype=jnp.float64)
    if len(L.shape) == 1:
        n_robots = 1
        n_pcs = L.shape[0]
        L = jnp.expand_dims(L, axis=0)
        D = jnp.expand_dims(D, axis=0)
        r = jnp.expand_dims(r, axis=0)
        rho = jnp.expand_dims(rho, axis=0)
        E = jnp.expand_dims(E, axis=0)
        G = jnp.expand_dims(G, axis=0)
    else:
        n_robots, n_pcs = L.shape

    if robots_type == 'default':
        L = jnp.tile(1e-1 * jnp.ones(n_pcs), (n_robots,1))
        D = jnp.tile(jnp.diag(jnp.tile(jnp.array([5e-6, 5e-3, 5e-3]), n_pcs)), (n_robots,1,1))
        r = jnp.tile(2e-2 * jnp.ones(n_pcs),(n_robots,1))
        rho = jnp.tile(1070 * jnp.ones(n_pcs),(n_robots,1))
        E = jnp.tile(2e3 * jnp.ones(n_pcs),(n_robots,1))
        G = jnp.tile(1e3 * jnp.ones(n_pcs),(n_robots,1))
    elif robots_type == 'random':
        key, *keys_robot = jax.random.split(key, 9)
        L_init = jax.random.uniform(keys_robot[0], minval=7e-2, maxval=3e-1)
        D_init_1 = jax.random.uniform(keys_robot[1], minval=5e-7, maxval=5e-5)
        D_init_2 = jax.random.uniform(keys_robot[2], minval=5e-4, maxval=5e-2)
        D_init_3 = jax.random.uniform(keys_robot[3], minval=5e-4, maxval=5e-2)
        r_init = jax.random.uniform(keys_robot[4], minval=7e-3, maxval=5e-2)
        rho_init = jax.random.uniform(keys_robot[5], minval=900, maxval=1200)
        E_init = jax.random.uniform(keys_robot[6], minval=1800, maxval=2200)
        G_init = jax.random.uniform(keys_robot[7], minval=800, maxval=1200)

        L = jnp.tile(L_init * jnp.ones(n_pcs), (n_robots,1))
        D = jnp.tile(jnp.diag(jnp.tile(jnp.array([D_init_1, D_init_2, D_init_3]), n_pcs)), (n_robots,1,1))
        r = jnp.tile(r_init * jnp.ones(n_pcs),(n_robots,1))
        rho = jnp.tile(rho_init * jnp.ones(n_pcs),(n_robots,1))
        E = jnp.tile(E_init * jnp.ones(n_pcs),(n_robots,1))
        G = jnp.tile(G_init * jnp.ones(n_pcs),(n_robots,1))
    else:
        pass

    pcs_parameters = {
        "th0": jnp.tile(jnp.array(jnp.pi/2), n_robots),
        "L": L,
        "r": r,
        "rho": rho,
        "g": jnp.tile(jnp.array([0.0, 9.81]), (n_robots,1)), # !! gravity UP !!
        "E": E,
        "G": G,
        "D": D
    }
    robots_system = MultiPcsSystem(
        n_robots = n_robots,
        n_pcs = n_pcs,
        params_robots = pcs_parameters
    )

    # Define mapping
    match map_type:
        case 'linear':
            data_map = onp.load(load_model_path/'optimal_data_map.npz')
            A = jnp.array(data_map['A'], dtype=jnp.float64)
            c = jnp.array(data_map['c'], dtype=jnp.float64)

            def map_direct(y, yd, A, c):
                q = A @ y + c
                qd = A @ yd
                return q, qd
            map_direct = jax.jit(partial(map_direct, A=A, c=c))

            def map_inverse(q, qd, A, c):
                y = jnp.linalg.solve(A, (q - c).T).T
                yd = jnp.linalg.solve(A, qd.T).T
                return y, yd 
            map_inverse = jax.jit(partial(map_inverse, A=A, c=c))
            
        case 'encoder-decoder':
            mlp_map_loader = MLP(key, [1, 1]) # instance just for loading parameters
            p_encoder = mlp_map_loader.load_params(load_model_path/'optimal_data_encoder.npz') # tuple ((W1, b1), (W2, b2), ...)
            p_decoder = mlp_map_loader.load_params(load_model_path/'optimal_data_decoder.npz') # tuple ((W1, b1), (W2, b2), ...)

            layers_dim = []
            for i, layer in enumerate(p_encoder):
                W = layer[0] # shape (n_out_layer, n_in_layer)
                layers_dim.append(W.shape[1]) # n_in_layer
            layers_dim.append(W.shape[0]) # last layer: add output dimension (i.e. n_out_layer for the last layer)
            mlp_encoder_dummy = MLP(key, layers_dim)
            mlp_encoder = mlp_encoder_dummy.update_params(p_encoder)

            layers_dim = []
            for i, layer in enumerate(p_decoder):
                W = layer[0] # shape (n_out_layer, n_in_layer)
                layers_dim.append(W.shape[1]) # n_in_layer
            layers_dim.append(W.shape[0]) # last layer: add output dimension (i.e. n_out_layer for the last layer)
            mlp_decoder_dummy = MLP(key, layers_dim)
            mlp_decoder = mlp_decoder_dummy.update_params(p_decoder)
            
            def map_direct(y, yd, encoder):
                q, qd = encoder.forward_xd(y, yd)
                return q, qd
            map_direct = jax.jit(partial(map_direct, encoder=mlp_encoder))

            def map_inverse(q, qd, decoder):
                y, yd = decoder.forward_xd_batch(q, qd)
                return y, yd
            map_inverse = jax.jit(partial(map_inverse, decoder=mlp_decoder))

        case 'bijective':
            realnvp_map_loader = RealNVP(key, [jnp.ones(1)], 1, activation_fn='tanh') # instance just for loading parameters
            p_map = realnvp_map_loader.load_params(load_model_path/'optimal_data_map.npz')

            n_layers = len(p_map) # number of coupling layers
            masks = create_alternating_masks(input_dim=3*n_pcs*n_robots, num_layers=n_layers)
            hid_dim = p_map[0][0][0][0].shape[0]

            realnvp_map_dummy = RealNVP(key, masks, hid_dim, activation_fn='tanh')
            realnvp_map = realnvp_map_dummy.update_params(p_map)

            def map_direct(y, yd, map):
                q, qd = map.forward_with_derivatives(y, yd)
                return q, qd
            map_direct = jax.jit(partial(map_direct, map=realnvp_map))

            def map_inverse(q, qd, map):
                y, yd = map.inverse_with_derivatives_batch(q, qd)
                return y, yd
            map_inverse = jax.jit(partial(map_inverse, map=realnvp_map))
        
        case 'none':
            @jax.jit
            def map_direct(y, yd):
                q, qd = y, yd
                return q, qd

            @jax.jit
            def map_inverse(q, qd):
                y, yd = q, qd
                return y, yd

    # Define controller
    mlp_controller_loader = MLP(key, [1, 1]) # instance just for loading parameters
    if controller_type == 'unique':
        # load parameters
        p_controller = mlp_controller_loader.load_params(load_model_path/'optimal_data_controller.npz') # tuple ((W1, b1), (W2, b2), ...)
        # find out layers and dimensions: layers_dim = [dim_in, dim_hid1, dim_hid2, ..., dim_out]
        layers_dim = []
        for i, layer in enumerate(p_controller):
            W = layer[0] # shape (n_out_layer, n_in_layer)
            layers_dim.append(W.shape[1]) # n_in_layer
            if i == 0:
                n_input = W.shape[1] # save input dim for the controller
        layers_dim.append(W.shape[0]) # last layer: add output dimension (i.e. n_out_layer for the last layer)
        # set activation fn for the last layer
        if fb_controller_type == 'tanh_simlpe' or fb_controller_type == 'tanh_complete':
            last_activation_fn = 'tanh'
        else:
            last_activation_fn = 'linear'
        # re-build controller
        mlp_controller_dummy = MLP(key, layers_dim, last_layer=last_activation_fn)
        mlp_controller = mlp_controller_dummy.update_params(p_controller)
            
        def controller(z, u, mlp_controller):
            if n_input == 3*n_pcs*n_robots + 1:
                q, qd = jnp.split(z, 2)
                input_controller = jnp.concatenate([q, u])
            else:
                input_controller = jnp.concatenate([z, u])
            tau = mlp_controller(input_controller)
            return tau
        controller = jax.jit(partial(controller, mlp_controller=mlp_controller))

    elif controller_type == 'fb+ff':
        # load parameters for fb and ff controllers
        p_fb_controller = mlp_controller_loader.load_params(load_model_path/'optimal_data_fb_controller.npz') # tuple ((W1, b1), (W2, b2), ...)
        p_ff_controller = mlp_controller_loader.load_params(load_model_path/'optimal_data_ff_controller.npz') # tuple ((W1, b1), (W2, b2), ...)
        # reconstruct fb controller
        layers_dim_fb = []
        for i, layer in enumerate(p_fb_controller):
            W = layer[0] # shape (n_out_layer, n_in_layer)
            layers_dim_fb.append(W.shape[1]) # n_in_layer
            if i == 0:
                n_input_fb = W.shape[1] # save input dim for the controller
        layers_dim_fb.append(W.shape[0]) # last layer: add output dimension (i.e. n_out_layer for the last layer)
        # set activation fn for the last layer
        if fb_controller_type == 'tanh_simlpe' or fb_controller_type == 'tanh_complete':
            last_activation_fn_fb = 'tanh'
        else:
            last_activation_fn_fb = 'linear'
        # re-build controller
        mlp_fb_controller_dummy = MLP(key, layers_dim_fb, last_layer=last_activation_fn_fb)
        mlp_fb_controller = mlp_fb_controller_dummy.update_params(p_fb_controller)

        # reconstruct ff controller
        layers_dim_ff = []
        for i, layer in enumerate(p_ff_controller):
            W = layer[0] # shape (n_out_layer, n_in_layer)
            layers_dim_ff.append(W.shape[1]) # n_in_layer
        layers_dim_ff.append(W.shape[0]) # last layer: add output dimension (i.e. n_out_layer for the last layer)
        # set activation fn for the last layer
        if ff_controller_type == 'tanh':
            last_activation_fn_ff = 'tanh'
        else:
            last_activation_fn_ff = 'linear'
        # re-build controller
        mlp_ff_controller_dummy = MLP(key, layers_dim_ff, last_layer=last_activation_fn_ff)
        mlp_ff_controller = mlp_ff_controller_dummy.update_params(p_ff_controller)
        
        # total controller
        def controller(z, u, mlp_fb_controller, mlp_ff_controller):
            tau_ff = mlp_ff_controller(u)
            if n_input_fb == 3*n_pcs*n_robots:
                q, qd = jnp.split(z, 2)
                tau_fb = mlp_fb_controller(q)
            else:
                tau_fb = mlp_fb_controller(z)
            tau = tau_fb + tau_ff
            return tau
        controller = jax.jit(partial(controller, mlp_fb_controller=mlp_fb_controller, mlp_ff_controller=mlp_ff_controller))

    else:
        # no fb controller case
        key, key_V, key_d = jax.random.split(key, 3)
        scal_input = jnp.tile(jnp.array([0.001, 0.1, 0.01]), n_pcs*n_robots)
        V = scal_input[:,None] * jax.random.uniform(key_V, shape=(3*n_pcs*n_robots,1), minval=0.0, maxval=1.0) # random input-to-hidden weights
        d = scal_input * jax.random.uniform(key_d, shape=(3*n_pcs*n_robots,), minval=-1.0, maxval=1.0) # random input-to-hodden bias
        def controller(z, u, V, d):
            tau_ff = jnp.tanh(V @ u + d)
            return tau_ff
        controller = jax.jit(partial(controller, V=V, d=d))

    # Instantiate the reservoir
    reservoir = MultiPcsReservoir(
        robots_system=robots_system,
        map_direct=map_direct,
        map_inverse=map_inverse,
        controller=controller
    )

    # Other stuff
    dt_sim = 1e-4 # time step for the simulation
    time_u_train = dt_u * jnp.arange(0, N_train) # define time vector for the train input sequence
    time_u_test = dt_u * jnp.arange(0, N_test) # define time vector for the test input sequence
    saveat_train = time_u_train # for saving simulation results
    saveat_test = time_u_test # for saving simulation results
    

    # =====================================================
    # Training
    # =====================================================
    print(f'--- Experiment ---\n'
        f'name:  {experiment_name}\n'
        f'model: {load_model_path}')
    print()

    if train:
        # Train the output layer (predictor) (1): pass the train input sequence to the model
        print(f'--- Generating activations for training ---')
        start = time.perf_counter()
        (
            _,
            state_reservoir_ts, # reservoir's states evolution from k=0 to k=N-Nl-1. Shape (N-Nl, 2*n_hid)
            _, # pcs's states evolution from k=0 to k=N-Nl-1. Shape (N-Nl, 2*n_robots*3*n_pcs)
            _, # pcs actuation. Shape (N-Nl, 3*n_pcs*n_robots)
            _
        ) = reservoir(train_dataset, time_u_train, saveat_train, dt_sim)
        
        y_ts, _ = jnp.split(state_reservoir_ts, 2, axis=1) # reservoir's position evolution from k=0 to k=N-Nl-1. Shape (N-Nl, n_hid)
        activations = y_ts[Nw:] # remove the initial washout steps. Shape (N-Nl-Nw, n_hid). It's the reservoir's states evolution from k=Nw to k=N-Nl-1
        activations.block_until_ready()
        stop = time.perf_counter() 
        elatime_forward_pass_training = stop - start
        print(f'Elapsed time: {elatime_forward_pass_training}')
        activations = onp.array(activations)

        # Train the output layer (2): logistic regression of the output layer
        print('\nTraining the output layer (regression)...')
        start = time.perf_counter()
        scaler = preprocessing.StandardScaler().fit(activations)
        activations = scaler.transform(activations)
        predictor = Ridge(max_iter=1000).fit(activations, train_target)
        stop = time.perf_counter()
        elatime_train_output_layer = stop - start
        print(f'Elapsed time: {elatime_train_output_layer}')

        # Save the trained predictor and scaler
        #joblib.dump(scaler, data_folder/'scaler.pkl')
        #joblib.dump(predictor, data_folder/'predictor.pkl')

        # Train accuracy
        pred = predictor.predict(activations)
        rmse = jnp.sqrt(jnp.mean((pred - train_target) ** 2))
        rms_target = jnp.sqrt(jnp.mean(train_target ** 2))
        train_nrmse = (rmse / rms_target)
        print(f'Train NRMSE: {train_nrmse}')


    # =====================================================
    # Testing on the test set
    # =====================================================
    if train:
        print()

    # If training was not performed, load saved data
    if not train:
        scaler = joblib.load(experiment_name/'scaler.pkl')
        predictor = joblib.load(experiment_name/'predictor.pkl')

    # Forward on the test set
    print(f'--- Evaluating perfomances (test set) ---')
    start = time.perf_counter()
    (
        time_ts,
        state_reservoir_ts, # reservoir's states evolution from k=0 to k=N-Nl-1. Shape (N-Nl, 2*n_hid)
        state_pcs_ts, # pcs's states evolution from k=0 to k=N-Nl-1. Shape (N-Nl, 2*3*n_pcs*n_robots)
        actuation_ts, # pcs actuation. Shape (N-Nl, 3*n_pcs*n_robots)
        _
    ) = reservoir(test_dataset, time_u_test, saveat_test, dt_sim)

    y_ts, _ = jnp.split(state_reservoir_ts, 2, axis=1) # reservoir's position evolution from k=0 to k=N-Nl-1. Shape (N-Nl, n_hid)
    activations = y_ts[Nw:] # remove the initial washout steps. Shape (N-Nl-Nw, n_hid). It's the reservoir's states evolution from k=Nw to k=N-Nl-1
    activations.block_until_ready()
    stop = time.perf_counter() 
    elatime_forward_pass_testing = stop - start
    print(f'Elapsed time: {elatime_forward_pass_testing}')

    # Prediction and test accuracy
    activations = onp.array(activations)
    activations = scaler.transform(activations)
    pred = predictor.predict(activations) # prediction from k=Nw+Nl to k=N-1
    rmse = jnp.sqrt(jnp.mean((pred - test_target) ** 2))
    rms_target = jnp.sqrt(jnp.mean(test_target ** 2))
    test_nrmse = (rmse / rms_target)
    print(f'Test NRMSE: {test_nrmse}')


    # =====================================================
    # Show results of the test
    # =====================================================

    # Prepare variables
    y_ts, yd_ts = jnp.split(state_reservoir_ts, 2, axis=1) # reservoir states
    _, q_ts, _ = jax.vmap(robots_system.transform_Z)(state_pcs_ts) # shape (n_steps, n_robots, 3*n_pcs)
    Q_ts, _ = jnp.split(state_pcs_ts, 2, axis=1) # shape (n_steps, 3*n_pcs*n_robots)
    full_time = dt_u * onp.arange(0, N_test + Nl)
    full_sequence = onp.concatenate([onp.array(test_dataset).squeeze(), test_target[-Nl:]]) # full MG test sequence

    # Show max 15 DOFs in the plots
    if 3*n_pcs*n_robots > 15:
        n_show = 15
    else:
        n_show = 3*n_pcs*n_robots

    n_cols = min(2, n_show)
    n_rows = int(np.ceil(n_show / n_cols))

    # Show predicted sequence
    fig, ax = plt.subplots(1,1, figsize=(20,6))
    ax.plot(full_time, full_sequence, 'k--', label='full sequence')
    ax.plot(full_time[Nw:-Nl], full_sequence[Nw:-Nl], 'k', label='test sequence')
    ax.plot(full_time[Nw+Nl:], pred, 'r', label='predicted sequence')
    ax.grid(True)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('x')
    ax.set_title('Mackey-Glass sequence')
    ax.legend()

    plt.tight_layout()
    #plt.savefig(plots_folder/'Prediction', bbox_inches='tight')
    #plt.show()

    # Show reservoir/robot evolution
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12,9))
    for i, ax in enumerate(axs.flatten()):
        ax.plot(time_ts, y_ts[:,i], 'b', label=r'reservoir')
        ax.plot(time_ts, Q_ts[:,i], 'r', label=r'soft robots')
        ax.grid(True)
        ax.set_xlabel('t [s]')
        ax.set_ylabel('y, Q')
        ax.set_title(f'Component {i+1}')
        ax.legend()
    plt.tight_layout()
    #plt.savefig(plots_folder/'Example_inference_evolution', bbox_inches='tight') 
    #plt.show()

    # Show actuation signal Tau(t)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16,13))
    for i, ax in enumerate(axs.flatten()):
        ax2 = ax.twinx()
        ax2.plot(time_ts, test_dataset.squeeze(), 'k', alpha=0.3, label=r'reservoir input $u(t)$')
        ax2.set_ylabel(r'$u$')
        ax2.set_ylim([-0.6, 0.4])

        ax.plot(time_ts, actuation_ts[:,i], 'r', label=r'robots actuation $\tau(t)$')
        ax.set_xlabel('t [s]')
        ax.set_ylabel(r'$\tau$')

        ax.grid(True)
        ax.set_title(f'Component {i+1}')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

    plt.tight_layout()
    #plt.savefig(plots_folder/'Example_inference_actuation', bbox_inches='tight') 
    #plt.show()
    plt.close('all')


    # =========================================================
    # Animations
    # =========================================================

    # Prediction plot

    # ── accorcia sequenze ──────────────────────────────────────────────────────────
    print()
    print('PLOT ANIMATION')
    print('full_time: ', full_time[0],full_time[-1],full_time.shape)
    print('test_time: ', full_time[Nw],full_time[-Nl])
    removed_timesteps = 600
    full_time = full_time[:-removed_timesteps]
    full_sequence = full_sequence[:-removed_timesteps]
    pred = pred[:-removed_timesteps]
    print('full_time: ', full_time[0],full_time[-1],full_time.shape)
    print('test_time: ', full_time[Nw],full_time[-Nl])

    # ── stile globale ──────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':        'serif',
        'font.serif':         ['Computer Modern Roman', 'DejaVu Serif'],
        'mathtext.fontset':   'cm',
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.linewidth':     0.8,
        'xtick.major.width':  0.8,
        'ytick.major.width':  0.8,
        'xtick.direction':    'out',
        'ytick.direction':    'out',
        'xtick.labelsize':    11,
        'ytick.labelsize':    11,
        'axes.labelsize':     13,
        'axes.titlesize':     14,
        'legend.fontsize':    11,
        'legend.framealpha':  0.9,
        'legend.edgecolor':   '#cccccc',
        'legend.handlelength': 2.2,
        'figure.dpi':         130,
    })

    # ── colori ─────────────────────────────────────────────────────────────────────
    C_FULL   = "#4D4D4D"   # grigio chiaro — sfondo/full sequence
    C_TEST   = "#353535"   # quasi-nero — test sequence
    C_PRED   = '#c0392b'   # rosso scuro — predicted

    # ── figura ─────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 5))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#fafafa')

    # grid sottile
    ax.grid(True, which='major', linestyle=':', linewidth=0.5, color='#cccccc', alpha=0.8)
    ax.set_axisbelow(True)

    # ── curve statiche ─────────────────────────────────────────────────────────────
    ax.plot(full_time, full_sequence,
            color=C_FULL, lw=1.2, linestyle='--', alpha=0.6,
            zorder=2)

    ax.plot(full_time[Nw:-Nl], full_sequence[Nw:-Nl],
            color=C_TEST, lw=1.6, alpha=0.6,
            label=r'Test sequence', zorder=3)

    # ── curva animata ──────────────────────────────────────────────────────────────
    line, = ax.plot([], [], color=C_PRED, lw=2.0, alpha=1.0,
                    label=r'Predicted sequence', zorder=4)
    
    # tempo istantaneo
    dot0, = ax.plot([], [], 'o', color='k', ms=5, zorder=5, label=r'Current time')

    # punto "testa" che scorre
    dot, = ax.plot([], [], 'o', color=C_PRED, ms=5, zorder=6, label=r'Prediction')

    # ── assi e titolo ──────────────────────────────────────────────────────────────
    ax.set_xlim(full_time[0], full_time[-1])
    ax.set_xlabel(r'Time $[s]$', labelpad=8)
    ax.set_ylabel(r'$u$', labelpad=8)
    ax.set_title(r'Mackey–Glass sequence — reservoir prediction', pad=12)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))

    legend = ax.legend(
        ncol=2,
        loc='upper left',
        bbox_to_anchor=(0.0, 1.18),
        borderpad=0.7,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    fig.tight_layout(pad=1.4)

    # ── animazione ─────────────────────────────────────────────────────────────────
    t_anim = full_time[Nw+Nl:]
    t0_anim = full_time[Nw:-Nl]
    p_anim = pred
    p0_anim = full_sequence[Nw:-Nl]

    def update(frame):
        if frame == 0:
            line.set_data([], [])
            dot0.set_data([], [])
            dot.set_data([], [])
            return line, dot
        line.set_data(t_anim[:frame], p_anim[:frame])
        dot0.set_data([t0_anim[frame-1]], [p0_anim[frame-1]])
        dot.set_data([t_anim[frame-1]], [p_anim[frame-1]])
        return line, dot0, dot

    ani = FuncAnimation(fig, update,
                        frames=len(t_anim),
                        interval=15,
                        blit=True)

    ani.save(plots_folder/'prediction_plot_animation.mp4', writer='ffmpeg', dpi=150)
    #plt.show()

    # Single robot
    parameters_single = {
        "th0": jnp.array(
            [jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0, 0.0]
        ),  # Initial position and orientation
        "L": L[0],
        "r": r[0],
        "rho": rho[0],
        "g": jnp.array([0.0, 0.0, 9.81]),  # Gravity vector [m/s^2]
        "E": E[0],
        "G": G[0],
    }
    parameters_single["D"] = jnp.diag(jnp.array([D[0,0,0], D[0,0,0], D[0,0,0], D[0,1,1], D[0,2,2], D[0,2,2],
                        D[0,3,3], D[0,3,3], D[0,3,3], D[0,4,4], D[0,5,5], D[0,5,5]]))

    # Robots animation

    # Viser web-based visualization (opens in browser)
    # ViserRenderer provides interactive 3D visualization in the browser
    # with GUI controls for playback, speed, and looping.
    # Plotly plots are automatically added to the GUI at the end of the sidebar
    import plotly.graph_objects as go
    from soromox.rendering import (
        BackboneColorConfig,
        MatplotlibRenderer,
        Open3DRenderer,
        RendererColorConfig,
        ViserRenderer,
        get_color_theme,
    )
    from soromox.systems import PCS

    # ── accorcia sequenze ──────────────────────────────────────────────────────────
    print()
    print('ROBOTS ANIMATION')
    print('simulation time: ', time_ts[0],time_ts[-1],time_ts.shape)
    time_ts = time_ts[Nw:-removed_timesteps]
    q_ts = q_ts[Nw:-removed_timesteps]
    print('simulation time: ', time_ts[0],time_ts[-1],time_ts.shape)

    # ── build robot ────────────────────────────────────────────────────────────────
    robot = PCS(
        num_segments=2,
        params=parameters_single,
    )

    viser_renderer = ViserRenderer(robot, num_points=80, backbone_style="discrete")

    # ── Increase strains for better animations (shape of q_ts is (n_steps, n_robots, 3*n_pcs))
    print(q_ts.shape)
    q_ts = q_ts.at[:,:,0].set(1*q_ts[:,:,0]) # bending 1
    q_ts = q_ts.at[:,:,3].set(1*q_ts[:,:,3]) # bending 2
    q_ts = q_ts.at[:,:,1].set(1*q_ts[:,:,1]) # axial 1
    q_ts = q_ts.at[:,:,4].set(1*q_ts[:,:,4]) # axial 2
    q_ts = q_ts.at[:,:,2].set(1*q_ts[:,:,2]) # shear 1
    q_ts = q_ts.at[:,:,5].set(1*q_ts[:,:,5]) # shear 2

    # ── Add missing dimensions (2D -> 3D) and reshape from (n_steps, n_robots, 3*n_pcs) to (n_robots, n_steps, n_dofs)
    q_new = jnp.zeros((time_ts.shape[0], n_robots, 12)) # (n_steps, n_robots, 3*n_pcs), 3*n_pcs=6 -> (n_steps, n_robots, n_dofs), n_dofs=12
    q_new = (
        q_new
        .at[:, :, 0].set(q_ts[:, :, 0])
        .at[:, :, 3].set(q_ts[:, :, 1])
        .at[:, :, 4].set(q_ts[:, :, 2])
        .at[:, :, 6].set(q_ts[:, :, 3])
        .at[:, :, 9].set(q_ts[:, :, 4])
        .at[:, :, 10].set(q_ts[:, :, 5])
    )
    q_new = jnp.transpose(q_new, axes=(1,0,2)) # (n_steps, n_robots, n_dofs) -> (n_robots, n_steps, n_dofs)
    render_color = RendererColorConfig()

    # ── Animation ──────────────────────────────────────────────────────────────────
    viser_renderer.render_sequence(
        time_ts,
        q_new,
        playback_speed=1.0,
        loop=True,
        autoplay=True,
        plot_configurations=False,
        base_offsets=jnp.array([[0,0.5,0],[0,-0.5,0]]),
        robot_name="PCS",
    )


    # =========================================================
    # Show all plots
    # =========================================================
    plt.show()
    #plt.close('all')


# Statistical significance of the benchmarking experiments (Mann-Whitney U test)
if False:
    """
    Mann-Whitney U significance tests for the benchmarking experiments: our approach
    vs baselines ("unoptimized", "partially pretrained").
    All raw per-seed values are hard-coded below (up to 4 runs per configuration).

    Handling of problematic runs:
    - NaN run  (None below)  -> excluded; the test simply uses the remaining runs.
    - Diverged run ("div:X")  -> kept in the Mann-Whitney test (rank-based, so it
        only counts as the worst rank), but excluded from descriptive mean/std.

    Metric direction:
    - sMNIST      -> Accuracy, higher is better.
    - Mackey-Glass / Lorenz96 -> NRMSE, lower is better.

    Test: one-sided Mann-Whitney U ("our approach is better"), exact method (small samples).
    """

    # Raw data (hard coded) and setup
    data = {
        "sMNIST": {
            "lower_is_better": False, # classification accuracy: higher is better
            6:  {"ours": [0.6424, 0.6534, 0.6218, 0.6434],
                "unoptimized": [0.3712, 0.5376, 0.3976, 0.4248],
                "partially": [0.6082, 0.5962, 0.5900, 0.6052]},
            9:  {"ours": [0.6798, 0.6444, 0.6626, 0.6844],
                "unoptimized": [0.3666, 0.5308, 0.4522, 0.3330],
                "partially": [0.6068, 0.6314, 0.6222, 0.6140]},
            12: {"ours": [0.6304, 0.6734, 0.6572, 0.6862],
                "unoptimized": [0.4718, 0.5996, 0.5042, 0.5296],
                "partially": [0.5994, 0.6426, 0.6506, 0.6148]},
            15: {"ours": [0.7196, 0.7140, 0.7138, 0.7044],
                "unoptimized": [0.3714, 0.6318, 0.3660, 0.3940],
                "partially": [0.7284, 0.5410, 0.4690, 0.6860]},
        },
        "ADIAC": {
                "lower_is_better": False, # classification accuracy: higher is better
                6:  {"ours": [0.4595, 0.4357, 0.4405, 0.4476],
                    "unoptimized": [0.3476, 0.2714, 0.3095, 0.2929],
                    "partially": [0.3048, 0.4381, 0.3881, 0.4429]},
                9:  {"ours": [0.4548, 0.5357, 0.5190, 0.5310],
                    "unoptimized": [0.4238, 0.3976, 0.3714, 0.3500],
                    "partially": [0.1643, 0.4929, 0.3190, 0.5071]},
                12: {"ours": [0.5024, 0.5024, 0.5095, 0.4929],
                    "unoptimized": [0.4905, 0.3381, 0.4357, 0.3476],
                    "partially": [0.2381, 0.2190, 0.4429, 0.4167]},
                15: {"ours": [0.5024, 0.5619, 0.5333, 0.5262],
                    "unoptimized": [0.4857, 0.2119, 0.6000, 0.5190],
                    "partially": [0.4167, 0.2381, 0.2143, 0.4214]},
            },
        "Mackey-Glass": {
            "lower_is_better": True, # NRMSE: lower is better
            6:  {"ours": [0.529971, 0.523017, 0.523637, 0.524097],
                "unoptimized": [0.713295, 0.896077, 0.563369, 0.696531],
                "partially": [0.795480, 0.523017, 0.538880, 0.532117]},
            9:  {"ours": [0.498168, 0.500019, 0.529598, 0.492952],
                "unoptimized": [0.728679, 0.839122, 0.720081, 0.689492],
                "partially": [0.509839, 0.562238, 0.613869, 0.570043]},
            12: {"ours": [0.428501, 0.456062, 0.429164, 0.427103],
                "unoptimized": [0.712862, 0.883061, 0.556907, 0.635217],
                "partially": [None, 0.621812, 0.599603, 0.639629]},
            15: {"ours": [0.421557, 0.433578, 0.425380, 0.420137],
                "unoptimized": [0.711437, 0.812779, 0.717735, 0.678939],
                "partially": [0.497475, 0.481226, "div:301681", 0.476295]},
        },
        "Lorenz96": {
            "lower_is_better": True, # NRMSE, lower is better
            6:  {"ours": [0.569331, 0.565004, 0.566337, 0.566348],
                "unoptimized": [0.672494, 0.847201, 0.602625, 0.675680],
                "partially": [0.606512, 0.813911, 0.574410, 0.652679]},
            9:  {"ours": [0.522059, 0.521936, 0.521430, 0.522149],
                "unoptimized": [0.955513, 0.589739, 0.595031, 0.599063],
                "partially": [0.530333, 0.730225, 0.526154, 0.735342]},
            12: {"ours": [0.464620, 0.476204, 0.461021, 0.449093],
                "unoptimized": [0.539597, "div:6.303236515258028e+90", 0.546038, 0.470961],
                "partially": [0.519680, 0.500026, 0.479805, 0.524575]},
            15: {"ours": [0.451392, 0.473310, 0.455775, 0.443129],
                "unoptimized": [0.872217, 0.471598, 0.850005, 0.524543],
                "partially": [0.507516, 0.466502, 0.479136, 0.844571]},
        },
    }

    TASKS = ["sMNIST", "ADIAC", "Mackey-Glass", "Lorenz96"]
    NYS = [6, 9, 12, 15]
    BASELINES = ["unoptimized", "partially"]

    def parse_runs(raw):
        """Return (valid, test):
        valid = numeric, non-diverged, non-NaN (for mean/std);
        test  = numeric incl. diverged, excl. NaN (for the U test)."""
        valid, test = [], []
        for x in raw:
            if x is None: # NaN run: dropped everywhere
                continue
            if isinstance(x, str) and x.startswith("div:"):
                test.append(float(x.split("div:")[1]))
            else:
                valid.append(float(x)); test.append(float(x))
        return valid, test

    # Print means and std devs for each case
    print("=" * 82)
    print("DESCRIPTIVE  (mean +/- std over VALID runs; NaN & diverged excluded)")
    print("=" * 82)
    hdr = f"{'Task':<14}{'ny':>3}   {'ours':>17}{'unoptimized':>19}{'partially':>19}"
    print(hdr); print("-" * len(hdr))
    for task in TASKS:
        for ny in NYS:
            cells = []
            for m in ["ours", "unoptimized", "partially"]:
                valid, _ = parse_runs(data[task][ny][m])
                mean = np.mean(valid)
                std = np.std(valid, ddof=1) if len(valid) > 1 else 0.0
                tag = "" if len(valid) == 4 else f"[n{len(valid)}]"
                cells.append(f"{mean:.4f}+-{std:.4f}{tag}")
            print(f"{task:<14}{ny:>3}   {cells[0]:>17}{cells[1]:>19}{cells[2]:>19}")

    # Mann-Whitney test
    print("\n" + "=" * 82)
    print("MANN-WHITNEY U   ours vs baseline   (one-sided: 'ours is better', exact)")
    print("=" * 82)
    results = []
    for task in TASKS:
        lower = data[task]["lower_is_better"]
        for ny in NYS:
            _, ours = parse_runs(data[task][ny]["ours"])
            for b in BASELINES:
                _, base = parse_runs(data[task][ny][b])
                x = np.array(ours) * (-1 if lower else 1)
                y = np.array(base) * (-1 if lower else 1)
                U, p = mannwhitneyu(x, y, alternative="greater", method="exact")
                results.append(dict(task=task, ny=ny, base=b,
                                    no=len(x), nb=len(y), U=U, p=p))
    m = len(results)
    for r in results:
        r["pbonf"] = min(1.0, r["p"] * m)

    print(f"{'Task':<14}{'ny':>3}{'baseline':>13}{'n_o':>4}{'n_b':>4}"
        f"{'U':>6}{'p':>9}{'p_bonf':>9}{'sig':>5}")
    print("-" * 74)
    for r in results:
        sig = "*" if r["p"] < 0.05 else ""
        print(f"{r['task']:<14}{r['ny']:>3}{r['base']:>13}{r['no']:>4}{r['nb']:>4}"
            f"{r['U']:>6.1f}{r['p']:>9.4f}{r['pbonf']:>9.4f}{sig:>5}")

    ns = sum(1 for r in results if r["p"] < 0.05)
    print(f"\nSignificant (uncorrected, p<0.05): {ns}/{m}")
    print(f"Significant (Bonferroni):          "
        f"{sum(1 for r in results if r['pbonf'] < 0.05)}/{m}")
    print("Floor: 4v4 -> min one-sided p = 1/C(8,4) = 0.0143 ; "
        "4v3 -> 1/C(7,4) = 0.0286.")


### FLOPs count
if False:
    # Prepare data and functions
    n_y = np.arange(6, 500)
    
    def flops_ron(n_y, n_u, K, n_c=None, N_lag=None):
        """
        Args
        ----
        n_y : reservoir hidden dimension
        n_u : input dimension
        K : total timesteps
        n_c : number of classes (only for classification tasks, otherwise None)
        N_lag : prediction lag (only for forecasting, otherwise None)

        Return
        ------
        FLOPs : tuple (flops_reservoir, flops_readout, flops_total)
        """
        assert n_c is not None or N_lag is not None
        flops_reservoir = K * (2 * n_y**2 + 9 * n_y + 2 * n_y * n_u)
        flops_readout = 2 * n_y * n_u * (K - N_lag) if n_c is None else 2 * n_y * n_c + 3 * n_c
        flops_total = flops_reservoir + flops_readout
        return (flops_reservoir, flops_readout, flops_total)
    
    def flops_physical(n_y, n_u, K, n_c=None, N_lag=None):
        """
        Args
        ----
        n_y : reservoir hidden dimension
        n_u : input dimension
        K : total timesteps
        n_c : number of classes (only for classification tasks, otherwise None)
        N_lag : prediction lag (only for forecasting, otherwise None)

        Return
        ------
        FLOPs : tuple (flops_reservoir, flops_readout, flops_total)
        """
        assert n_c is not None or N_lag is not None
        flops_reservoir = K * (385 * n_y + 128 * n_u + 16640)
        flops_readout = (2 * n_u * n_y) * (K - N_lag) if n_c is None else 2 * n_c * n_y + 3 * n_c
        flops_total = flops_reservoir + flops_readout
        return (flops_reservoir, flops_readout, flops_total)

    def flops_physical_simpler(n_y, n_u, K, n_c=None, N_lag=None):
            """
            Args
            ----
            n_y : reservoir hidden dimension
            n_u : input dimension
            K : total timesteps
            n_c : number of classes (only for classification tasks, otherwise None)
            N_lag : prediction lag (only for forecasting, otherwise None)
    
            Return
            ------
            FLOPs : tuple (flops_reservoir, flops_readout, flops_total)
            """
            assert n_c is not None or N_lag is not None
            flops_reservoir = K * (96 * n_y + 2 * n_y * n_u + 17)
            flops_readout = (2 * n_u * n_y) * (K - N_lag) if n_c is None else 2 * n_c * n_y + 3 * n_c
            flops_total = flops_reservoir + flops_readout
            return (flops_reservoir, flops_readout, flops_total)
    
    # Compute FLOPs
    flops_res_ron_smnist, flops_read_ron_smnist, flops_tot_ron_smnist = flops_ron(
        n_y,
        n_u = 1,
        K = 784,
        n_c = 10
    )

    flops_res_ron_adiac, flops_read_ron_adiac, flops_tot_ron_adiac = flops_ron(
        n_y,
        n_u = 1,
        K = 176,
        n_c = 37
    )

    flops_res_ron_mg, flops_read_ron_mg, flops_tot_ron_mg = flops_ron(
        n_y,
        n_u = 1,
        K = 2000,
        N_lag = 84
    )

    flops_res_ron_lorenz, flops_read_ron_lorenz, flops_tot_ron_lorenz = flops_ron(
        n_y,
        n_u = 5,
        K = 2000,
        N_lag = 25
    )

    flops_res_phy_smnist, flops_read_phy_smnist, flops_tot_phy_smnist = flops_physical(
        n_y,
        n_u = 1,
        K = 784,
        n_c = 10
    )

    flops_res_phy_adiac, flops_read_phy_adiac, flops_tot_phy_adiac = flops_physical(
        n_y,
        n_u = 1,
        K = 176,
        n_c = 37
    )

    flops_res_phy_mg, flops_read_phy_mg, flops_tot_phy_mg = flops_physical(
        n_y,
        n_u = 1,
        K = 2000,
        N_lag = 84
    )

    flops_res_phy_lorenz, flops_read_phy_lorenz, flops_tot_phy_lorenz = flops_physical(
        n_y,
        n_u = 5,
        K = 2000,
        N_lag = 25
    )

    flops_res_phy_simp_smnist, flops_read_phy_simp_smnist, flops_tot_phy_simp_smnist = flops_physical_simpler(
        n_y,
        n_u = 1,
        K = 784,
        n_c = 10
    )
    
    flops_res_phy_simp_adiac, flops_read_phy_simp_adiac, flops_tot_phy_simp_adiac = flops_physical_simpler(
        n_y,
        n_u = 1,
        K = 176,
        n_c = 37
    )
    
    flops_res_phy_simp_mg, flops_read_phy_simp_mg, flops_tot_phy_simp_mg = flops_physical_simpler(
        n_y,
        n_u = 1,
        K = 2000,
        N_lag = 84
    )
    
    flops_res_phy_simp_lorenz, flops_read_phy_simp_lorenz, flops_tot_phy_simp_lorenz = flops_physical_simpler(
        n_y,
        n_u = 5,
        K = 2000,
        N_lag = 25
    )

    # Plot stuff (with full-scale MLP controllers)
    plt.rcParams.update({
        'font.family':        'serif',
        'font.serif':         ['Computer Modern Roman', 'DejaVu Serif'],
        'mathtext.fontset':   'cm',
    })

    plt.figure()
    plt.plot(n_y, flops_res_phy_smnist, 'b', label=r'phy. res. (sMNIST)', linewidth=2.5)
    plt.plot(n_y, flops_res_phy_adiac, 'r', label=r'phy. res. (ADIAC)', linewidth=2.5)
    plt.plot(n_y, flops_res_phy_mg, 'm', label=r'phy. res. (Mackey-Glass)', linewidth=2.5)
    plt.plot(n_y, flops_res_phy_lorenz, 'g', label=r'phy. res. (Lorenz96)', linewidth=1.2)
    plt.plot(n_y, flops_res_ron_smnist, 'b--', label=r'RON (sMNIST)', linewidth=2.5)
    plt.plot(n_y, flops_res_ron_adiac, 'r--', label=r'RON (ADIAC)', linewidth=2.5)
    plt.plot(n_y, flops_res_ron_mg, 'm--', label=r'RON (Mackey-Glass)', linewidth=2.5)
    plt.plot(n_y, flops_res_ron_lorenz, 'g--', label=r'RON (Lorenz96)', linewidth=2.5)
    # ax = plt.gca()
    # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.yscale('log')
    plt.xlabel(r'$n_y$', fontsize=24)
    plt.ylabel(r'FLOPs', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.ylim(1e2, 1e10)
    plt.title(r'FLOPs count (reservoir)', fontsize=22)
    plt.grid(True)
    #plt.legend(ncol=1, fontsize=14)
    plt.tight_layout()
    #plt.show()

    plt.figure()
    plt.plot(n_y, flops_read_phy_smnist, 'b', label=r'phy. res. (sMNIST)', linewidth=1.5)
    plt.plot(n_y, flops_read_phy_adiac, 'r', label=r'phy. res. (ADIAC)', linewidth=1.5)
    plt.plot(n_y, flops_read_phy_mg, 'm', label=r'phy. res. (Mackey-Glass)', linewidth=1.5)
    plt.plot(n_y, flops_read_phy_lorenz, 'g', label=r'phy. res. (Lorenz96)', linewidth=1.5)
    plt.plot(n_y, flops_read_ron_smnist, 'b--', label=r'RON (sMNIST)', linewidth=2.5)
    plt.plot(n_y, flops_read_ron_adiac, 'r--', label=r'RON (ADIAC)', linewidth=2.5)
    plt.plot(n_y, flops_read_ron_mg, 'm--', label=r'RON (Mackey-Glass)', linewidth=2.5)
    plt.plot(n_y, flops_read_ron_lorenz, 'g--', label=r'RON (Lorenz96)', linewidth=2.5)
    # ax = plt.gca()
    # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.yscale('log')
    plt.xlabel(r'$n_y$', fontsize=24)
    plt.ylabel(r'FLOPs', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.ylim(1e2, 10e9)
    plt.title(r'FLOPs count (readout layer)', fontsize=22)
    plt.grid(True)
    #plt.legend(ncol=2, fontsize=14, loc='lower right')
    plt.tight_layout()
    #plt.show()

    plt.figure()
    plt.plot(n_y, flops_tot_phy_smnist, 'b', label=r'phy. res. (sMNIST)', linewidth=2.5)
    plt.plot(n_y, flops_tot_phy_adiac, 'r', label=r'phy. res. (ADIAC)', linewidth=2.5)
    plt.plot(n_y, flops_tot_phy_mg, 'm', label=r'phy. res. (Mackey-Glass)', linewidth=2.5)
    plt.plot(n_y, flops_tot_phy_lorenz, 'g', label=r'phy. res. (Lorenz96)', linewidth=1.2)
    plt.plot(n_y, flops_tot_ron_smnist, 'b--', label=r'RON (sMNIST)', linewidth=2.5)
    plt.plot(n_y, flops_tot_ron_adiac, 'r--', label=r'RON (ADIAC)', linewidth=2.5)
    plt.plot(n_y, flops_tot_ron_mg, 'm--', label=r'RON (Mackey-Glass)', linewidth=2.5)
    plt.plot(n_y, flops_tot_ron_lorenz, 'g--', label=r'RON (Lorenz96)', linewidth=2.5)
    # ax = plt.gca()
    # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.yscale('log')
    plt.xlabel(r'$n_y$', fontsize=24)
    plt.ylabel(r'FLOPs', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.ylim(1e2, 1e10)
    plt.title(r'FLOPs count (total)', fontsize=22)
    plt.grid(True)
    #plt.legend(ncol=1, fontsize=16)
    plt.tight_layout()
    plt.show()

    # Plot stuff (with simpler controllers)    
    plt.figure()
    plt.plot(n_y, flops_res_phy_simp_smnist, 'b', label=r'phy. res. (sMNIST)', linewidth=2.5)
    plt.plot(n_y, flops_res_phy_simp_adiac, 'r', label=r'phy. res. (ADIAC)', linewidth=2.5)
    plt.plot(n_y, flops_res_phy_simp_mg, 'm', label=r'phy. res. (Mackey-Glass)', linewidth=2.5)
    plt.plot(n_y, flops_res_phy_simp_lorenz, 'g', label=r'phy. res. (Lorenz96)', linewidth=1.2)
    plt.plot(n_y, flops_res_ron_smnist, 'b--', label=r'RON (sMNIST)', linewidth=2.5)
    plt.plot(n_y, flops_res_ron_adiac, 'r--', label=r'RON (ADIAC)', linewidth=2.5)
    plt.plot(n_y, flops_res_ron_mg, 'm--', label=r'RON (Mackey-Glass)', linewidth=2.5)
    plt.plot(n_y, flops_res_ron_lorenz, 'g--', label=r'RON (Lorenz96)', linewidth=2.5)
    # ax = plt.gca()
    # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.yscale('log')
    plt.xlabel(r'$n_y$', fontsize=24)
    plt.ylabel(r'FLOPs', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.ylim(1e2, 1e10)
    plt.title(r'FLOPs count (reservoir)', fontsize=22)
    plt.grid(True)
    #plt.legend(ncol=1, fontsize=14)
    plt.tight_layout()
    #plt.show()
    
    plt.figure()
    plt.plot(n_y, flops_read_phy_simp_smnist, 'b', label=r'phy. res. (sMNIST)', linewidth=1.5)
    plt.plot(n_y, flops_read_phy_simp_adiac, 'r', label=r'phy. res. (ADIAC)', linewidth=1.5)
    plt.plot(n_y, flops_read_phy_simp_mg, 'm', label=r'phy. res. (Mackey-Glass)', linewidth=1.5)
    plt.plot(n_y, flops_read_phy_simp_lorenz, 'g', label=r'phy. res. (Lorenz96)', linewidth=1.5)
    plt.plot(n_y, flops_read_ron_smnist, 'b--', label=r'RON (sMNIST)', linewidth=2.5)
    plt.plot(n_y, flops_read_ron_adiac, 'r--', label=r'RON (ADIAC)', linewidth=2.5)
    plt.plot(n_y, flops_read_ron_mg, 'm--', label=r'RON (Mackey-Glass)', linewidth=2.5)
    plt.plot(n_y, flops_read_ron_lorenz, 'g--', label=r'RON (Lorenz96)', linewidth=2.5)
    # ax = plt.gca()
    # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.yscale('log')
    plt.xlabel(r'$n_y$', fontsize=24)
    plt.ylabel(r'FLOPs', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.ylim(1e2, 10e9)
    plt.title(r'FLOPs count (readout layer)', fontsize=22)
    plt.grid(True)
    #plt.legend(ncol=2, fontsize=14, loc='lower right')
    plt.tight_layout()
    #plt.show()
    
    plt.figure()
    plt.plot(n_y, flops_tot_phy_simp_smnist, 'b', label=r'phy. res. (sMNIST)', linewidth=2.5)
    plt.plot(n_y, flops_tot_phy_simp_adiac, 'r', label=r'phy. res. (ADIAC)', linewidth=2.5)
    plt.plot(n_y, flops_tot_phy_simp_mg, 'm', label=r'phy. res. (Mackey-Glass)', linewidth=2.5)
    plt.plot(n_y, flops_tot_phy_simp_lorenz, 'g', label=r'phy. res. (Lorenz96)', linewidth=1.2)
    plt.plot(n_y, flops_tot_ron_smnist, 'b--', label=r'RON (sMNIST)', linewidth=2.5)
    plt.plot(n_y, flops_tot_ron_adiac, 'r--', label=r'RON (ADIAC)', linewidth=2.5)
    plt.plot(n_y, flops_tot_ron_mg, 'm--', label=r'RON (Mackey-Glass)', linewidth=2.5)
    plt.plot(n_y, flops_tot_ron_lorenz, 'g--', label=r'RON (Lorenz96)', linewidth=2.5)
    # ax = plt.gca()
    # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.yscale('log')
    plt.xlabel(r'$n_y$', fontsize=24)
    plt.ylabel(r'FLOPs', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.ylim(1e2, 1e10)
    plt.title(r'FLOPs count (total)', fontsize=22)
    plt.grid(True)
    #plt.legend(ncol=1, fontsize=16)
    plt.tight_layout()
    plt.show()


# Noise, uncertainties, disturbance experiments
if False:
    ny = np.arange(6, 16, 3)

    # sMNIST
    acc_ideal = np.array([64.03, 66.78, 66.18, 71.30])
    acc_noise = np.array([57.7, 67.26, 64.32, 0])
    
    stddev_ideal = np.array([1.32, 1.82, 2.41, 0.63])
    stddev_noise = np.array([0.76, 4.16, 0.4, 0])
    
    fig = plt.figure(figsize=(4.1, 3))

    plt.plot(ny, acc_ideal, label='Ideal', color="#10C200FF", linestyle='-', linewidth='1.8', marker='o', markersize='8') # ideal
    plt.fill_between(ny, acc_ideal-stddev_ideal, acc_ideal+stddev_ideal, color ="#10C200FF", alpha=0.15)
    
    plt.plot(ny, acc_noise, label='Noise + disturbance + uncertainties', color="#FF4949", linestyle='-', linewidth='1.8', marker='s', markersize='8') # noise, disturbance, uncertainties
    plt.fill_between(ny, acc_noise-stddev_noise, acc_noise+stddev_noise, color ="#FF4949", alpha=0.15)
    
    plt.grid(True, alpha=0.6)
    plt.xlabel(r'$n_y$', fontsize=13)
    plt.ylabel(r'accuracy [%]', fontsize=13)
    plt.title(r'sMNIST ($\uparrow$)', fontsize=14)
    plt.xticks(np.arange(6, 16), fontsize=13)
    plt.yticks(fontsize=13)
    #plt.yticks(np.arange(30, 81, 10))
    plt.xlim(5.5, 15.5)
    plt.ylim(0.0, 80)
    #plt.legend(ncol=1, fontsize=9, loc='best')
    
    fig.tight_layout()
    #fig.savefig("smnist_scalability.pdf", bbox_inches="tight")
    #plt.show()

    # ADIAC    
    acc_ideal = np.array([44.58, 51.01, 50.18, 53.10])
    acc_noise = np.array([43.99, 53.39, 45.36, 50.96])

    stddev_ideal = np.array([1.03, 3.75, 0.68, 2.45])
    stddev_noise = np.array([1.68, 2.15, 2.65, 2.28])
    
    fig = plt.figure(figsize=(4.1, 3))
    
    plt.plot(ny, acc_ideal, label='Ideal', color="#10C200FF", linestyle='-', linewidth='1.8', marker='o', markersize='8') # ideal
    plt.fill_between(ny, acc_ideal-stddev_ideal, acc_ideal+stddev_ideal, color ="#10C200FF", alpha=0.15)
    
    plt.plot(ny, acc_noise, label='Noise + disturbance + uncertainties', color="#FF4949", linestyle='-', linewidth='1.8', marker='s', markersize='8') # noise, disturbance, uncertainties
    plt.fill_between(ny, acc_noise-stddev_noise, acc_noise+stddev_noise, color ="#FF4949", alpha=0.15)
    
    plt.grid(True, alpha=0.6)
    plt.xlabel(r'$n_y$', fontsize=13)
    plt.ylabel(r'accuracy [%]', fontsize=13)
    plt.title(r'ADIAC ($\uparrow$)', fontsize=14)
    plt.xticks(np.arange(6, 16), fontsize=13)
    plt.yticks(fontsize=13)
    #plt.yticks(np.arange(30, 81, 10))
    plt.xlim(5.5, 15.5)
    plt.ylim(0.0, 60)
    #plt.legend(ncol=1, fontsize=9, loc='best')
    
    fig.tight_layout()
    #fig.savefig("smnist_scalability.pdf", bbox_inches="tight")
    #plt.show()

    # Mackey-Glass    
    acc_ideal = np.array([0.525, 0.505, 0.435, 0.424])
    acc_noise = np.array([0.549, 0.530, 0.499, 0.460])
    
    stddev_ideal = np.array([0.003, 0.016, 0.014, 0.006])
    stddev_noise = np.array([0.004, 0.026, 0.014, 0.009])
    
    fig = plt.figure(figsize=(4.1, 3))
    
    plt.plot(ny, acc_ideal, label='Ideal', color="#10C200FF", linestyle='-', linewidth='1.8', marker='o', markersize='8') # ideal
    plt.fill_between(ny, acc_ideal-stddev_ideal, acc_ideal+stddev_ideal, color ="#10C200FF", alpha=0.15)
    
    plt.plot(ny, acc_noise, label='Noise + disturbance + uncertainties', color="#FF4949", linestyle='-', linewidth='1.8', marker='s', markersize='8') # noise, disturbance, uncertainties
    plt.fill_between(ny, acc_noise-stddev_noise, acc_noise+stddev_noise, color ="#FF4949", alpha=0.15)
    
    plt.grid(True, alpha=0.6)
    plt.xlabel(r'$n_y$', fontsize=13)
    plt.ylabel(r'NRMSE [–]', fontsize=13)
    plt.title(r'Mackey-Glass ($\downarrow$)', fontsize=14)
    plt.xticks(np.arange(6, 16), fontsize=13)
    plt.yticks(fontsize=13)
    #plt.yticks(np.arange(0.2, 0.9, 0.1))
    plt.xlim(5.5, 15.5)
    plt.ylim(0.0, 0.7)
    #plt.legend(ncol=1, fontsize=9, loc='best')
    
    fig.tight_layout()
    #fig.savefig("mg_scalability.pdf", bbox_inches="tight")
    #plt.show()

    # Lorenz96
    acc_ideal = np.array([0.567, 0.522, 0.463, 0.455])
    acc_noise = np.array([0.575, 0.564, 0.502, 0.556])

    stddev_ideal = np.array([0.002, 0.001, 0.011, 0.013])
    stddev_untrained = np.array([0.004, 0.018, 0.003, 0.013])
    
    # Plot
    fig = plt.figure(figsize=(4.1, 3))
    
    plt.plot(ny, acc_ideal, label='Ideal', color="#10C200FF", linestyle='-', linewidth='1.8', marker='o', markersize='8') # ideal
    plt.fill_between(ny, acc_ideal-stddev_ideal, acc_ideal+stddev_ideal, color ="#10C200FF", alpha=0.15)
    
    plt.plot(ny, acc_noise, label='Noise + disturbance + uncertainties', color="#FF4949", linestyle='-', linewidth='1.8', marker='s', markersize='8') # noise, disturbance, uncertainties
    plt.fill_between(ny, acc_noise-stddev_noise, acc_noise+stddev_noise, color ="#FF4949", alpha=0.15)
    
    plt.grid(True, alpha=0.6)
    plt.xlabel(r'$n_y$', fontsize=13)
    plt.ylabel(r'NRMSE [–]', fontsize=13)
    plt.title(r'Lorenz96 ($\downarrow$)', fontsize=14)
    plt.xticks(np.arange(6, 16), fontsize=13)
    plt.yticks(fontsize=13)
    #plt.yticks(np.arange(0.2, 0.9, 0.1))
    plt.xlim(5.5, 15.5)
    plt.ylim(0.0, 0.7)
    #plt.legend(ncol=1, fontsize=9, loc='best')
    
    fig.tight_layout()
    #fig.savefig("mg_scalability.pdf", bbox_inches="tight")
    #plt.show()
    
    # Legend in a separate figure
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    
    fig_leg = plt.figure(figsize=(9, 0.5))
    fig_leg.legend(handles, labels, loc='center', ncol=4, frameon=True, fontsize=9)
    plt.show()


# Computational cost of pretraining
if False:
    # Data
    ny = np.arange(6, 16, 3)

    data = {
        "sMNIST": {
            "Normal pretraining": {
                "mean": [5615, 5534, 8428, 7295],
                "std":  [413, 336, 892, 467],
            },
            "Fixed morphology": {
                "mean": [5055, 4435, 8166, 6693],
                "std":  [263, 1548, 872, 551],
            },
            "Simpler controller": {
                "mean": [6001, 5729, 9803, 8043],
                "std":  [38, 356, 65, 66],
            },
        },
        "ADIAC": {
            "Normal pretraining": {
                "mean": [5920, 5440, 8673, 7430],
                "std":  [46, 93, 459, 206],
            },
            "Fixed morphology": {
                "mean": [5210, 4627, 8398, 6766],
                "std":  [25, 28, 320, 286],
            },
            "Simpler controller": {
                "mean": [np.nan] * 4,
                "std":  [np.nan] * 4,
            },
        },
        "Mackey-Glass": {
            "Normal pretraining": {
                "mean": [5485, 5067, 10356, 6782],
                "std":  [258, 472, 1741, 1507],
            },
            "Fixed morphology": {
                "mean": [5437, 4463, 9111, 4856],
                "std":  [518, 437, 482, 1439],
            },
            "Simpler controller": {
                "mean": [5994, 5926, 5208, 8008],
                "std":  [45, 31, 10, 60],
            },
        },
        "Lorenz96": {
            "Normal pretraining": {
                "mean": [5673, 6217, 7640, 6512],
                "std":  [67, 1179, 136, 966],
            },
            "Fixed morphology": {
                "mean": [4491, 4924, 7554, 6761],
                "std":  [892, 165, 70, 856],
            },
            "Simpler controller": {
                "mean": [np.nan] * 4,
                "std":  [np.nan] * 4,
            },
        },
    }

    # Plots configurations
    styles = {
        "Normal pretraining": dict(color="#10C200FF", marker='o'),
        "Fixed morphology":   dict(color="#FF4949",   marker='s'),
        "Simpler controller": dict(color="#1F77B4",   marker='^'),
    }

    ylims = {
        "sMNIST":       (3000, 11000),
        "ADIAC":        (3000, 11000),
        "Mackey-Glass": (3000, 12500),
        "Lorenz96":     (3000, 9000),
    }

    ylabel = r'Elapsed time [min]'

    # Plots
    for task, curves in data.items():
        fig = plt.figure(figsize=(4.1, 3))

        for label, vals in curves.items():
            mean = np.array(vals["mean"], dtype=float)
            std = np.array(vals["std"], dtype=float)
            st = styles[label]

            plt.plot(ny, mean / 60, label=label, color=st["color"], linestyle='-',
                    linewidth=1.8, marker=st["marker"], markersize=8)
            plt.fill_between(ny, (mean - std)/60, (mean + std)/60, color=st["color"], alpha=0.15)

        plt.grid(True, alpha=0.6)
        plt.xlabel(r'$n_y$', fontsize=13)
        plt.ylabel(ylabel, fontsize=13)
        plt.title(task, fontsize=14)
        plt.xticks(np.arange(6, 16), fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlim(5.5, 15.5)
        #plt.ylim(*ylims[task])

        fig.tight_layout()

    # Legend in a separate plot
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()

    fig_leg = plt.figure(figsize=(9, 0.5))
    fig_leg.legend(handles, labels, loc='center', ncol=3, frameon=True, fontsize=9)
    plt.show()


# ESN baseline
if False:
    # Data
    ny = np.arange(6, 16, 3)

    data = {
        "sMNIST": {
            "RON": {
                "mean": [65.88, 68.66, 70.51, 78.57],
                "std":  [0] * 4,
            },
            "ESN": {
                "mean": [58.52, 63.37, 64.89, 67.30],
                "std":  [1.73, 1.24, 1.13, 1.21],
            },
            "Pretrained (ours)": {
                "mean": [64.03, 66.78, 66.18, 71.30],
                "std":  [1.32, 1.82, 2.41, 0.63],
            },
        },
        "ADIAC": {
            "RON": {
                "mean": [46.80, 53.20, 57.03, 62.15],
                "std":  [0] * 4,
            },
            "ESN": {
                "mean": [46.55, 51.51, 54.01, 57.95],
                "std":  [2.39, 3.47, 3.85, 2.20],
            },
            "Pretrained (ours)": {
                "mean": [44.58, 51.01, 50.18, 53.10],
                "std":  [1.03, 3.75, 0.68, 2.45],
            },
        },
        "Mackey-Glass": {
            "RON": {
                "mean": [0.513, 0.432, 0.358, 0.309],
                "std":  [0] * 4,
            },
            "ESN": {
                "mean": [0.520, 0.483, 0.435, 0.388],
                "std":  [0.006, 0.056, 0.048, 0.050],
            },
            "Pretrained (ours)": {
                "mean": [0.525, 0.505, 0.435, 0.425],
                "std":  [0.003, 0.016, 0.014, 0.006],
            },
        },
        "Lorenz96": {
            "RON": {
                "mean": [0.564, 0.515, 0.417, 0.385],
                "std":  [0] * 4,
            },
            "ESN": {
                "mean": [0.602, 0.529, 0.491, 0.447],
                "std":  [0.023, 0.020, 0.023, 0.015],
            },
            "Pretrained (ours)": {
                "mean": [0.567, 0.522, 0.463, 0.455],
                "std":  [0.002, 0.001, 0.011, 0.013],
            },
        },
    }

    # Plots configurations
    styles = {
        "RON": dict(color="#7E7E7E", marker='^', linestyle='--'),
        "ESN": dict(color="#FF00F2",   marker='*', linestyle='--'),
        "Pretrained (ours)":   dict(color="#10C200FF",   marker='o', linestyle='-'),
    }

    # ylims = {
    #     "sMNIST":       (3000, 11000),
    #     "ADIAC":        (3000, 11000),
    #     "Mackey-Glass": (3000, 12500),
    #     "Lorenz96":     (3000, 9000),
    # }

    ylabels = {
        "sMNIST":       'Accuracy [%]',
        "ADIAC":        'Accuracy [%]',
        "Mackey-Glass": 'NRMSE [-]',
        "Lorenz96":     'NRMSE [-]',
    }

    # Plots
    for task, curves in data.items():
        fig = plt.figure(figsize=(4.1, 3))

        for label, vals in curves.items():
            mean = np.array(vals["mean"], dtype=float)
            std = np.array(vals["std"], dtype=float)
            st = styles[label]

            plt.plot(ny, mean, label=label, color=st["color"], linestyle=st["linestyle"],
                    linewidth=1.8, marker=st["marker"], markersize=8)
            plt.fill_between(ny, (mean - std), (mean + std), color=st["color"], alpha=0.15)
            plt.ylabel(ylabels[task], fontsize=13)

        plt.grid(True, alpha=0.6)
        plt.xlabel(r'$n_y$', fontsize=13)
        plt.title(task, fontsize=14)
        plt.xticks(np.arange(6, 16), fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlim(5.5, 15.5)
        #plt.ylim(*ylims[task])

        fig.tight_layout()

    # Legend in a separate plot
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()

    fig_leg = plt.figure(figsize=(9, 0.5))
    fig_leg.legend(handles, labels, loc='center', ncol=3, frameon=True, fontsize=9)
    plt.show()


# Correlation analysis between dynamics-matching pretraining and downstream PRC performance
if True:
    # Consider various scenarios, ny, tasks, etc
    tasks = ["sMNIST", "Adiac", "MG", "lorenz"]
    nys = [6, 9, 12, 15] # reservoir dim
    pretraining_scenarios = ["default", "no_robots"] # "default" = full pretraining, "no_robots" = partial pretraining
    downstream_scenarios = ["a", "b"] # "a" = full pretraining, "b" = partial pretraining
    runs = [1, 2, 3, 4] # different random seeds
    stages = ["pretraining", "downstream"]

    # Read and collect all data in a dict with keys (task, ny, scenario, run)
    data = {}
    for task in tasks:
        for ny in nys:
            for pre_scenario, down_scenario in zip(pretraining_scenarios, downstream_scenarios):
                for run in runs:
                    # find paths
                    if Path(data_folder/'more_soft_robots_optimization'/task/f'N{ny}'/f'{pre_scenario}_run{run}'/'metrics.txt').exists():
                        path_pre = Path(data_folder/'more_soft_robots_optimization'/task/f'N{ny}'/f'{pre_scenario}_run{run}'/'metrics.txt')
                    else:
                        path_pre = Path(data_folder/'more_soft_robots_optimization'/task/f'N{ny}'/f'{pre_scenario}_run{run}'/'metrics_NoOptimization.txt')

                    if Path(data_folder/'more_soft_robots_reservoir'/task/f'N{ny}'/f'{down_scenario}_run{run}'/'performances.txt').exists():
                        path_down = Path(data_folder/'more_soft_robots_reservoir'/task/f'N{ny}'/f'{down_scenario}_run{run}'/'performances.txt')
                    else:
                        path_down = Path(data_folder/'more_soft_robots_reservoir'/task/f'N{ny}'/f'{down_scenario}_run{run}'/'note.txt')

                    # collect metrics for that case
                    metrics_pre = extract_pretraining_metrics(path_pre)
                    metrics_down = extract_downstream_metrics(path_down, task)

                    # create dict with data
                    data[(task, ny, pre_scenario, run)] = {
                        "pretraining": metrics_pre.get("test_rmse", np.nan) if metrics_pre.get("test_rmse", np.nan)<1e2 else np.nan,
                        "downstream": metrics_down.get("performance", np.nan) if metrics_down.get("performance", np.nan)<1e2 else np.nan,
                    }

    # Normalize metrics: compute the acceleration target RMS in the test pretraining dataset
    rms_dataset = {} # keys (task, ny)
    for task in tasks:
        for ny in nys:
            dataset = np.load(dataset_folder/'soft robot optimization'/f'{task}_RON_N{ny}/dataset_m1e5_N{ny}.npz')
            ydd = dataset["ydd"] # accelerations of the RON oscillators. Shape (m, ny)
            rms_dataset[(task, ny)] = np.sqrt(np.mean(np.sum(ydd ** 2, axis=1)))

    # Normalize metrics: insert teacher RON downstream performance
    ron = {
        "sMNIST": dict(zip(nys, [0.6588, 0.6866, 0.7051, 0.7857])),
        "Adiac": dict(zip(nys, [0.4680, 0.5320, 0.5703, 0.6215])),
        "MG": dict(zip(nys, [0.513, 0.432, 0.358, 0.309])),
        "lorenz": dict(zip(nys, [0.564, 0.515, 0.417, 0.385])),
    }
    higher_is_better = {"sMNIST": True, "Adiac": True, "MG": False, "lorenz": False}

    # Normalize metrics
    for (task, ny, scenario, run), v in data.items():
        down_perf, ron_perf = v["downstream"], ron[task][ny]
        v["downstream_rel"] = down_perf / ron_perf if higher_is_better[task] else ron_perf / down_perf # relative downstream performance
        v["log_norm_pretraining"] = np.log(v["pretraining"] / rms_dataset[(task, ny)]) # log normalized pretraining test RMSE
    # Now data is a dict with keys (task, ny, scenario, run) and values that are dict with keys "pretraining", "downstream", "log_norm_pretraining" and "downstream_rel"



    def get_points(task):
        """Punti validi (coppie senza NaN) per un task."""
        pts = []
        for (t, ny, sc, run), v in data.items():
            if t != task:
                continue
            x, y = v["log_norm_pretraining"], v["downstream_rel"]
            if np.isnan(x) or np.isnan(y):
                continue
            pts.append({"ny": ny, "sc": sc, "run": run, "x": x, "y": y})
        return pts

    def boot_ci(x, y, n=5000, seed=0):
        rng = np.random.default_rng(seed)
        idx = np.arange(len(x))
        vals = []
        for _ in range(n):
            s = rng.choice(idx, len(idx))
            vals.append(spearmanr(x[s], y[s])[0])
        return np.nanpercentile(vals, [2.5, 97.5])

    def centered_ranks(a):
        r = rankdata(a)
        return r - r.mean()

    def analyze(task):
        pts = get_points(task)
        x = np.array([p["x"] for p in pts])
        y = np.array([p["y"] for p in pts])

        # 1) pooled
        rho = spearmanr(x, y)[0]
        lo, hi = boot_ci(x, y)
        print(f"{task} pooled:        rho={rho:+.2f}  CI95=[{lo:+.2f}, {hi:+.2f}]  n={len(x)}")

        # 2) controllato per ny: rank centrati dentro ogni cella ny
        xe, ye = [], []
        for ny in nys:
            cell = [p for p in pts if p["ny"] == ny]
            xe.append(centered_ranks([p["x"] for p in cell]))
            ye.append(centered_ranks([p["y"] for p in cell]))
        xe, ye = np.concatenate(xe), np.concatenate(ye)
        print(f"{task} within-ny:     rho={spearmanr(xe, ye)[0]:+.2f}  n={len(xe)}")

        # 3) dentro ogni scenario
        for sc in pretraining_scenarios:
            cell = [p for p in pts if p["sc"] == sc]
            xs = np.array([p["x"] for p in cell])
            ys = np.array([p["y"] for p in cell])
            print(f"{task} [{sc:9s}]: rho={spearmanr(xs, ys)[0]:+.2f}  n={len(cell)}")

    #analyze("sMNIST")

    def within_cell(task):
        pts = get_points(task)
        xe, ye = [], []
        for ny in nys:
            for sc in pretraining_scenarios:
                cell = [p for p in pts if p["ny"] == ny and p["sc"] == sc]
                if len(cell) < 3:   # celle con troppi NaN: salta
                    continue
                xe.append(centered_ranks([p["x"] for p in cell]))
                ye.append(centered_ranks([p["y"] for p in cell]))
        xe, ye = np.concatenate(xe), np.concatenate(ye)
        rho = spearmanr(xe, ye)[0]
        print(f"{task} within-(ny,scenario): rho={rho:+.2f}  n={len(xe)}")

    within_cell("sMNIST")