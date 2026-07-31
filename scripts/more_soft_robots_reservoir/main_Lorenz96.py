# =====================================================
# Setup
# =====================================================

# Choose device (cpu or gpu)
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Imports
import numpy as onp
import jax
import jax.numpy as jnp
from sklearn import preprocessing
from sklearn.linear_model import Ridge
import joblib
from tqdm import tqdm

import matplotlib.pyplot as plt

from pathlib import Path
import sys
import time

from soromox.systems.my_systems import PlanarPCS_simple

# Folders
curr_folder = Path(__file__).parent # current folder
sys.path.append(str(curr_folder.parent)) # scripts folder
main_folder = curr_folder.parent.parent # main folder "codes"
saved_data_folder = main_folder/'saved data' # folder with saved data (trained architectures)
from utilis import *

# Jax settings
jax.config.update("jax_enable_x64", True) # double precision
jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)


"""
This script:
    1.  Takes a certain optimized reservoir's architecture (robots + map + controller) specified by the user in `load_model_path`
        (! map type and controller type must be specified by hand in `map_type` and `fb_controller_type`, `ff_controller_type`).

    2.  Loads the Lorenz96 dataset.

    3a. If `train` is True, output layer (scaler + predictor) is trained with the given reservoir. Trained scaler and 
        predictor are then saved in data folder named `experiment_name`. 
    3b. If `train` is False, loads a given output layer (scaler + predictor) from data folder named `experiment_name`.

    4.  The full architecture (reservoir + output layer) is tested on a test sequence.

    5.  Saves plots and metrics on the test sequence (and train sequence if training was performed) in `experiment_name` data and plots 
        folders. Also saves the dynamics of the reservoir during inference.
"""
# =====================================================
# Run for different random seeds
# =====================================================

seeds = [123, 1234, 12345, 123456]

for run, seed in enumerate(seeds):
    n_run = run + 1
    key = jax.random.key(seed)

    # =====================================================
    # Script settings
    # =====================================================

    # General
    dt_u = 0.05 # time step for the input u
    Nw = 200 # washout steps
    Nl = 25 # prediction lag
    n_input = 5 # input dimension
    batch = 128 # number of trajectories in the datasets

    # Output layer (scaler + predictor)
    experiment_name = f'lorenz/N6/a_run{n_run}' # name of the experiment to save/load
    train = True # if True, perform training (output layer). Otherwise, test saved 'experiment_name' model

    # Reservoir (robots + map + controller)
    load_model_path = saved_data_folder/'more_soft_robots_optimization'/f'lorenz/N6/default_run{n_run}' # choose the reservoir to load (robots + map + controller)
    map_type = 'linear' # 'linear', 'encoder-decoder', 'bijective', 'none'
    controller_type = 'fb+ff' # if 'unique': Tau = Tau_tot(Z,u). If 'fb+ff': Tau = Tau_fb(Z) + Tau_ff(u). If 'ff': Tau = Tau_ff(u) (randomly initialized tanh(V*u+d)) !!! If 'unique', the controller tau_tot is defined in fb_controller_type
    fb_controller_type = 'mlp' # 'linear_simple', 'linear_complete', 'tanh_simple', 'tanh_complete', 'mlp'
    ff_controller_type = 'mlp' # 'linear', 'tanh', 'mlp'
    robots_type = 'saved' # 'saved' (those in 'load_model_path'), 'random' (randomly sampled), 'default'


    # =====================================================
    # Folders
    # =====================================================

    plots_folder = main_folder/'plots and videos'/curr_folder.stem/experiment_name # folder for plots and videos
    data_folder = main_folder/'saved data'/curr_folder.stem/experiment_name # folder for saving data

    data_folder.mkdir(parents=True, exist_ok=True)
    plots_folder.mkdir(parents=True, exist_ok=True)


    # =====================================================
    # Datasets
    # =====================================================

    # Load Lorenz96 dataset
    train_dataset = get_lorenz(dim=5, F=8, num_batch=batch, lag=Nl, washout=Nw, seed=0) # batch of sequences from k=0 to k=N-1. Shape (B, N, n_input)
    test_dataset = get_lorenz(dim=5, F=8, num_batch=batch, lag=Nl, washout=Nw, seed=1) # batch of sequences from k=0 to k=N-1. Shape (B, N, n_input)

    # Convert to double precision jax
    train_dataset = jnp.array(train_dataset, dtype=jnp.float64)
    test_dataset = jnp.array(test_dataset, dtype=jnp.float64)

    # Extract datapoints and labels from datasets
    N_train = train_dataset.shape[1] # number of total steps for the train set sequence
    N_test = test_dataset.shape[1] # number of total steps for the test set sequence
    Np_train = train_dataset.shape[1] - Nl # number of datapoints Np=N-Nl for the train set sequence
    Np_test = test_dataset.shape[1] - Nl # number of datapoints Np=N-Nl for the test set sequence

    train_datapoints = train_dataset[:, :Np_train] # from k=0 to k=N-Nl-1. Shape (B, N-Nl, n_input)
    test_datapoints = test_dataset[:, :Np_test] # from k=0 to k=N-Nl-1. Shape (B, N-Nl, n_input)

    train_target = train_dataset[:, Nw+Nl:].reshape((-1, n_input)) # from k=Nw+Nl to k=N-1. Shape (B*(N-Nl-Nw), n_input)
    test_target = test_dataset[:, Nw+Nl:].reshape((-1, n_input)) # from k=Nw+Nl to k=N-1. Shape (B*(N-Nl-Nw), n_input)


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
        V = scal_input[:,None] * jax.random.uniform(key_V, shape=(3*n_pcs*n_robots, n_input), minval=0.0, maxval=1.0) # random input-to-hidden weights
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
    time_u_train = dt_u * jnp.arange(0, Np_train) # define time vector for the train input sequence
    time_u_test = dt_u * jnp.arange(0, Np_test) # define time vector for the test input sequence
    saveat_train = time_u_train # for saving simulation results
    saveat_test = time_u_test # for saving simulation results

    reservoir_batched = jax.jit(jax.vmap(reservoir, in_axes=(0,None,None,None))) # vmap reservoir's forward pass
    num_minibatches_train = 16 # split batch B into 'num_minibatches' minibatches of 'n_minibatch' elements each
    n_minibatch_train = 8 # split batch B into 'num_minibatches' minibatches of 'n_minibatch' elements each
    num_minibatches_test = 16 # split batch B into 'num_minibatches' minibatches of 'n_minibatch' elements each
    n_minibatch_test = 8 # split batch B into 'num_minibatches' minibatches of 'n_minibatch' elements each


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
        state_reservoir_ts = []
        for i in tqdm(range(num_minibatches_train), 'Model forward'):
            (
                _,
                state_reservoir_ts_i, # reservoir's states evolution from k=0 to k=N-Nl-1. Shape (num_minibatches, N-Nl, 2*n_hid)
                _, # pcs's states evolution from k=0 to k=N-Nl-1. Shape (num_minibatches, N-Nl, 2*n_robots*3*n_pcs)
                _, # pcs actuation. Shape (num_minibatches, N-Nl, 3*n_pcs*n_robots)
                _
            ) = reservoir_batched(train_datapoints[i*n_minibatch_train:(i+1)*n_minibatch_train], time_u_train, saveat_train, dt_sim)
            state_reservoir_ts.append(state_reservoir_ts_i)

        state_reservoir_ts = jnp.concatenate(state_reservoir_ts) # reservoir's states evolution from k=0 to k=N-Nl-1. Shape (B, N-Nl, 2*n_hid)
        y_ts, _ = jnp.split(state_reservoir_ts, 2, axis=2) # reservoir's position evolution from k=0 to k=N-Nl-1. Shape (B, N-Nl, n_hid)
        activations = y_ts[:, Nw:] # remove the initial washout steps. Shape (B, N-Nl-Nw, n_hid). It's the reservoir's states evolution from k=Nw to k=N-Nl-1
        activations.block_until_ready()
        stop = time.perf_counter() 
        elatime_forward_pass_training = stop - start
        print(f'Elapsed time: {elatime_forward_pass_training}')
        activations = onp.array(activations)
        activations = activations.reshape(-1, 3*n_robots*n_pcs) # shape (B, N-Nl-Nw, n_hid) -> (B*(N-Nl-Nw), n_hid)

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
        joblib.dump(scaler, data_folder/'scaler.pkl')
        joblib.dump(predictor, data_folder/'predictor.pkl')

        # Train accuracy
        pred = predictor.predict(activations) # shape (B*(N-Nl-Nw), n_input)
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
        scaler = joblib.load(data_folder/'scaler.pkl')
        predictor = joblib.load(data_folder/'predictor.pkl')

    # Forward on the test set
    print(f'--- Evaluating perfomances (test set) ---')
    start = time.perf_counter()
    state_reservoir_ts, state_pcs_ts, actuation_ts = [], [], []
    for i in tqdm(range(num_minibatches_test), 'Model forward'):
        (
            time_ts,
            state_reservoir_ts_i, # reservoir's states evolution from k=0 to k=N-Nl-1. Shape (num_minibatches, N-Nl, 2*n_hid)
            state_pcs_ts_i, # pcs's states evolution from k=0 to k=N-Nl-1. Shape (num_minibatches, N-Nl, 2*n_robots*3*n_pcs)
            actuation_ts_i, # pcs actuation. Shape (num_minibatches, N-Nl, 3*n_pcs*n_robots)
            _
        ) = reservoir_batched(test_datapoints[i*n_minibatch_test:(i+1)*n_minibatch_test], time_u_test, saveat_test, dt_sim)
        state_reservoir_ts.append(state_reservoir_ts_i)
        state_pcs_ts.append(state_pcs_ts_i)
        actuation_ts.append(actuation_ts_i)

    state_reservoir_ts = jnp.concatenate(state_reservoir_ts) # reservoir's states evolution from k=0 to k=N-Nl-1. Shape (B, N-Nl, 2*n_hid)
    state_pcs_ts = jnp.concatenate(state_pcs_ts) # pcs's states evolution from k=0 to k=N-Nl-1. Shape (B, N-Nl, 2*n_robots*3*n_pcs)
    actuation_ts = jnp.concatenate(actuation_ts) # pcs actuation. Shape (B, N-Nl, 3*n_pcs*n_robots)
    y_ts, _ = jnp.split(state_reservoir_ts, 2, axis=2) # reservoir's position evolution from k=0 to k=N-Nl-1. Shape (B, N-Nl, n_hid)
    activations = y_ts[:, Nw:] # remove the initial washout steps. Shape (B, N-Nl-Nw, n_hid). It's the reservoir's states evolution from k=Nw to k=N-Nl-1
    activations.block_until_ready()
    stop = time.perf_counter() 
    elatime_forward_pass_testing = stop - start
    print(f'Elapsed time: {elatime_forward_pass_testing}')

    # Prediction and test accuracy
    activations = onp.array(activations)
    activations = activations.reshape(-1, 3*n_robots*n_pcs) # shape (B, N-Nl-Nw, n_hid) -> (B*(N-Nl-Nw), n_hid)
    activations = scaler.transform(activations)
    pred = predictor.predict(activations) # shape (B*(N-Nl-Nw), n_input)
    rmse = jnp.sqrt(jnp.mean((pred - test_target) ** 2))
    rms_target = jnp.sqrt(jnp.mean(test_target ** 2))
    test_nrmse = (rmse / rms_target)
    print(f'Test NRMSE: {test_nrmse}')


    # =====================================================
    # Show results of the test (only 1st from the batch)
    # =====================================================

    # Prepare variables
    y_ts, yd_ts = jnp.split(state_reservoir_ts[0], 2, axis=1) # reservoir states
    _, q_ts, _ = jax.vmap(robots_system.transform_Z)(state_pcs_ts[0]) # shape (n_steps, n_robots, 3*n_pcs)
    Q_ts, _ = jnp.split(state_pcs_ts[0], 2, axis=1) # shape (n_steps, 3*n_pcs*n_robots)
    full_time = dt_u * onp.arange(0, Np_test + Nl)
    full_sequence = test_dataset[0] # full test sequence
    prediction = pred.reshape(batch, N_test-Nl-Nw, -1)[0]  # shape (B*(N-Nl-Nw), n_inp) -> (B, N-Nl-Nw, n_inp) -> (N-Nl-Nw, n_inp)
    time_ts = time_ts[0]
    actuation_ts = actuation_ts[0]

    # Show max 15 DOFs in the plots
    if 3*n_pcs*n_robots > 15:
        n_show = 15
    else:
        n_show = 3*n_pcs*n_robots

    n_cols = min(2, n_show)
    n_rows = int(np.ceil(n_show / n_cols))

    # Show predicted sequence
    fig, axs = plt.subplots(n_input, 1, figsize=(12,12))
    for i, ax in enumerate(axs):
        ax.plot(full_time, full_sequence[:, i], 'k--', label='full sequence')
        ax.plot(full_time[Nw:N_test-Nl], full_sequence[Nw:N_test-Nl, i], 'k', label='test sequence')
        ax.plot(full_time[Nw+Nl:], prediction[:, i], 'r', label='predicted sequence')
        ax.grid(True)
        ax.set_xlabel('t [s]') if i==n_input-1 else ax.set_xlabel('')
        ax.set_ylabel(rf'$u_{{{i+1}}}$')
        ax.set_title(f'Component {i+1}')
        ax.legend()
    plt.tight_layout()
    plt.savefig(plots_folder/'Prediction', bbox_inches='tight')
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
    plt.savefig(plots_folder/'Example_inference_evolution', bbox_inches='tight') 
    #plt.show()

    # Show actuation signal Tau(t)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16,13))
    for i, ax in enumerate(axs.flatten()):
        ax.plot(time_ts, actuation_ts[:,i], 'r', label=r'robots actuation $\tau(t)$')
        ax.set_xlabel('t [s]')
        ax.set_ylabel(r'$\tau$')
        ax.grid(True)
        ax.set_title(f'Component {i+1}')
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(plots_folder/'Example_inference_actuation', bbox_inches='tight') 
    #plt.show()

    # Show robot animation
    for n in range(n_robots):
        animate_robot_matplotlib(
            robot = robots_system.get_robot(n),
            t_list = time_ts,
            q_list = q_ts[:,n],
            interval = 1e-3, 
            slider = False,
            animation = True,
            show = False,
            duration = 10,
            fps = 30,
            save_path = plots_folder/f'Example_inference_animation_robot_{n+1}.gif',
        )


    # =========================================================
    # Save text file with performances and data
    # =========================================================

    if not train:
        Np_train = '(training was not performed)'
        train_nrmse = '(training was not performed)'
        elatime_forward_pass_training = '(training was not performed)'
        elatime_train_output_layer = '(training was not performed)'

    with open(data_folder/'performances.txt', 'w') as file:
        file.write(f"SETUP\n")
        file.write(f"   Train set size (n. of datapoints/labels): {Np_train}\n")
        file.write(f"   Test set size (n. of datapoints/labels):  {Np_test}\n")
        file.write(f"   Prediction lag:                           {Nl}\n")
        file.write(f"   Washout steps:                            {Nw}\n\n")
        file.write(f"RESERVOIR PROPERTIES\n")
        file.write(f"   Model path:  {load_model_path}\n")
        file.write(f"   Dimension:   {3*n_pcs*n_robots}\n")
        file.write(f"   n. robots:   {n_robots}\n")
        file.write(f"   n. segments: {n_pcs}\n")
        if robots_type == 'default':
            file.write(f"   Robots:      default robots were used\n")
        elif robots_type == 'random':
            file.write(f"   Robots:      randomly generated robots were used\n")
        else:
            file.write(f"   Robots:      those in {load_model_path}\n")
        file.write(f"   Map:         {map_type}\n")
        if controller_type == 'unique':
            file.write(f"   Controller:  {fb_controller_type} (unique)\n\n")
        elif controller_type == 'fb+ff':
            file.write(f"   Controller:  {fb_controller_type} (fb) + {ff_controller_type} (ff)\n\n")
        else:
            file.write(f"   Controller:  no fb + random ff\n\n")
        file.write(f"METRICS\n")
        file.write(f"   Elapsed time forward pass (train set): {elatime_forward_pass_training}\n")
        file.write(f"   Elapsed time training output layer:    {elatime_train_output_layer}\n")
        file.write(f"   Elapsed time forward pass (test set):  {elatime_forward_pass_testing}\n")
        file.write(f"   NRMSE (train set): {train_nrmse}\n")
        file.write(f"   NRMSE (test set):  {test_nrmse}\n")


    # =========================================================
    # Show all plots
    # =========================================================
    #plt.show()
    plt.close('all')