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
from sklearn.linear_model import LogisticRegression
import joblib
import copy

import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
import sys
import time

from soromox.systems.my_systems import PlanarPCS_simple

# Folders
curr_folder = Path(__file__).parent # current folder
sys.path.append(str(curr_folder.parent)) # scripts folder
main_folder = curr_folder.parent.parent # main folder "codes"
dataset_folder = main_folder/'datasets' # folder with the dataset
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

    2.  Loads the MNIST dataset and extracts part of it. In particular, a portion `train_set_portion` is extracted from the 
        full train MNIST set and a portion `test_set_portion` from the test MNIST set.

    3a. If `train` is True, output layer (scaler + classifier) is trained with the given reservoir. Trained scaler and 
        classifier are then saved in data folder named `experiment_name`. 
    3b. If `train` is False, loads a given output layer (scaler + classifier) from data folder named `experiment_name`.

    4.  The full architecture (reservoir + output layer) is tested on the specified portion of the test set.

    5.  An example from the MNIST test set, specified in `example_idx`, is loaded and used for inference (a black image can
        be tested if example_idx='black').

    6.  Another example (the following image) is taken from the MNIST dataset and used for inference. Dynamics of the reservoir
        is compared with the previous one.

    7.  Saves plots and metrics on the test set (and train set if training was performed) in `experiment_name` data and plots 
        folders. Also saves the dynamics of the reservoir for the given example.
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
    example_idx = 0 # if it is an integer i, loads the i-th image from MNIST test set. Otherwise 'black' for black image
    train_set_portion = 0.5 # fraction (or number of images) of the original train set (60 000 images) to use. If 1: full dataset
    test_set_portion = 0.5 # fraction (or number of images) of the original test set (10 000 images) to use. If 1: full dataset
    batch_size = 1000 # batch size for training and testing. Should be as high as possible, consistently with pc memory and datasets sizes
    dt_u = 0.006 # time step for the input u. (in the RON paper dt = 0.042 s)

    # Output layer (scaler + classifier)
    experiment_name = f'sMNIST/N6/a_run{n_run}' # name of the experiment to save/load
    train = True # if True, perform training (output layer). Otherwise, test saved 'experiment_name' model

    # Reservoir (robots + map + controller)
    load_model_path = saved_data_folder/'more_soft_robots_optimization'/f'sMNIST/N6/default_run{n_run}' # choose the reservoir to load (robots + map + controller)
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

    # Load MNIST dataset
    train_set, test_set = load_mnist_data(dataset_folder/'MNIST') # shape (n_images, 1, 28, 28)

    # Convert (n_imgs, 1, 28, 28) --> (n_imgs, 784, 1)
    train_set["images"] = train_set["images"].reshape(train_set["images"].shape[0], -1, 1)
    test_set["images"] = test_set["images"].reshape(test_set["images"].shape[0], -1, 1)

    # Convert to jax
    train_set["images"] = jnp.array(train_set["images"], dtype=jnp.float64)
    train_set["labels"] = jnp.array(train_set["labels"], dtype=jnp.float64)

    test_set["images"] = jnp.array(test_set["images"], dtype=jnp.float64)
    test_set["labels"] = jnp.array(test_set["labels"], dtype=jnp.float64)

    # Take only a portion of the test/train sets
    fraction_train = train_set_portion if train_set_portion < 1.1 else train_set_portion/len(train_set["labels"])
    fraction_test = test_set_portion if test_set_portion < 1.1 else test_set_portion/len(test_set["labels"])

    key, subkey1, subkey2 = jax.random.split(key, 3)
    train_set, _, _ = split_dataset(subkey1, train_set, fraction_train)
    test_set, _, _ = split_dataset(subkey2, test_set, fraction_test)

    train_set_size = len(train_set["labels"])
    test_set_size = len(test_set["labels"])


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
    reservoir_forward = jax.jit(jax.vmap(reservoir, in_axes=(0,None,None,None))) # vmap reservoir's forward
    time_u = jnp.linspace(0, dt_u * (len(train_set["images"][0]) - 1), len(train_set["images"][0])) # define time vector for the input image 
    saveat = jnp.arange(0, time_u[-1], dt_u) # for saving simulation results


    # =====================================================
    # Training
    # =====================================================
    print(f'--- Experiment ---\n'
        f'name:  {experiment_name}\n'
        f'model: {load_model_path}')
    print()

    if train:
        # Train the output layer (classifier) (1): pass all the inputs in the train set to the model
        print(f'--- Generating previsions for training ---')
        key, subkey = jax.random.split(key)
        batch_ids = batch_indx_generator(subkey, train_set_size, batch_size=batch_size) # create indices for the batches
        last_states, labels = [], []
        start = time.perf_counter()
        for i in tqdm(range(len(batch_ids)), 'Model forward'):
            batch_i_ids = batch_ids[i]
            train_batch = extract_batch(train_set, batch_i_ids)
            _, _, _, _, last_states_batch = reservoir_forward(train_batch["images"], time_u, saveat, dt_sim) # shape (batch_size, num_hidden_units)
            last_states.append(last_states_batch)
            labels.append(train_batch["labels"])

        last_states = jnp.concatenate(last_states) # shape (num_train_images, num_hidden_units)
        labels = jnp.concatenate(labels) # shape (num_train_images,)
        last_states.block_until_ready()
        labels.block_until_ready()
        stop = time.perf_counter() 
        elatime_forward_pass_training = stop - start
        print(f'Elapsed time: {elatime_forward_pass_training}')

        # Train the output layer (classifier) (2): logistic regression of the output layer
        print(f'\n--- Training the classifier (regression) ---')
        start = time.perf_counter()
        scaler = preprocessing.StandardScaler().fit(onp.array(last_states))
        activations = scaler.transform(onp.array(last_states))
        classifier = LogisticRegression(max_iter=1000).fit(onp.array(activations), onp.array(labels))
        stop = time.perf_counter()
        elatime_train_output_layer = stop - start
        print(f'Elapsed time: {elatime_train_output_layer}')

        # Save the trained classifier and scaler
        joblib.dump(scaler, data_folder/'scaler.pkl')
        joblib.dump(classifier, data_folder/'classifier.pkl')

        # Train accuracy
        train_accuracy = classifier.score(activations, labels)
        print(f'Accuracy on the train set: {train_accuracy}')


    # =====================================================
    # Testing on the test set
    # =====================================================
    if train:
        print()

    # If training was not performed, load saved data
    if not train:
        scaler = joblib.load(data_folder/'scaler.pkl')
        classifier = joblib.load(data_folder/'classifier.pkl')

    # Forward on the test set
    print(f'--- Evaluating perfomances (test set) ---')
    key, subkey = jax.random.split(key)
    batch_ids = batch_indx_generator(subkey, test_set_size, batch_size=batch_size) # create indices for the batches
    last_states, labels = [], []
    start = time.perf_counter()
    for i in tqdm(range(len(batch_ids)), 'Model forward'):
        batch_i_ids = batch_ids[i]
        test_batch = extract_batch(test_set, batch_i_ids)
        _, _, _, _, last_states_batch = reservoir_forward(test_batch["images"], time_u, saveat, dt_sim) # shape (batch_size, num_hidden_units)
        last_states.append(last_states_batch)
        labels.append(test_batch["labels"])

    last_states = jnp.concatenate(last_states) # shape (num_test_images, num_hidden_units)
    labels = jnp.concatenate(labels) # shape (num_test_images,)
    last_states.block_until_ready()
    labels.block_until_ready()
    activations = scaler.transform(onp.array(last_states))
    stop = time.perf_counter() 
    elatime_forward_pass_testing = stop - start
    print(f'Elapsed time: {elatime_forward_pass_testing}')

    # Accuracy
    test_accuracy = classifier.score(activations, labels)
    print(f'Accuracy on the test set: {test_accuracy}')

    # Visualize the activations for all the test set
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    for i in range(last_states.shape[1]):
        ax1.scatter(last_states[:,i], (i+1)*np.ones(len(last_states)), label=f'Component {i+1}')
    ax1.set_title('Last states')
    ax1.set_xlabel(r'$y(t_{f})$')
    ax1.set_ylabel('component')
    ax1.grid(True)

    for i in range(activations.shape[1]):
        ax2.scatter(activations[:,i], (i+1)*np.ones(len(activations)), label=f'Component {i+1}')
    ax2.set_title('Activations')
    ax2.set_xlabel(r'$\tilde{y}$')
    ax2.set_ylabel('component')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(plots_folder/'all_testset_activations', bbox_inches='tight')
    #plt.show()


    # =====================================================
    # Testing on a single image
    # =====================================================
    print(f'\n--- Testing single example ---')

    # Load image to test
    if example_idx == 'black':
        image = jnp.zeros((784,)) # completely black image (null input), shape (784,)
        image_raw = jnp.zeros((28,28)) # shape (28, 28)
        label = 0
    else:
        image = test_set["images"][example_idx] # shape (784,)
        image_raw = image.reshape(28,28) # shape (28, 28)
        label = test_set["labels"][example_idx]

    # Try inference
    start = time.perf_counter()
    (
        time_ts,
        state_reservoir_ts, # reservoir's states evolution. Shape (n_timesteps, 2*n_hid)
        state_pcs_ts, # pcs's states evolution. Shape (n_timesteps, 2*3*n_pcs*n_robots)
        actuation_ts, # pcs actuation. Shape (n_timesteps, 3*n_pcs*n_robots)
        last_states
    ) = reservoir(image, time_u, saveat, dt_sim)
    y_ts, yd_ts = jnp.split(state_reservoir_ts, 2, axis=1) # reservoir states
    _, q_ts, _ = jax.vmap(robots_system.transform_Z)(state_pcs_ts) # shape (n_steps, n_robots, 3*n_pcs)
    Q_ts, _ = jnp.split(state_pcs_ts, 2, axis=1) # shape (n_steps, 3*n_pcs*n_robots)
    stop = time.perf_counter()
    print(f'Elapsed time (simulation): {stop-start}')

    last_states = last_states[None,:] # sklear requires input (n_inputs, dim_inputs)
    activations = scaler.transform(last_states)
    pred = classifier.predict(activations)[0] # prediction
    probs = classifier.predict_proba(activations).squeeze() # probabilities

    # Show prediction
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(image_raw, cmap='gray')
    ax1.set_title('Input')

    ax2.bar(np.arange(10), probs, color='skyblue')
    ax2.set_title(f'Prediction: {pred}')
    ax2.set_xlabel('classes')
    ax2.set_ylabel('probability')
    ax2.set_xticks(np.arange(10))
    plt.tight_layout()
    plt.savefig(plots_folder/'Example_inference', bbox_inches='tight') 
    #plt.show()

    # Show max 15 DOFs in the plots
    if 3*n_pcs*n_robots > 15:
        n_show = 15
    else:
        n_show = 3*n_pcs*n_robots

    n_cols = min(2, n_show)
    n_rows = int(np.ceil(n_show / n_cols))

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
        ax2 = ax.twinx()
        ax2.plot(time_u, image, 'k', alpha=0.3, label=r'reservoir input $u(t)$')
        ax2.set_ylabel(r'$u$')
        ax2.set_ylim([-0.1, 2])

        ax.plot(time_ts, actuation_ts[:,i], 'r', label=r'robots actuation $\tau(t)$')
        ax.set_xlabel('t [s]')
        ax.set_ylabel(r'$\tau$')

        ax.grid(True)
        ax.set_title(f'Component {i+1}')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

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
    # Compare with another image
    # =========================================================
    print(f'\n--- Testing another example (for comparison) ---')

    # Load another image from MNIST dataset
    if example_idx == 'black':
        image2 = test_set["images"][0] # shape (784,)
    else:
        image2 = test_set["images"][example_idx+1] # shape (784,)

    # Try inference
    start = time.perf_counter()
    (
        time_ts2,
        state_reservoir_ts2,
        state_pcs_ts2,
        actuation_ts2,
        last_states2
    ) = reservoir(image2, time_u, saveat, dt_sim)
    y_ts2, yd_ts2 = jnp.split(state_reservoir_ts2, 2, axis=1)
    _, q_ts2, _ = jax.vmap(robots_system.transform_Z)(state_pcs_ts2) # shape (n_steps, n_robots, 3*n_pcs)
    Q_ts2, _ = jnp.split(state_pcs_ts2, 2, axis=1) # shape (n_steps, 3*n_pcs*n_robots)
    stop = time.perf_counter()
    print(f'Elapsed time (simulation): {stop-start}')

    # Compare reservoir/robot evolutions
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12,9))
    for i, ax in enumerate(axs.flatten()):
        ax.plot(time_ts, y_ts[:,i], 'b', label='reservoir (ex. 1)')
        ax.plot(time_ts, Q_ts[:,i], 'r', label='soft robot (ex. 1)')
        ax.plot(time_ts2, y_ts2[:,i], 'b--', label='reservoir (ex. 2)')
        ax.plot(time_ts2, Q_ts2[:,i], 'r--', label='soft robot (ex. 2)')
        ax.grid(True)
        ax.set_xlabel('t [s]')
        ax.set_ylabel('y, q')
        ax.set_title(f'Component {i+1}')
        ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(plots_folder/'Comparison_inference_evolution', bbox_inches='tight') 
    #plt.show()

    # Compare actuation signals Tau(t)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16,13))
    for i, ax in enumerate(axs.flatten()):
        ax2 = ax.twinx()
        ax2.plot(time_u, image, 'k', alpha=0.3, label=r'reservoir input $u(t)$ (ex. 1)')
        ax2.plot(time_u, image2, 'k--', alpha=0.3, label=r'reservoir input $u(t)$ (ex. 2)')
        ax2.set_ylabel(r'$u$')
        ax2.set_ylim([-0.1, 2])

        ax.plot(time_ts, actuation_ts[:,i], 'r', label=r'robot actuation $\tau(t)$ (ex. 1)')
        ax.plot(time_ts2, actuation_ts2[:,i], 'r--', label=r'robot actuation $\tau(t)$ (ex. 2)')
        ax.set_xlabel('t [s]')
        ax.set_ylabel(r'$\tau$')
        y_min, y_max = ax.get_ylim()
        ax.set_ylim([y_min, 1.5*y_max])

        ax.grid(True)
        ax.set_title(f'Component {i+1}')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(plots_folder/'Comparison_inference_actuation', bbox_inches='tight') 
    #plt.show()


    # =========================================================
    # Save text file with performances and data
    # =========================================================

    if not train:
        train_set_size = '(training was not performed)'
        train_accuracy = '(training was not performed)'
        elatime_forward_pass_training = '(training was not performed)'
        elatime_train_output_layer = '(training was not performed)'

    with open(data_folder/'performances.txt', 'w') as file:
        file.write(f"SETUP\n")
        file.write(f"   Train set size: {train_set_size}\n")
        file.write(f"   Test set size:  {test_set_size}\n\n")
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
        file.write(f"   Accuracy (train set): {train_accuracy}\n")
        file.write(f"   Accuracy (test set):  {test_accuracy}\n")


    # =========================================================
    # Show all plots
    # =========================================================
    #plt.show()
    plt.close('all')