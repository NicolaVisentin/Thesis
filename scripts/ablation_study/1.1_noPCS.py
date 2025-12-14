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
import optax

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from pathlib import Path
import time
import sys

from soromox.systems.my_systems import PlanarPCS_simple

curr_folder = Path(__file__).parent      # current folder
sys.path.append(str(curr_folder.parent)) # scripts folder
from utilis import *

# Jax settings
print(f"\nAvailable devices: {jax.devices()}\n")
jax.config.update("jax_enable_x64", True)  # double precision
jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)
seed = 123
key = jax.random.key(seed)

# Folders
main_folder = curr_folder.parent.parent                                            # main folder "codes"
plots_folder = main_folder/'plots and videos'/curr_folder.stem/Path(__file__).stem # folder for plots and videos
dataset_folder = main_folder/'datasets'                                            # folder with the dataset
data_folder = main_folder/'saved data'/curr_folder.stem/Path(__file__).stem        # folder for saving data
data_folder_ref = main_folder/'saved data'/curr_folder.stem/'0.0_reference'        # folder with referenece saved data

data_folder.mkdir(parents=True, exist_ok=True)
plots_folder.mkdir(parents=True, exist_ok=True)


# =====================================================
# Script settings
# =====================================================
train_samples = True            # if True, short training on many samples. If False, long training on the best sample
ref_data_prefix = 'SAMPLES_REF' # prefix of the REFERENCE data (for same initial condition)
load_case_prefix = 'BEST_REF'   # if train_samples is False, choose prefix of the experiment to load


# =====================================================
# Functions for optimization
# =====================================================

# Converts A -> A_raw
@jax.jit
def A2Araw(A: Array, s_thresh: float=0.0) -> Tuple:
    """A_raw is tuple (U,s,Vt) with SVD of A = U*diag(s)*Vt, where s vector is parametrized with softplus
    to ensure s_i > thresh >= 0 for all i."""
    U, s, Vt = jnp.linalg.svd(A)          # decompose A = U*S*V.T, with s=diag(S) and Vt=V^T
    s_raw = InverseSoftplus(s - s_thresh) # convert singular values
    A_raw = (U, s_raw, Vt)
    return A_raw

# Converts A_raw -> A
@jax.jit
def Araw2A(A_raw: Tuple, s_thresh: float=0.0) -> Array:
    """A_raw is tuple (U,s,Vt) with SVD of A = U*diag(s)*V^T, where s vector is parametrized with softplus
    to ensure s_i > thresh >= 0 for all i."""
    U, s_raw, Vt = A_raw
    s = jax.nn.softplus(s_raw) + s_thresh
    A = U @ jnp.diag(s) @ Vt
    return A

# Loss function (COMPLETE, also considers robot update). Used to compare different samples.
@jax.jit
def LossComplete(
        params_optimiz : Sequence, 
        data_batch : Dict, 
        robot : PlanarPCS_simple,
        mlp_controller : MLP,
        s_thresh : float
) -> Tuple[float, Dict]:
    """
    Computes loss function over a batch of data for certain parameters. In this case:
    Takes a batch of datapoints (y, yd), computes the forward dynamics ydd_hat = f_approximator(y,yd)
    and computes the loss as the MSE in the batch between predictions ydd_hat and labels ydd. In particular,
    our approximator here is: f_approximator = T_out( f_pcs( T_in(y,yd) ) ), where T_in: (q, qd)=(A*y+c, A*yd)
    and T_out: ydd_hat=A*qdd.

    Args
    ----
    params_optimiz : Sequence
        Parameters for the opimization. In this case a list with:
        - **Phi**: Tuple (MAP, CONTR). MAP is tuple with mapping params (A_raw, c), where A_raw is tuple 
                   with "raw" SVD of A (U, s_raw, V). CONTR is a tuple with MLP controller parameters (layers). 
        - **phi**: Tuple with pcs params (L_raw, D_raw, r_raw, rho_raw, E_raw, G_raw).
    robot : PlanarPCS_simple
        Robot instance.
    mlp_controller : MLP
        MLP instance.
    s_thresh : float
        Threshold on the singular values for the mapping.
    data_batch : Dict
        Dictionary with datapoints and labels to compute the loss. In this case has keys:
        - **"y"**: Batch of datapoints y. Shape (batch_size, n_ron)
        - **"yd"**: Batch of datapoints yd. Shape (batch_size, n_ron)
        - **"ydd"**: Batch of labels ydd. Shape (batch_size, n_ron)

    Returns
    -------
    loss : float
        Scalar loss computed as MSE in the batch between predictions and labels.
    metrics : Dict[float, Dict]
        Dictionary of useful metrics.
    """
    # extract everything
    y_batch = data_batch["y"]
    yd_batch = data_batch["yd"]
    ydd_batch = data_batch["ydd"]

    Phi, phi = params_optimiz

    MAP, CONTR = Phi
    A_raw, c = MAP
    L_raw, D_raw, r_raw, rho_raw, E_raw, G_raw = phi

    # convert parameters
    A = Araw2A(A_raw, s_thresh)
    L = jax.nn.softplus(L_raw)
    D = jnp.diag(jax.nn.softplus(D_raw))
    r = jax.nn.softplus(r_raw)
    rho = jax.nn.softplus(rho_raw)
    E = jax.nn.softplus(E_raw)
    G = jax.nn.softplus(G_raw)

    # update robot and controller
    robot_updated = robot.update_params({"L": L, "D": D, "r": r, "rho": rho, "E": E, "G": G})
    controller_updated = mlp_controller.update_params(CONTR)

    # convert variables
    q_batch = y_batch @ jnp.transpose(A) + c # shape (batch_size, 3*n_pcs)
    qd_batch = yd_batch @ jnp.transpose(A)   # shape (batch_size, 3*n_pcs)

    # predictions
    z_batch = jnp.concatenate([q_batch, qd_batch], axis=1) # state z=[q^T, qd^T]. Shape (batch_size, 2*3*n_pcs)
    tau_batch = controller_updated.forward_batch(z_batch)
    actuation_arg = (tau_batch,)

    forward_dynamics_vmap = jax.vmap(robot_updated.forward_dynamics, in_axes=(None,0,0))
    zd_batch = forward_dynamics_vmap(0, z_batch, actuation_arg) # state derivative zd=[qd^T, qdd^T]. Shape (batch_size, 2*3*n_pcs)
    _, qdd_batch = jnp.split(zd_batch, 2, axis=1) 

    # compute loss (compare predictions ydd_hat=B*qdd+d with labels ydd)
    ydd_hat_batch = jnp.linalg.solve(A, qdd_batch.T).T # convert predictions from qdd to ydd
    MSE = jnp.mean(jnp.sum((ydd_hat_batch - ydd_batch)**2, axis=1))
    loss = MSE

    # store metrics
    metrics = {
        "MSE": MSE,
        "predictions": ydd_hat_batch,
        "labels": ydd_batch,
    }
    
    return loss, metrics

# Loss function
@jax.jit
def Loss(
        params_optimiz : Sequence, 
        data_batch : Dict, 
        robot : PlanarPCS_simple,
        mlp_controller : MLP,
        s_thresh : float,
        additional_arg : Tuple
) -> Tuple[float, Dict]:
    """
    Computes loss function over a batch of data for certain parameters. In this case:
    Takes a batch of datapoints (y, yd), computes the forward dynamics ydd_hat = f_approximator(y,yd)
    and computes the loss as the MSE in the batch between predictions ydd_hat and labels ydd. In particular,
    our approximator here is: f_approximator = T_out( f_pcs( T_in(y,yd) ) ), where T_in: (q, qd)=(A*y+c, A*yd)
    and T_out: ydd_hat=A*qdd.

    Args
    ----
    params_optimiz : Sequence
        Parameters for the opimization. In this case a list with:
        - **Phi**: Tuple (MAP, CONTR). MAP is tuple with mapping params (A_raw, c), where A_raw is tuple 
                   with "raw" SVD of A (U, s_raw, V). CONTR is a tuple with MLP controller parameters (layers). 
        - **phi**: Tuple with pcs params (L_raw, D_raw, r_raw, rho_raw, E_raw, G_raw). !!! THEY ARE NOT USED FOR 
                   UPDATING THE ROBOT BUT ONLY FOR GRADIENT COMPUTATION (THAT WILL ALWAYS BE ZERO, SINCE THEY 
                   ARE NOT USED). THIS TRICK IS USED TO MAINTAIN THE SAME STRUCTURE OF THE FUNCTION. !!!
    robot : PlanarPCS_simple
        Robot instance.
    mlp_controller : MLP
        MLP instance.
    s_thresh : float
        Threshold on the singular values for the mapping.
    data_batch : Dict
        Dictionary with datapoints and labels to compute the loss. In this case has keys:
        - **"y"**: Batch of datapoints y. Shape (batch_size, n_ron)
        - **"yd"**: Batch of datapoints yd. Shape (batch_size, n_ron)
        - **"ydd"**: Batch of labels ydd. Shape (batch_size, n_ron)
    additional_arg: Tuple
        Tuple with pcs params (L_raw, D_raw, r_raw, rho_raw, E_raw, G_raw). !!! THEY ARE USED FOR UPDATING THE ROBOT,
        BUT NOT FOR GRADIENTS COMPUTATION !!!

    Returns
    -------
    loss : float
        Scalar loss computed as MSE in the batch between predictions and labels.
    metrics : Dict[float, Dict]
        Dictionary of useful metrics.
    """
    # extract everything
    y_batch = data_batch["y"]
    yd_batch = data_batch["yd"]
    ydd_batch = data_batch["ydd"]

    Phi, phi = params_optimiz

    MAP, CONTR = Phi
    A_raw, c = MAP
    L_raw, D_raw, r_raw, rho_raw, E_raw, G_raw = phi
    L_raw_fixed, D_raw_fixed, r_raw_fixed, rho_raw_fixed, E_raw_fixed, G_raw_fixed = additional_arg

    # convert parameters
    A = Araw2A(A_raw, s_thresh)
    L = jax.nn.softplus(L_raw)
    D = jnp.diag(jax.nn.softplus(D_raw))
    r = jax.nn.softplus(r_raw)
    rho = jax.nn.softplus(rho_raw)
    E = jax.nn.softplus(E_raw)
    G = jax.nn.softplus(G_raw)

    L_fixed = jax.nn.softplus(L_raw_fixed)
    D_fixed = jnp.diag(jax.nn.softplus(D_raw_fixed))
    r_fixed = jax.nn.softplus(r_raw_fixed)
    rho_fixed = jax.nn.softplus(rho_raw_fixed)
    E_fixed = jax.nn.softplus(E_raw_fixed)
    G_fixed = jax.nn.softplus(G_raw_fixed)

    # update robot (with parameters NOT used for gradient computation) and controller
    robot_updated = robot.update_params(
        {"L": L_fixed, "D": D_fixed, "r": r_fixed, 
         "rho": rho_fixed, "E": E_fixed, "G": G_fixed}
    )
    controller_updated = mlp_controller.update_params(CONTR)

    # convert variables
    q_batch = y_batch @ jnp.transpose(A) + c # shape (batch_size, 3*n_pcs)
    qd_batch = yd_batch @ jnp.transpose(A)   # shape (batch_size, 3*n_pcs)

    # predictions
    z_batch = jnp.concatenate([q_batch, qd_batch], axis=1) # state z=[q^T, qd^T]. Shape (batch_size, 2*3*n_pcs)
    tau_batch = controller_updated.forward_batch(z_batch)
    actuation_arg = (tau_batch,)

    forward_dynamics_vmap = jax.vmap(robot_updated.forward_dynamics, in_axes=(None,0,0))
    zd_batch = forward_dynamics_vmap(0, z_batch, actuation_arg) # state derivative zd=[qd^T, qdd^T]. Shape (batch_size, 2*3*n_pcs)
    _, qdd_batch = jnp.split(zd_batch, 2, axis=1) 

    # compute loss (compare predictions ydd_hat=B*qdd+d with labels ydd)
    ydd_hat_batch = jnp.linalg.solve(A, qdd_batch.T).T # convert predictions from qdd to ydd
    MSE = jnp.mean(jnp.sum((ydd_hat_batch - ydd_batch)**2, axis=1))
    loss = MSE

    # store metrics
    metrics = {
        "MSE": MSE,
        "predictions": ydd_hat_batch,
        "labels": ydd_batch,
    }
    
    return loss, metrics

# Some useful vmapped functions
LossComplete_vmap = jax.vmap(LossComplete, in_axes=(0,None,None,None,None)) # for evaluation
Loss_vmap = jax.vmap(Loss, in_axes=(0,None,None,None,None,0)) # for evaluation
A2Araw_vmap = jax.vmap(A2Araw, in_axes=(0,None))
Araw2A_vmap = jax.vmap(Araw2A, in_axes=(0,None))


# =====================================================
# Prepare datasets
# =====================================================

# Load dataset: m data from a RON with n_ron oscillators
dataset = onp.load(dataset_folder/'soft robot optimization/N6_simplified/dataset_m1e5_N6_simplified.npz')
y = dataset["y"]     # position samples of the RON oscillators. Shape (m, n_ron)
yd = dataset["yd"]   # velocity samples of the RON oscillators. Shape (m, n_ron)
ydd = dataset["ydd"] # accelerations of the RON oscillators. Shape (m, n_ron)

# Convert into jax
y_dataset = jnp.array(y)
yd_dataset = jnp.array(yd)
ydd_dataset = jnp.array(ydd)

dataset = {
    "y": y_dataset,
    "yd": yd_dataset,
    "ydd": ydd_dataset,
}

# Split into train/validation/test
key, subkey = jax.random.split(key)
train_set, val_set, test_set = split_dataset(
    subkey,
    dataset,
    train_ratio=0.7,
    test_ratio=0.2,
)
train_size, n_ron = train_set["y"].shape


# =====================================================
# Optimization hyperparameters
# =====================================================

if train_samples:
    # Epochs and batches
    n_epochs = onp.load(data_folder_ref/f'{ref_data_prefix}_all_loss_curves.npz')["train_losses_ts"].shape[1] # number of epochs
    batch_size = 2**6 # batch size

    batches_per_epoch = batch_indx_generator(key, train_size, batch_size).shape[0]

    # Optimizer and learning rate
    lr = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=1e-3,
        warmup_steps=15*batches_per_epoch,
        decay_steps=n_epochs*batches_per_epoch,
        end_value=1e-5
    )
    optimizer = optax.adam(learning_rate=lr)

    # Number of samples (optimizations to run in parallel)
    n_samples = onp.load(data_folder_ref/f'{ref_data_prefix}_all_loss_curves.npz')["train_losses_ts"].shape[0]
else:
    # Epochs and batches
    n_epochs = 1500   # number of epochs
    batch_size = 2**6 # batch size

    batches_per_epoch = batch_indx_generator(key, train_size, batch_size).shape[0]

    # Optimizer and learning rate
    lr = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=1e-3,
        warmup_steps=15*batches_per_epoch,
        decay_steps=n_epochs*batches_per_epoch,
        end_value=1e-5
    )
    optimizer = optax.adam(learning_rate=lr)

    # Number of samples (optimizations to run in parallel)
    n_samples = 1


# =====================================================
# Initial guesses for parameters
# =====================================================

if train_samples:
    # Load data (samples reference case)
    all_robot_before = onp.load(data_folder_ref/f'{ref_data_prefix}_all_data_robot_before.npz') # load all robot data before training
    all_map_before = onp.load(data_folder_ref/f'{ref_data_prefix}_all_data_map_before.npz')     # load all map data before training

    # PCS robot
    n_pcs = 2

    L0 = jnp.array(all_robot_before["L_before"])
    D0 = jnp.array(all_robot_before["D_before"])
    r0 = jnp.array(all_robot_before["r_before"])
    rho0 = jnp.array(all_robot_before["rho_before"])
    E0 = jnp.array(all_robot_before["E_before"])
    G0 = jnp.array(all_robot_before["G_before"])

    L0_raw = InverseSoftplus(L0)
    D0_raw = InverseSoftplus(D0)
    r0_raw = InverseSoftplus(r0)
    rho0_raw = InverseSoftplus(rho0)
    E0_raw = InverseSoftplus(E0)
    G0_raw = InverseSoftplus(G0)

    phi0 = (L0_raw, D0_raw, r0_raw, rho0_raw, E0_raw, G0_raw)

    # Mapping (identity matrix in this case)
    s_thresh = 1e-4
    A0 = jnp.array(all_map_before["A_before"])
    c0 = jnp.array(all_map_before["c_before"])

    A0_raw = A2Araw_vmap(A0, s_thresh)

    MAP0 = (A0_raw, c0)

    # MLP fb controller
    key, keyController = jax.random.split(key)
    mlp_sizes = [2*3*n_pcs, 64, 64, 3*n_pcs] 
    mlp_controller = MLP(key=keyController, layer_sizes=mlp_sizes, scale_init=0.001) # dummy instance

    CONTR0 = mlp_controller.load_params(data_folder_ref/f'{ref_data_prefix}_all_data_controller_before.npz')
else:
    # Load data (! best reference, not best noPCS in this case !)
    all_robot_before = onp.load(data_folder_ref/f'{load_case_prefix}_all_data_robot_before.npz') # load all robot data before training
    all_map_before = onp.load(data_folder_ref/f'{load_case_prefix}_all_data_map_before.npz')     # load all map data beofre training

    # PCS robot
    n_pcs = 2

    L0 = jnp.array(all_robot_before["L_before"])
    D0 = jnp.array(all_robot_before["D_before"])
    r0 = jnp.array(all_robot_before["r_before"])
    rho0 = jnp.array(all_robot_before["rho_before"])
    E0 = jnp.array(all_robot_before["E_before"])
    G0 = jnp.array(all_robot_before["G_before"])

    L0_raw = InverseSoftplus(L0)
    D0_raw = InverseSoftplus(D0)
    r0_raw = InverseSoftplus(r0)
    rho0_raw = InverseSoftplus(rho0)
    E0_raw = InverseSoftplus(E0)
    G0_raw = InverseSoftplus(G0)

    phi0 = (L0_raw, D0_raw, r0_raw, rho0_raw, E0_raw, G0_raw)

    # Mapping
    s_thresh = 1e-4 # min threshold for s_i during optimization
    A0 = jnp.array(all_map_before["A_before"])
    c0 = jnp.array(all_map_before["c_before"])

    A0_raw = A2Araw_vmap(A0, s_thresh)

    MAP0 = (A0_raw, c0)

    # MLP fb controller
    key, subkey = jax.random.split(key)
    mlp_sizes = [2*3*n_pcs, 64, 64, 3*n_pcs] 
    mlp_controller = MLP(key=subkey, layer_sizes=mlp_sizes, scale_init=0.001) # dummy instance

    CONTR0 = mlp_controller.load_params(data_folder_ref/f'{load_case_prefix}_all_data_controller_before.npz')

# Collect all parameters
Phi0 = (MAP0, CONTR0)
params_optimiz0 = (Phi0, phi0)

# "Dummy" instantiation of robot class
parameters = {
    "th0": jnp.array(jnp.pi/2),
    "L": L0[0],
    "r": r0[0],
    "rho": rho0[0],
    "g": jnp.array([0.0, 9.81]), # !! gravity UP !!
    "E": E0[0],
    "G": G0[0],
    "D": jnp.diag(D0[0])
}
robot = PlanarPCS_simple(
    num_segments = n_pcs,
    params = parameters,
    order_gauss = 5
)

# Compute RMSE on the test set before optimization for the various guesses
_, metrics_check = LossComplete_vmap(params_optimiz0, test_set, robot, mlp_controller, s_thresh)
_, metrics = Loss_vmap(params_optimiz0, test_set, robot, mlp_controller, s_thresh, phi0)
RMSE_before_check = onp.sqrt(metrics_check["MSE"])
RMSE_before = onp.sqrt(metrics["MSE"])
print('RMSE before:         ', RMSE_before)
print('RMSE beofre (check): ', RMSE_before_check)

# Compute actuation power mean squared value on the test set before optimization for the various guesses
powers_msv_before = []
for i in range(n_samples):
    params_i = mlp_controller.extract_params_from_batch(CONTR0, i)
    q = test_set["y"] @ A0[i].T + c0[i]                # shape (testset_size, 3*n_pcs)
    qd = test_set["yd"] @ A0[i].T                      # shape (testset_size, 3*n_pcs)
    z = jnp.concatenate([q, qd], axis=1)               # shape (testset_size, 2*3*n_pcs)
    tau_i = mlp_controller._forward_batch(params_i, z) # shape (testset_size, 3*n_pcs)
    power_i = jnp.sum(tau_i * qd, axis=1)              # shape (testset_size,)
    power_msv_i = jnp.mean(power_i**2)                 # scalar
    powers_msv_before.append(power_msv_i)
powers_msv_before = jnp.stack(powers_msv_before, axis=0) # shape (n_samples,)


# =====================================================
# Optimizations
# =====================================================

# vmap function for training
train_in_parallel = jax.jit(
    jax.vmap(train_with_scan_modified, in_axes=(0,None,0,None,None,None,None,None,0)),
    static_argnums=(1,3,6,7)
)

# Correct signature for loss function
Loss = jax.jit(partial(Loss, robot=robot, mlp_controller=mlp_controller, s_thresh=s_thresh))

# Run trainings in parallel
keys = jax.random.split(key, n_samples+1)
key, keysTrain = keys[0], keys[1:]

print(f"Starting optimizations ({n_samples} samples, {n_epochs} epochs)...")
start = time.perf_counter()
results = train_in_parallel(
    keysTrain,
    optimizer,
    params_optimiz0,
    Loss,
    train_set,
    val_set,
    n_epochs,
    batch_size,
    phi0,
)
params_optimiz_after = results["params_optimiz"]
train_loss_ts = results["train_loss_ts"]
train_MSE_ts = results["train_MSE_ts"]
val_loss_ts = results["val_loss_ts"]
val_MSE_ts = results["val_MSE_ts"]

jax.block_until_ready(params_optimiz_after)
end = time.perf_counter()
elatime_optimiz = end - start
print(f'Elapsed time: {elatime_optimiz} s')

# Extract (raw) parameters
Phi_after, phi_after = params_optimiz_after

MAP_after, CONTR_after = Phi_after
A_raw_after, c_after = MAP_after
L_raw_after, D_raw_after, r_raw_after, rho_raw_after, E_raw_after, G_raw_after = phi_after

# Convert parameters
A_after = Araw2A_vmap(A_raw_after, s_thresh)
L_after = jax.nn.softplus(L_raw_after)
D_after = jax.nn.softplus(D_raw_after)
r_after = jax.nn.softplus(r_raw_after)
rho_after = jax.nn.softplus(rho_raw_after)
E_after = jax.nn.softplus(E_raw_after)
G_after = jax.nn.softplus(G_raw_after)

# Compute RMSE on the test set after optimization for the various guesses
_, metrics_check = LossComplete_vmap(params_optimiz_after, test_set, robot, mlp_controller, s_thresh)
_, metrics = Loss_vmap(params_optimiz_after, test_set, robot, mlp_controller, s_thresh, phi_after)
RMSE_after_check = onp.sqrt(metrics_check["MSE"])
RMSE_after = onp.sqrt(metrics["MSE"])
print('RMSE after:         ', RMSE_after)
print('RMSE after (check): ', RMSE_after_check)

# Compute actuation power mean squared value on the test set after optimization for the various guesses
powers_msv_after = []
for i in range(n_samples):
    params_i = mlp_controller.extract_params_from_batch(CONTR_after, i)
    q = test_set["y"] @ A_after[i].T + c_after[i]      # shape (testset_size, 3*n_pcs)
    qd = test_set["yd"] @ A_after[i].T                 # shape (testset_size, 3*n_pcs)
    z = jnp.concatenate([q, qd], axis=1)               # shape (testset_size, 2*3*n_pcs)
    tau_i = mlp_controller._forward_batch(params_i, z) # shape (testset_size, 3*n_pcs)
    power_i = jnp.sum(tau_i * qd, axis=1)              # shape (testset_size,)
    power_msv_i = jnp.mean(power_i**2)                 # scalar
    powers_msv_after.append(power_msv_i)
powers_msv_after = jnp.stack(powers_msv_after, axis=0) # shape (n_samples,)

# Find best result
idx_best = jnp.argmin(RMSE_after)


# =====================================================
# Save results
# =====================================================

# Save hyperparameters
with open(data_folder/'hyperparameters.txt', 'w') as file:
    file.write(f"Optimizer: adam\n")
    file.write(f"learning rate: cosine (max=1e-3, min=1e-5) + linear warmup (start=1e-6, duration=15)\n")
    file.write(f"Epochs: {n_epochs}\n")
    file.write(f"Batch size: {batch_size}\n")
    file.write(f"Samples: {n_samples}\n")

# Save all n_samples sets of parameters before training
onp.savez(
    data_folder/'all_data_robot_before.npz', 
    L_before=onp.array(L0), 
    D_before=onp.array(D0),
    r_before=onp.array(r0),
    rho_before=onp.array(rho0),
    E_before=onp.array(E0),
    G_before=onp.array(G0),
)
onp.savez(
    data_folder/'all_data_map_before.npz', 
    A_before=onp.array(A0), 
    c_before=onp.array(c0)
)
mlp_controller._save_params(CONTR0, data_folder/'all_data_controller_before.npz')
onp.savez(
    data_folder/'all_rmse_before.npz', 
    RMSE_before=onp.array(RMSE_before)
)
onp.savez(
    data_folder/'all_powers_msv_before.npz', 
    powers_msv_before=onp.array(powers_msv_before)
)

# Save all n_samples sets of parameters after training
onp.savez(
    data_folder/'all_data_robot_after.npz', 
    L_after=onp.array(L_after), 
    D_after=onp.array(D_after),
    r_after=onp.array(r_after),
    rho_after=onp.array(rho_after),
    E_after=onp.array(E_after),
    G_after=onp.array(G_after),
)
onp.savez(
    data_folder/'all_data_map_after.npz', 
    A_after=onp.array(A_after), 
    c_after=onp.array(c_after)
)
mlp_controller._save_params(CONTR_after, data_folder/'all_data_controller_after.npz') 
onp.savez(
    data_folder/'all_rmse_after.npz', 
    RMSE_after=onp.array(RMSE_after)
)
onp.savez(
    data_folder/'all_powers_msv_after.npz', 
    powers_msv_after=onp.array(powers_msv_after)
)

# Save all loss curves
onp.savez(
    data_folder/'all_loss_curves.npz', 
    train_losses_ts = train_loss_ts,
    train_MSEs_ts = train_MSE_ts,
    val_losses_ts = val_loss_ts,
    val_MSEs_ts = val_MSE_ts
)


# =====================================================
# Visualize results
# =====================================================

# Visualize results of multiple optimizations
plt.figure()
plt.plot(range(n_samples), RMSE_before, 'gx', label='test RMSE before')
plt.plot(range(n_samples), RMSE_after, 'go', label='test RMSE after')
plt.plot(range(n_samples), onp.sqrt(train_MSE_ts[:,-1]), 'ro', label='final train RMSE')
plt.plot(range(n_samples), onp.sqrt(val_loss_ts[:,-1]), 'bo', label='final validation RMSE')
plt.yscale('log')
plt.grid(True)
plt.xlabel('test')
plt.ylabel('RMSE')
plt.title(f'Results for various initial guesses (elapsed time: {elatime_optimiz:.2f} s)')
plt.legend()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(plots_folder/'All_results', bbox_inches='tight')
#plt.show()

# Visualize best result
fig, ax1 = plt.subplots()

train_loss_line, = ax1.plot(range(n_epochs), train_loss_ts[idx_best,:n_epochs], 'r', label='train loss')
val_loss_line, = ax1.plot(onp.arange(1,n_epochs+1), val_loss_ts[idx_best,:n_epochs], 'b', label='validation loss')
train_MSE_line, = ax1.plot(range(n_epochs), train_MSE_ts[idx_best,:n_epochs], 'r--', label='train MSE')
val_MSE_line, = ax1.plot(onp.arange(1,n_epochs+1), val_MSE_ts[idx_best,:n_epochs], 'b--', label='validation MSE')
ax1.set_yscale('log')
ax1.set_xlabel('iterations')
ax1.set_ylabel('loss', color='k')
ax1.tick_params(axis='y', labelcolor='k')

ax2 = ax1.twinx()
lr_line, = ax2.plot(range(n_epochs), [lr(i*batches_per_epoch) for i in range(n_epochs)], linewidth=0.5, label='learning rate', color='gray')
ax2.set_ylabel('lr', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

lines = [train_loss_line, val_loss_line, train_MSE_line, val_MSE_line, lr_line]
labels = ['train loss', 'validation loss', 'train MSE', 'validation MSE', 'learning rate']
ax1.legend(lines, labels, loc='center right')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title(f'Best loss curve')
plt.tight_layout()
plt.savefig(plots_folder/'Best_result', bbox_inches='tight')
plt.show()