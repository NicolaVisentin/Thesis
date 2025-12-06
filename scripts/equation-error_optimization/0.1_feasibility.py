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
import diffrax
from diffrax import Tsit5, ConstantStepSize

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from pathlib import Path
from tqdm import tqdm
import time
import sys

curr_folder = Path(__file__).parent      # current folder
sys.path.append(str(curr_folder.parent)) # scripts folder
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

# Folders
main_folder = curr_folder.parent.parent                                            # main folder "codes"
plots_folder = main_folder/'plots and videos'/curr_folder.stem/Path(__file__).stem # folder for plots and videos
dataset_folder = main_folder/'datasets'                                            # folder with the dataset
saved_data_folder = main_folder/'saved data'                                       # folder for saved data
# data_folder = saved_data_folder/curr_folder.stem/Path(__file__).stem               # folder for saving data

# data_folder.mkdir(parents=True, exist_ok=True)
plots_folder.mkdir(parents=True, exist_ok=True)


# =====================================================
# Script settings
# =====================================================
use_scan = False         # choose whether to use normal for loop or lax.scan
show_simulations = True # choose whether to perform time simulations of the approximator (and comparison with RON)


# =====================================================
# Functions for optimization
# =====================================================

# Approximator dynamics
@jax.jit
def approximator_fd(z, params):
    y, yd = jnp.split(z, 2)
    p1, p2 = params
    ydd = -p1 * yd - p2 * y
    zd = jnp.concatenate([yd, ydd])
    return zd

# Loss function
@jax.jit
def Loss(
        params_optimiz : Tuple, 
        data_batch : Dict, 
) -> Tuple[float, Dict]:
    """
    Computes loss function over a batch of data for certain parameters. In this case:
    Takes a batch of datapoints (y, yd), computes the forward dynamics ydd_hat = f_approximator(y,yd)
    and computes the loss as the MSE in the batch between predictions ydd_hat and labels ydd.

    Args
    ----
    params_optimiz : Tuple
        Parameters for the opimization. In this case (p1, p2).
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
    y_batch, yd_batch, ydd_batch = data_batch["y"], data_batch["yd"], data_batch["ydd"]

    # predictions
    z = jnp.concatenate([y_batch, yd_batch], axis=1)

    forward_dynamics_vmap = jax.vmap(approximator_fd, in_axes=(0,None))
    zd = forward_dynamics_vmap(z, params_optimiz)
    _, ydd_hat_batch = jnp.split(zd, 2, axis=1) 

    # compute loss
    MSE = jnp.mean(jnp.sum((ydd_hat_batch - ydd_batch)**2, axis=1))
    loss = MSE

    # store metrics
    metrics = {
        "MSE": MSE,
        "predictions": ydd_hat_batch,
        "labels": ydd_batch,
    }

    return loss, metrics


# =====================================================
# Prepare datasets
# =====================================================

# Load dataset: m data from a RON with n_ron oscillators
dataset = onp.load(dataset_folder/'soft robot optimization/dataset_m1e5_N6_simplified.npz')
y = dataset["y"]     # position samples of the RON oscillators. Shape (m, n_ron)
yd = dataset["yd"]   # velocity samples of the RON oscillators. Shape (m, n_ron)
ydd = dataset["ydd"] # accelerations of the RON oscillators. Shape (m, n_ron)

# Extract only 1st oscillator (all of them are decoupled)
y = y[:,0,None]
yd = yd[:,0,None]
ydd = ydd[:,0,None]

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
train_size = len(train_set["y"])


# =====================================================
# Approximator before optimization
# =====================================================
print('--- BEFORE OPTIMIZATION ---')

# First guess of the parameters
p1 = 5.0
p2 = 3.0
params_optimiz = (p1, p2)

# If required, simulate approximator and compare its behaviour in time with the RON's one
if show_simulations:
    # Load simulation results from RON
    RON_evolution_data = onp.load(saved_data_folder/'RON_evolution_N6_simplified_a.npz')
    time_RONsaved = jnp.array(RON_evolution_data['time'])
    y_RONsaved = jnp.array(RON_evolution_data['y'][:,0,None])
    yd_RONsaved = jnp.array(RON_evolution_data['yd'][:,0,None])

    # Simulate current approximator
    z0 = jnp.concatenate([y_RONsaved[0], yd_RONsaved[0]])

    t0 = time_RONsaved[0]
    t1 = time_RONsaved[-1]
    dt = 1e-4
    saveat = np.arange(time_RONsaved[0], time_RONsaved[-1], (time_RONsaved[1]-time_RONsaved[0]))
    solver = Tsit5()
    step_size = ConstantStepSize()
    max_steps = int(1e6)
    term = lambda t, z, args: approximator_fd(z, params_optimiz)

    # Simulate approximator
    print('Simulating...')
    start = time.perf_counter()
    solution = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(term),
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=z0,
        solver=solver,
        stepsize_controller=step_size,
        max_steps=max_steps,
        saveat=diffrax.SaveAt(ts=saveat),
    )
    end = time.perf_counter()
    print(f'Elapsed time (simulation): {end-start} s')

    z_hat = solution.ys
    y_hat, yd_hat = jnp.split(z_hat, 2, axis=1)

    # Plot y(t) and y_hat(t)
    fig, ax = plt.subplots(1,1)
    ax.plot(saveat, y_hat, 'b', label=r'$\hat{y}(t)$')
    ax.plot(time_RONsaved, y_RONsaved, 'b--', label=r'$y_{RON}(t)$')
    ax.grid(True)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('y')
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_folder/'RONvsAPPR_time_before', bbox_inches='tight')
    #plt.show()

    # Plot phase planes
    fig, ax = plt.subplots(1,1)
    ax.plot(y_hat, yd_hat, 'b', label=r'$(\hat{y}, \, \hat{\dot{y}})$')
    ax.plot(y_RONsaved, yd_RONsaved, 'b--', label=r'RON $(y, \, \dot{y})$')
    ax.grid(True)
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$\dot{y}$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_folder/'RONvsAPPR_phaseplane_before', bbox_inches='tight')
    plt.show()
else:
    print('[simulation skipped]')

# Test RMSE on the test set before optimization
_, metrics = Loss(
    params_optimiz=params_optimiz, 
    data_batch=test_set,
)
RMSE = onp.sqrt(metrics["MSE"])
pred = onp.array(metrics["predictions"])
labels = onp.array(metrics["labels"])
print(f'Test accuracy: RMSE = {RMSE:.6e}')
print(f'Example:\n'
      f'    (y, yd) = ({onp.array(test_set["y"][69])}, {onp.array(test_set["yd"][69])})\n'
      f'    prediction: ydd_hat = {pred[69]}\n'
      f'    label: ydd = {labels[69]}\n'
      f'    |error|: {onp.abs( labels[69] - pred[69] )}'
)

############################################################################
##### SOME CHECKS ##########################################################
"""
# !! Check (jitted) loss computation !!
start = time.perf_counter() # warm-up
loss, metrics = Loss(
    params_optimiz,
    test_set,
)
print(loss) 
end = time.perf_counter()
print(f'time (warmup): {end-start} s')

start = time.perf_counter()
loss, metrics = Loss(
    params_optimiz,
    test_set,
)
print(loss) 
end = time.perf_counter()
print(f'time (already compiled): {end-start} s')
#exit()
# !! End check !!

#########

# !! Check loss + gradients computation !!
loss_and_grad = jax.jit(jax.value_and_grad(Loss, argnums=(0,), has_aux=True))
start = time.perf_counter()  # warm-up
(loss, metrics), grads = loss_and_grad(
    params_optimiz,
    test_set,
)
print(loss, grads)
end = time.perf_counter()
print(f'time (warmup): {end-start} s')

start = time.perf_counter()
(loss, metrics), grads = loss_and_grad(
    params_optimiz,
    test_set,
)
print(loss, grads)
end = time.perf_counter()
print(f'time (already compiled): {end-start} s')
exit()
# !! End check !!
"""
############################################################################
############################################################################


# =====================================================
# Optimization
# =====================================================

if True:
    print(F'\n--- OPTIMIZATION ---')

    # Optimization parameters
    n_iter = 250 # number of epochs
    batch_size = 2**8

    batches_per_epoch = batch_indx_generator(key, train_size, batch_size).shape[0]

    # Setup optimizer
    lr = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=1e-3,
        warmup_steps=20*batches_per_epoch,
        decay_steps=n_iter*batches_per_epoch,
        end_value=1e-6
    )
    optimizer = optax.adam(learning_rate=lr)
    optimiz_state = optimizer.init(params_optimiz) # initialize optimizer

    # Optimization iterations
    start = time.perf_counter()
    if use_scan:
        key, subkey = jax.random.split(key)
        results = train_with_scan(
            key=subkey,
            optimizer=optimizer,
            params_optimiz=params_optimiz,
            loss_fn=Loss,
            train_set=train_set,
            val_set=val_set,
            n_iter=n_iter,
            batch_size=batch_size,
        )
        params_optimiz = results["params_optimiz"]
        train_loss_ts = results["train_loss_ts"]
        train_MSE_ts = results["train_MSE_ts"]
        val_loss_ts = results["val_loss_ts"]
        val_MSE_ts = results["val_MSE_ts"]
    else:
        train_loss_ts = onp.zeros(n_iter)
        val_loss_ts = onp.zeros(n_iter)
        train_MSE_ts = onp.zeros(n_iter)
        val_MSE_ts = onp.zeros(n_iter)
        for epoch in tqdm(range(n_iter), 'Training'): # for each epoch...
            p1_print, p2_print = p1, p2

            # shuffle train dataset
            key, subkey = jax.random.split(key)
            batch_ids = batch_indx_generator(key=subkey, dataset_size=train_size, batch_size=batch_size)

            # perform training
            train_loss_sum = 0
            train_MSE_sum = 0
            grads_old = []
            for i in tqdm(range(len(batch_ids)), 'Current epoch', leave=False): # for each batch...
                batch_i_ids = batch_ids[i]
                train_batch = extract_batch(train_set, batch_i_ids)
                # perform optimization step
                params_optimiz_new, optimiz_state, loss, grads, train_metrics = train_step(
                    loss_fn=Loss,
                    optimiz_state=optimiz_state,
                    optimiz_update=optimizer.update,
                    params_optimiz=params_optimiz,
                    train_batch=train_batch,
                )
                # check for nan
                if onp.isnan(loss):
                    n_iter = epoch
                    faulty_grads = grads_old
                    p1_print, p2_print = params_optimiz
                    break
                grads_old = grads
                params_optimiz = params_optimiz_new
                train_loss_sum += loss                # sum train loss
                train_MSE_sum += train_metrics["MSE"] # sum train MSE

            # exit outer loop if nan
            if onp.isnan(loss):
                tqdm.write(
                    f"Epoch {epoch:02d} | "
                    f"NaN loss detected! | "
                    f"Last p1={p1_print} | "
                    f"Last p2={p2_print} | "
                    f"gradients={faulty_grads}"
                )
                break
            # compute mean training loss
            train_loss_epoch = train_loss_sum / len(batch_ids)
            train_MSE_epoch = train_MSE_sum / len(batch_ids)

            # perform validation
            p1, p2 = params_optimiz
            val_loss_epoch, val_metrics = Loss(
                params_optimiz=params_optimiz, 
                data_batch=val_set,
            )
            
            # print progress and save losses
            tqdm.write(
                f"Epoch {epoch:02d} | "
                f"p1={p1_print} | "
                f"p2={p2_print} | "
                f"train loss={train_loss_epoch:.3e} | "
                f"val loss={val_loss_epoch:.3e}"
            )
            train_loss_ts[epoch] = train_loss_epoch
            val_loss_ts[epoch] = val_loss_epoch
            train_MSE_ts[epoch] = train_MSE_epoch
            val_MSE_ts[epoch] = val_metrics["MSE"]

    jax.block_until_ready(params_optimiz)
    end = time.perf_counter()
    print(f'Elapsed time (optimization): {end-start} s')

    # Print optimal parameters (and save them)
    params_optimiz_opt = params_optimiz
    p1_opt, p2_opt = params_optimiz_opt
    print(f'p1_opt={p1_opt}\n'
        f'p2_opt={p2_opt}'
    )
    # onp.savez(
    #     data_folder/'optimal_data.npz', 
    #     p1=p1, 
    #     p2=p2, 
    # )

    # Visualization
    fig, ax1 = plt.subplots()

    train_loss_line, = ax1.plot(range(n_iter), train_loss_ts[:n_iter], 'r', label='train loss')
    val_loss_line, = ax1.plot(onp.arange(1,n_iter+1), val_loss_ts[:n_iter], 'b', label='validation loss')
    train_MSE_line, = ax1.plot(range(n_iter), train_MSE_ts[:n_iter], 'r--', label='train MSE')
    val_MSE_line, = ax1.plot(onp.arange(1,n_iter+1), val_MSE_ts[:n_iter], 'b--', label='validation MSE')
    ax1.set_yscale('log')
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('loss', color='k')
    ax1.tick_params(axis='y', labelcolor='k')

    ax2 = ax1.twinx()
    lr_line, = ax2.plot(range(n_iter), [lr(i*batches_per_epoch) for i in range(n_iter)], linewidth=0.5, label='learning rate', color='gray')
    ax2.set_ylabel('lr', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    lines = [train_loss_line, val_loss_line, train_MSE_line, val_MSE_line, lr_line]
    labels = ['train loss', 'validation loss', 'train MSE', 'validation MSE', 'learning rate']
    ax1.legend(lines, labels, loc='center right')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('Loss curve')
    plt.figtext(
        0.5, -0.05, 
        f"p1_opt={p1_opt}\n"
        f"p2_opt={p2_opt}",
        ha="center", va="top"
    )
    plt.tight_layout()
    plt.savefig(plots_folder/'Loss', bbox_inches='tight')
    plt.show()


# =====================================================
# Approximator after optimization
# =====================================================
print('\n--- AFTER OPTIMIZATION ---')

# # Load optimal parameters
# data_opt = onp.load(data_folder/'optimal_data.npz')
# p1_opt = data_opt['p1']
# p2_opt = data_opt['p2']

params_optimiz_opt = (p1_opt, p2_opt)

# If required, simulate final approximator
if show_simulations:
    print('Simulating...')
    start = time.perf_counter()
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, z, args: approximator_fd(z, params_optimiz_opt)),
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=z0,
        solver=solver,
        stepsize_controller=step_size,
        max_steps=max_steps,
        saveat=diffrax.SaveAt(ts=saveat),
    )
    end = time.perf_counter()
    print(f'Elapsed time (simulation): {end-start} s')

    z_hat = solution.ys
    y_hat, yd_hat = jnp.split(z_hat, 2, axis=1)

    # Plot y(t) and y_hat(t)
    fig, ax = plt.subplots(1,1)
    ax.plot(saveat, y_hat, 'b', label=r'$\hat{y}(t)$')
    ax.plot(time_RONsaved, y_RONsaved, 'b--', label=r'$y_{RON}(t)$')
    ax.grid(True)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('y')
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_folder/'RONvsAPPR_time_after', bbox_inches='tight')
    #plt.show()

    # Plot phase planes
    fig, ax = plt.subplots(1,1)
    ax.plot(y_hat, yd_hat, 'b', label=r'$(\hat{y}, \, \hat{\dot{y}})$')
    ax.plot(y_RONsaved, yd_RONsaved, 'b--', label=r'RON $(y, \, \dot{y})$')
    ax.grid(True)
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$\dot{y}$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_folder/'RONvsAPPR_phaseplane_after', bbox_inches='tight')
    plt.show()
else:
    print('[simulation skipped]')

# Test RMSE on the test set after optimization
_, metrics = Loss(
    params_optimiz=params_optimiz_opt, 
    data_batch=test_set,
)
RMSE = onp.sqrt(metrics["MSE"])
pred = onp.array(metrics["predictions"])
labels = onp.array(metrics["labels"])
print(f'Test accuracy: RMSE = {RMSE:.6e}')
print(f'Example:\n'
      f'    (y, yd) = ({onp.array(test_set["y"][69])}, {onp.array(test_set["yd"][69])})\n'
      f'    prediction: ydd_hat = {pred[69]}\n'
      f'    label: ydd = {labels[69]}\n'
      f'    |error|: {onp.abs( labels[69] - pred[69] )}'
)