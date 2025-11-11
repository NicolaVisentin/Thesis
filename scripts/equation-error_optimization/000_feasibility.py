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
main_folder = curr_folder.parent.parent                           # main folder "codes"
plots_folder = main_folder/'plots and videos'/Path(__file__).stem # folder for plots and videos
dataset_folder = main_folder/'datasets'                           # folder with the dataset
saved_data_folder = main_folder/'saved data'                      # folder for saved data

plots_folder.mkdir(parents=True, exist_ok=True)


# =====================================================
# Script settings
# =====================================================
use_scan = False         # choose whether to use normal for loop or lax.scan


# =====================================================
# Functions for optimization
# =====================================================

# NN approximator
def init_mlp_params(key, sizes):
    """Initialize MLP parameters as list of (W, b)."""
    keys = jax.random.split(key, len(sizes) - 1)
    params = []
    for k, (m, n) in zip(keys, zip(sizes[:-1], sizes[1:])):
        W = jax.random.normal(k, (n, m)) * jnp.sqrt(2.0 / m)
        b = jnp.zeros((n,))
        params.append((W, b))
    return params


def mlp_forward(params, x):
    """Forward pass of MLP."""
    for W, b in params[:-1]:
        x = jnp.tanh(W @ x + b)
    W, b = params[-1]
    return W @ x + b  # output layer (no activation)

@jax.jit
def approximator_fd(t, z, params):
    """NN-based approximator of the dynamics."""
    y, yd = jnp.split(z, 2)
    inp = jnp.concatenate([y, yd], axis=0)
    ydd_hat = mlp_forward(params, inp)
    zd = jnp.concatenate([yd, ydd_hat])
    return zd

# Loss function
@jax.jit
def Loss(
        params_optimiz : tuple, 
        data_batch : dict, 
) -> tuple[float, dict]:
    # extract everything
    y_batch, yd_batch, ydd_batch = data_batch["y"], data_batch["yd"], data_batch["ydd"]

    # predictions
    z = jnp.concatenate([y_batch, yd_batch], axis=1)

    forward_dynamics_vmap = jax.vmap(approximator_fd, in_axes=(None,0,None))
    zd = forward_dynamics_vmap(0, z, params_optimiz)
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

# Loss and grads function
loss_and_grads = jax.jit(jax.value_and_grad(Loss, argnums=0, has_aux=True))

# Optimization train step
@partial(jax.jit, static_argnums=(1,))
def train_step(
    optimiz_state,
    optimiz_update,
    params_optimiz : tuple,
    train_batch : dict,
):
    # compute loss and gradients
    (loss, metrics), grads = loss_and_grads(params_optimiz, train_batch)
    
    # update optimization parameters
    updates, optimiz_state = optimiz_update(grads, optimiz_state, params_optimiz)
    params_optimiz = optax.apply_updates(params_optimiz, updates)

    return params_optimiz, optimiz_state, loss, grads, metrics


# =====================================================
# Prepare datasets
# =====================================================

# Load dataset: m data from a RON with n_ron oscillators
dataset = onp.load(dataset_folder/'soft robot optimization/dataset_feasibility_1.npz')
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
# Robot before optimization
# =====================================================
print('--- BEFORE OPTIMIZATION ---')

# First guess of the parameters
key, subkey = jax.random.split(key)
mlp_sizes = [2, 16, 16, 1]  # [input, hidden1, hidden2, output]
params_optimiz = init_mlp_params(subkey, mlp_sizes)

# Load simulation results from RON
RON_evolution_data = onp.load(saved_data_folder/'RON_evolution_feasibility_1a.npz')
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

# Simulate robot
print('Simulating...')
start = time.perf_counter()
solution = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, z, args: approximator_fd(t, z, params_optimiz)),
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
plt.savefig(plots_folder/'RONvsPCS_time_before', bbox_inches='tight')
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
plt.savefig(plots_folder/'RONvsPCS_phaseplane_before', bbox_inches='tight')
plt.show()

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
        # Inner loop function (train iterating through batches within an epoch)
        @partial(jax.jit, static_argnums=(4,5))
        def train_one_epoch(key, optimiz_state, params_optimiz, train_set, batch_size, train_size):
            # get indices to extract shuffled batches
            key, subkey = jax.random.split(key)
            batch_ids = batch_indx_generator(key=subkey, dataset_size=train_size, batch_size=batch_size)

            # step function for the batches
            def batch_step(carry, batch_i_ids):
                optimiz_state, params_optimiz = carry
                # extract a random batch
                train_batch = extract_batch(train_set, batch_i_ids)
                # update parameters
                params_optimiz, optimiz_state, loss, _, train_metrics = train_step(
                    optimiz_state=optimiz_state,
                    optimiz_update=optimizer.update,
                    params_optimiz=params_optimiz,
                    train_batch=train_batch,
                )
                return (optimiz_state, params_optimiz), (loss, train_metrics["MSE"])

            # run scan on the batches to complete one epoch
            (optimiz_state, params_optimiz), (loss_vec, MSE_vec) = jax.lax.scan(
                batch_step,
                (optimiz_state, params_optimiz),
                batch_ids
            )

            # compute train loss for this epoch
            train_loss_epoch = jnp.mean(loss_vec)
            train_MSE_epoch = jnp.mean(MSE_vec)

            return key, optimiz_state, params_optimiz, train_loss_epoch, train_MSE_epoch

        # Outer loop (iterate epochs)
        def epoch_step(carry, _):
            key, optimiz_state, params_optimiz = carry
            # run inner loop (perform training)
            key, optimiz_state, params_optimiz, train_loss_epoch, train_MSE_epoch = train_one_epoch(
                key, optimiz_state, params_optimiz,
                train_set,
                batch_size, train_size
            )
            # perform validation after training on the epoch
            val_loss_epoch, val_metrics = Loss(
                params_optimiz=params_optimiz,
                data_batch=val_set,
            )
            return (key, optimiz_state, params_optimiz), (train_loss_epoch, train_MSE_epoch, val_loss_epoch, val_metrics["MSE"])

        # Run scan on outer loop
        (_, _, params_optimiz), (train_loss_ts, train_MSE_ts, val_loss_ts, val_MSE_ts) = jax.lax.scan(
            epoch_step,
            (key, optimiz_state, params_optimiz),
            xs=None,
            length=n_iter
        )

    else:
        train_loss_ts = onp.zeros(n_iter)
        val_loss_ts = onp.zeros(n_iter)
        train_MSE_ts = onp.zeros(n_iter)
        val_MSE_ts = onp.zeros(n_iter)
        for epoch in tqdm(range(n_iter), 'Training'): # for each epoch...
            # shuffle train dataset
            key, subkey = jax.random.split(key)
            batch_ids = batch_indx_generator(key=subkey, dataset_size=train_size, batch_size=batch_size)

            # perform training
            train_loss_sum = 0
            train_MSE_sum = 0
            for i in tqdm(range(len(batch_ids)), 'Current epoch', leave=False): # for each batch...
                batch_i_ids = batch_ids[i]
                train_batch = extract_batch(train_set, batch_i_ids)
                # perform optimization step
                params_optimiz_new, optimiz_state, loss, grads, train_metrics = train_step(
                    optimiz_state=optimiz_state,
                    optimiz_update=optimizer.update,
                    params_optimiz=params_optimiz,
                    train_batch=train_batch,
                )
                # check for nan
                if onp.isnan(loss):
                    n_iter = epoch
                    break
                params_optimiz = params_optimiz_new
                train_loss_sum += loss                # sum train loss
                train_MSE_sum += train_metrics["MSE"] # sum train MSE

            # exit outer loop if nan
            if onp.isnan(loss):
                tqdm.write(
                    f"Epoch {epoch:02d} | "
                    f"NaN loss detected!"
                )
                break
            # compute mean training loss
            train_loss_epoch = train_loss_sum / len(batch_ids)
            train_MSE_epoch = train_MSE_sum / len(batch_ids)

            # perform validation
            val_loss_epoch, val_metrics = Loss(
                params_optimiz=params_optimiz, 
                data_batch=val_set,
            )
            
            # print progress and save losses
            tqdm.write(
                f"Epoch {epoch:02d} | "
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
    plt.tight_layout()
    plt.savefig(plots_folder/'Loss', bbox_inches='tight')
    plt.show()

# =====================================================
# Robot after optimization
# =====================================================
print('\n--- AFTER OPTIMIZATION ---')

# Simulate final approximator
print('Simulating...')
start = time.perf_counter()
solution = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, z, args: approximator_fd(t, z, params_optimiz_opt)),
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
plt.savefig(plots_folder/'RONvsPCS_time_after', bbox_inches='tight')
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
plt.savefig(plots_folder/'RONvsPCS_phaseplane_after', bbox_inches='tight')
plt.show()

# Test RMSE on the test set before optimization
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