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
from jax import Array
import optax
from diffrax import Tsit5, Euler, Heun, Midpoint, Ralston, Bosh3, Dopri5, Dopri8, ImplicitEuler, Kvaerno3
from diffrax import ConstantStepSize, PIDController

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.ticker import MaxNLocator
import pickle

from pathlib import Path
from tqdm import tqdm
import time
import sys
import select

from soromox.systems.planar_pcs import PlanarPCS
from soromox.systems.planar_pcs_simplified import PlanarPCS_simple
from soromox.utils.lie_algebra.se2 import exp_SE2

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
plots_folder.mkdir(parents=True, exist_ok=True)

dataset_folder = main_folder/'datasets'                           # folder with the dataset

# data_folder = main_folder/'saved data'/Path(__file__).stem        # folder for saving data
# data_folder.mkdir(parents=True, exist_ok=True)

# Functions for plotting
def draw_robot(
        robot: PlanarPCS | PlanarPCS_simple, 
        q: Array, 
        num_points: int = 50
):
    L_max = jnp.sum(robot.L)
    s_ps = jnp.linspace(0, L_max, num_points)

    chi_ps = robot.forward_kinematics_batched(q, s_ps)  # (N,3)
    curve = onp.array(chi_ps[:, 1:], dtype=onp.float64) # (N,2)
    pos_tip = curve[-1]                                 # [x_tip, y_tip]

    return curve, pos_tip

def animate_robot_matplotlib(
    robot: PlanarPCS | PlanarPCS_simple,
    t_list: Array,  # shape (T,)
    q_list: Array,  # shape (T, DOF)
    target: Array = None,
    num_points: int = 50,
    interval: int = 50,
    slider: bool = None,
    animation: bool = None,
    show: bool = True,
):
    if slider is None and animation is None:
        raise ValueError("Either 'slider' or 'animation' must be set to True.")
    if animation and slider:
        raise ValueError(
            "Cannot use both animation and slider at the same time. Choose one."
        )

    width = jnp.linalg.norm(robot.L) * 3
    height = width

    if target is not None:
        t_old = onp.linspace(0, 1, len(target))
        t_new = onp.linspace(0, 1, len(q_list))
        target = onp.interp(t_new, t_old, target)

    def draw_base(ax, robot, L=robot.L[0] / 2):
        angle1 = robot.th0 - jnp.pi / 2
        angle2 = robot.th0 + jnp.pi / 2
        x1, y1 = L * jnp.cos(angle1), L * jnp.sin(angle1)
        x2, y2 = L * jnp.cos(angle2), L * jnp.sin(angle2)
        ax.plot([x1, x2], [y1, y2], color="black", linestyle="-", linewidth=2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw_base(ax, robot, L=0.1)

    if animation:
        (line,) = ax.plot([], [], lw=4, color="blue")
        (tip,) = ax.plot([], [], 'ro', markersize=5)
        (targ,) = ax.plot([], [], color='r', alpha=0.5)
        ax.set_xlim(-width / 2, width / 2)
        ax.set_ylim(0, height)
        ax.grid(True)
        title_text = ax.set_title("t = 0.00 s")

        def init():
            line.set_data([], [])
            tip.set_data([], [])
            targ.set_data([], [])
            title_text.set_text("t = 0.00 s")
            return line, tip, targ, title_text

        def update(frame_idx):
            q = q_list[frame_idx]
            t = t_list[frame_idx]
            curve, tip_pos = draw_robot(robot, q, num_points)
            line.set_data(curve[:, 0], curve[:, 1])
            tip.set_data([tip_pos[0]], [tip_pos[1]])
            if target is not None:
                x_target = target[frame_idx]
                targ.set_data([x_target,x_target], [0,height])
            title_text.set_text(f"t = {t:.2f} s")
            return (line, tip, title_text) + ((targ,) if target is not None else ())

        ani = FuncAnimation(
            fig,
            update,
            frames=len(q_list),
            init_func=init,
            blit=False,
            interval=interval,
        )

    elif slider:

        def update_plot(frame_idx):
            ax.cla()  # Clear current axes
            ax.set_xlim(-width / 2, width / 2)
            ax.set_ylim(0, height)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_title(f"t = {t_list[frame_idx]:.2f} s")
            ax.grid(True)
            q = q_list[frame_idx]
            curve, tip_pos = draw_robot(robot, q, num_points)
            ax.plot(curve[:, 0], curve[:, 1], lw=4, color="blue")
            ax.plot([tip_pos[0]], [tip_pos[1]], 'ro', markersize=5)
            if target is not None:
                x_target = target[frame_idx]
                ax.plot([x_target,x_target], [0,height], 'r', alpha=0.5)
            fig.canvas.draw_idle()

        # Create slider
        ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])  # [left, bottom, width, height]
        slider = Slider(
            ax=ax_slider,
            label="Frame",
            valmin=0,
            valmax=len(t_list) - 1,
            valinit=0,
            valstep=1,
        )
        slider.on_changed(update_plot)

        update_plot(0)  # Initial plot

    if show:
        plt.show()

    plt.close(fig)


# =====================================================
# Script settings
# =====================================================

use_scan = False         # choose whether to use normal for loop or lax.scan
show_simulations = True # choose whether to show simulations of the robot (just for visualization)


# =====================================================
# Istantiate robot
# =====================================================
print('--- BEFORE OPTIMIZATION ---')

# Select planar PCS model (with or without Coriolis effect)
RobotModel = PlanarPCS_simple # PlanarPCS, PlanarPCS_simple

# Initialize robot
n_pcs = 2 # number of segments

L = jnp.array([1.0e-1, 1.0e-1])
D = jnp.diag(jnp.array([1.0e-4, 1.0e-1, 1.0e-1,
                        1.0e-4, 1.0e-1, 1.0e-1]))
parameters = {
    "th0": jnp.array(jnp.pi/2),
    "L": L,
    "r": jnp.array([2e-2, 2e-2]),
    "rho": jnp.array([1070, 1070]),
    "g": jnp.array([0.0, 9.81]), # !! gravity UP !!
    "E": jnp.array([2e3, 2e3]),
    "G": 1e3 * jnp.ones((n_pcs,)),
    "D": D
}

robot = RobotModel(
    num_segments = n_pcs,
    params = parameters,
    order_gauss = 5
)

if show_simulations:
    # Simulation parameters
    q0 = jnp.array([-1.0, 0.0, 0.0,
                    1.0, 0.0, 0.0]) # k, S_x, S_y
    qd0 = jnp.zeros_like(q0)
    u = jnp.zeros_like(q0)

    t0 = 0
    t1 = 2
    dt =1e-4
    saveat = jnp.linspace(t0, t1, 1001)
    solver = Tsit5() # Tsit5(), Euler(), Heun(), Midpoint(), Ralston(), Bosh3(), Dopri5(), Dopri8()
    #step_size = PIDController(rtol=1e-6, atol=1e-6, dtmin=1e-4, force_dtmin=True) # ConstantStepSize(), PIDController(rtol=, atol=)
    step_size = ConstantStepSize()
    max_steps = int(1e6)

    # Simulate robot
    print('Simulating robot...')
    start = time.perf_counter()
    ts, q_ts, _ = robot.resolve_upon_time(
        q0 = q0, 
        qd0 = qd0,
        u = u, 
        t0 = t0, 
        t1 = t1, 
        dt = dt, 
        saveat_ts = saveat,
        solver = solver,
        stepsize_controller = step_size,
        max_steps = max_steps
    )
    end = time.perf_counter()
    print(f'Elapsed time (simulation): {end-start} s')

    animate_robot_matplotlib(
        robot = robot,
        t_list = saveat,
        q_list = q_ts,
        interval = 1e-2, 
        slider = True,
        animation = False,
        show = True
    )
else:
    print('[simulation skipped]')


# =====================================================
# Functions for optimization
# =====================================================

# Loss function
@jax.jit
def Loss(
        params_softplus : Array, 
        A_flat : Array, 
        c : Array, 
        y_batch : Array,
        yd_batch : Array, 
        ydd_batch : Array, 
) -> tuple[float, dict]:
    """
    Loss function. Takes a batch of datapoints (y, yd), converts them into configurations (q, qd)
    of the robot as q = A*y + c, qd = A*yd, computes the forward dynamics qdd = forward_dynamics_robot(q, qd)
    and computes the loss as the MSE in the batch between predictions qdd and labels A*ydd.

    Args
    ----
    params_softplus : Array
        Raw parameters of the robot, as params_softplus = InverseSoftplus(params). In this case 
        lengths and damping coefficients of the segments. Shape (n_pcs+3n_pcs,)
    A_flat : Array
        Mapping matrix (flattened) between RON and robot configurations: q = A*y + c. Shape (3n_pcs*n_ron,)
        or (3n_pcs,)=(n_ron,) if diagonal.
    c : Array
        Mapping vector between RON and robot configurations: q = A*y + c. Shape (3n_pcs,)
    y_batch : Array
        Batch of datapoints y. Shape (batch_size, n_ron)
    yd_batch : Array
        Batch of datapoints yd. Shape (batch_size, n_ron)
    ydd_batch : Array
        Batch of labels ydd. Shape (batch_size, n_ron)

    Returns
    -------
    loss : float
        Scalar loss computed as MSE in the batch between predictions qdd and labels A*ydd.
    metrics : dict
        Dictionary of useful metrics.
    """
    # useful values
    n_pcs = int(len(c) / 3)
    n_ron = y_batch.shape[1]

    # get physical parameters of the robot
    params = jax.nn.softplus(params_softplus)
    L = params[:n_pcs]
    D = jnp.diag(params[n_pcs:])

    # update robot
    robot_updated = robot.update_params({"L": L, "D": D})

    # rebuild A matrix
    if len(A_flat) == n_ron:
        A = jnp.diag(A_flat)
    else:
        A = jnp.reshape(A_flat, (3*n_pcs, n_ron))

    # generate input configurations for the robot
    q_batch = y_batch @ jnp.transpose(A) + c # shape (batch_size, n_pcs)
    qd_batch = yd_batch @ jnp.transpose(A)   # shape (batch_size, n_pcs)

    # predictions
    z = jnp.concatenate([q_batch, qd_batch], axis=1) # state z=[q^T, qd^T]. Shape (batch_size, 2*n_pcs)

    forward_dynamics_vmap = jax.vmap(robot_updated.forward_dynamics, in_axes=(None,0))
    zd = forward_dynamics_vmap(0, z) # state derivative zd=[qd^T, qdd^T]. Shape (batch_size, 2*n_pcs)
    _, previsions_qdd_batch = jnp.split(zd, 2, axis=1) 

    # compute loss
    labels_qdd_batch = ydd_batch @ jnp.transpose(A) # convert labels from ydd to qdd
    MSE = jnp.mean(jnp.sum((previsions_qdd_batch - labels_qdd_batch)**2, axis=1))
    alpha = 1e3
    regulariz_A = alpha / (jnp.linalg.norm(A)**2 + 1e-6)
    loss = MSE + regulariz_A

    # store metrics
    metrics = {
        "MSE": MSE,
        "predictions": previsions_qdd_batch,
    }

    return loss, metrics


# Optimization train step
@partial(jax.jit, static_argnums=(1,))
def train_step(
    optimiz_state,
    optimiz_update,
    optimiz_params : tuple,
    y_batch : Array,
    yd_batch : Array,
    ydd_batch : Array,
):
    """
    Optimization step. Takes current optimization parameters and optimizer status and computes loss and gradients propagation
    in a batch of data, finally updates and returns the optimization parameters and optimizer status (and the loss/grads). 

    Args
    ----
    optimiz_state
        Optimizer state. It's an optax object.
    optimiz_update
        Optimizer update. It's an optax object.
    optimiz_params : tuple
        Collects all optimization parameters. In our case (robot_params_softplus, A_flat, c).
    y_batch : Array
        Batch of datapoints y. Shape (batch_size, n_ron)
    yd_batch : Array
        Batch of datapoints yd. Shape (batch_size, n_ron)
    ydd_batch : Array
        Batch of labels ydd. Shape (batch_size, n_ron)

    Returns
    -------
    optimiz_params_updated
        Updated optimization parameters.
    optimiz_state_updated
        Updated optimizer state.
    loss
        Loss value for this batch.
    grads
        Gradients.
    metrics
        Metrics dictionary for this batch.
    """
    # extract optimization parameters
    params_softplus, A_flat, c = optimiz_params

    # compute loss and gradients
    loss_and_grads = jax.jit(jax.value_and_grad(Loss, argnums=(0,1,2), has_aux=True))
    (loss, metrics), grads = loss_and_grads(params_softplus, A_flat, c, y_batch, yd_batch, ydd_batch)
    
    # update optimization parameters
    updates, optimiz_state = optimiz_update(grads, optimiz_state, optimiz_params)
    optimiz_params = optax.apply_updates(optimiz_params, updates)

    return optimiz_params, optimiz_state, loss, grads, metrics


# =====================================================
# Prepare datasets
# =====================================================

# Load dataset: m data from a RON with n_ron oscillators
dataset = onp.load(dataset_folder/'soft robot optimization/dataset1e4_y_yd_ydd.npz')
y = dataset["y"]     # position samples of the RON oscillators. Shape (m, n_ron)
yd = dataset["yd"]   # velocity samples of the RON oscillators. Shape (m, n_ron)
ydd = dataset["ydd"] # accelerations of the RON oscillators. Shape (m, n_ron)

# Convert into jax
y = jnp.array(y)
yd = jnp.array(yd)
ydd = jnp.array(ydd)

dataset = {
    "y": y,
    "yd": y,
    "ydd": y,
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
# Optimization
# =====================================================

# First guess of the parameters
params_robot = jnp.concatenate([L, jnp.diag(D)])
params_robot_softplus = InverseSoftplus(params_robot)
A_flat = jnp.diag(jnp.eye(6))
c = jnp.ones(6)

optimiz_params = (params_robot_softplus, A_flat, c)

# Test RMSE on the test set before optimization
_, metrics = Loss(
    params_softplus=params_robot_softplus, 
    A_flat=A_flat, 
    c=c, 
    y_batch=test_set["y"],
    yd_batch=test_set["yd"], 
    ydd_batch=test_set["ydd"],
)
RMSE = onp.sqrt(metrics["MSE"])
pred = onp.array(metrics["predictions"])
print(f'Test accuracy (before training): RMSE = {RMSE:.6e}')
print(f'Example:\n'
      f'    (y, yd) = ({onp.array(test_set["y"][69])}, {onp.array(test_set["yd"][69])})\n'
      f'    (q, qd) = (A*y+c, A*yd) = ({onp.array(test_set["y"][69]@onp.diag(A_flat).T+c)}, {onp.array(test_set["yd"][69]@onp.diag(A_flat).T)})\n'
      f'    prediction: qdd = {pred[69]}\n'
      f'    label: A*ydd = {onp.array(test_set["ydd"][69]@onp.diag(A_flat).T)}\n'
      f'    |error|: A*ydd = {onp.abs( onp.array(test_set["ydd"][69]@onp.diag(A_flat).T) - pred[69] )}'
)


############################################################################
##### SOME CHECKS ##########################################################
"""
# !! Check (jitted) loss computation !!
start = time.perf_counter() # warm-up
loss, metrics = Loss(
    params_robot_softplus,
    A_flat,
    c,
    train_set["y"],
    train_set["yd"],
    train_set["ydd"],
)
print(loss) 
end = time.perf_counter()
print(f'time (warmup): {end-start} s')

start = time.perf_counter()
loss, metrics = Loss(
    params_robot_softplus,
    A_flat,
    c,
    train_set["y"],
    train_set["yd"],
    train_set["ydd"],
)
print(loss) 
end = time.perf_counter()
print(f'time (already compiled): {end-start} s')
exit()
# !! End check !!

#########

# !! Check loss + gradients computation !!
loss_and_grad = jax.jit(jax.value_and_grad(Loss, argnums=(0,1,2), has_aux=True))
start = time.perf_counter()  # warm-up
(loss, metrics), grads = loss_and_grad(
    params_robot_softplus,
    A_flat,
    c,
    train_set["y"],
    train_set["yd"],
    train_set["ydd"],
)
print(loss, grads)
end = time.perf_counter()
print(f'time (warmup): {end-start} s')

start = time.perf_counter()
(loss, metrics), grads = loss_and_grad(
    params_robot_softplus,
    A_flat,
    c,
    train_set["y"],
    train_set["yd"],
    train_set["ydd"],
)
print(loss, grads)
end = time.perf_counter()
print(f'time (already compiled): {end-start} s')
exit()
# !! End check !!
"""
############################################################################
############################################################################
print(F'\n--- OPTIMIZATION ---')

# Setup optimizer
lr = optax.piecewise_constant_schedule(1e-12, {20: 10, 50: 10, 150: 10, 250: 10, 300: 10, 350: 10, 400: 10, 450: 10})
optimizer = optax.sgd(learning_rate=lr)
optimiz_state = optimizer.init(optimiz_params) # initialize optimizer

# Optimization iterations
n_iter = 120 # number of epochs
batch_size = 2**12

start = time.perf_counter()
if use_scan:
    # Inner loop function (train iterating through batches within an epoch)
    @partial(jax.jit, static_argnums=(6,7))
    def train_one_epoch(key, optimiz_state, optimiz_params, y_train, yd_train, ydd_train, batch_size, train_size):
        # get indices to extract shuffled batches
        key, subkey = jax.random.split(key)
        batch_ids = batch_indx_generator(key=subkey, dataset_size=train_size, batch_size=batch_size)

        # step function for the batches
        def batch_step(carry, batch_i_ids):
            optimiz_state, optimiz_params = carry
            # extract a random batch
            y_batch = y_train[batch_i_ids]
            yd_batch = yd_train[batch_i_ids]
            ydd_batch = ydd_train[batch_i_ids]
            # update parameters
            optimiz_params, optimiz_state, loss, _, _ = train_step(
                optimiz_state=optimiz_state,
                optimiz_update=optimizer.update,
                optimiz_params=optimiz_params,
                y_batch=y_batch,
                yd_batch=yd_batch,
                ydd_batch=ydd_batch,
            )
            return (optimiz_state, optimiz_params), loss

        # run scan on the batches to complete one epoch
        (optimiz_state, optimiz_params), loss_vec = jax.lax.scan(
            batch_step,
            (optimiz_state, optimiz_params),
            batch_ids
        )

        # compute train loss for this epoch
        train_loss_epoch = jnp.mean(loss_vec)

        return key, optimiz_state, optimiz_params, train_loss_epoch

    # Outer loop (iterate epochs)
    def epoch_step(carry, _):
        key, optimiz_state, optimiz_params = carry
        # run inner loop (perform training)
        key, optimiz_state, optimiz_params, train_loss_epoch = train_one_epoch(
            key, optimiz_state, optimiz_params,
            train_set["y"], train_set["yd"], train_set["ydd"],
            batch_size, train_size
        )
        # perform validation after training on the epoch
        params_robot_softplus, A_flat, c = optimiz_params
        val_loss_epoch, _ = Loss(
            params_softplus=params_robot_softplus,
            A_flat=A_flat,
            c=c,
            y_batch=val_set["y"],
            yd_batch=val_set["yd"],
            ydd_batch=val_set["ydd"],
        )
        return (key, optimiz_state, optimiz_params), (train_loss_epoch, val_loss_epoch)

    # Run scan on outer loop
    (_, _, optimiz_params), (train_loss_ts, val_loss_ts) = jax.lax.scan(
        epoch_step,
        (key, optimiz_state, optimiz_params),
        xs=None,
        length=n_iter
    )

else:
    train_loss_ts = onp.zeros(n_iter)
    val_loss_ts = onp.zeros(n_iter)
    for epoch in tqdm(range(n_iter), 'Training'): # for each epoch...
        params_robot_print = jax.nn.softplus(params_robot_softplus)
        A_flat_print = A_flat
        c_print = c
        # shuffle train dataset
        key, subkey = jax.random.split(key)
        batch_ids = batch_indx_generator(key=subkey, dataset_size=train_size, batch_size=batch_size)

        # perform training
        train_loss_sum = 0
        for i in tqdm(range(len(batch_ids)), 'Current epoch', leave=False): # for each batch...
            batch_i_ids = batch_ids[i]
            y_train_batch = train_set["y"][batch_i_ids]
            yd_train_batch = train_set["yd"][batch_i_ids]
            ydd_train_batch = train_set["ydd"][batch_i_ids]
            # perform optimization step
            optimiz_params_new, optimiz_state, loss, grads, _ = train_step(
                optimiz_state=optimiz_state,
                optimiz_update=optimizer.update,
                optimiz_params=optimiz_params,
                y_batch=y_train_batch,
                yd_batch=yd_train_batch,
                ydd_batch=ydd_train_batch,
            )
            # check for nan
            if onp.isnan(loss):
                n_iter = epoch
                faulty_grads = grads_old
                params_robot_softplus_print, A_flat_print, c_print = optimiz_params
                params_robot_print = jax.nn.softplus(params_robot_softplus_print)
                break
            grads_old = grads
            optimiz_params = optimiz_params_new
            train_loss_sum += loss # sum train loss

        # exit outer loop if nan
        if onp.isnan(loss):
            tqdm.write(
                f"Epoch {epoch:02d} | "
                f"NaN loss detected! | "
                f"Last L={params_robot_print[:n_pcs]} | "
                f"Last D={params_robot_print[n_pcs:]} | "
                f"Last A={onp.array2string(A_flat_print, precision=2, formatter={'float_kind': lambda x: f'{x:.2f}'})} | "
                f"Last c={onp.array2string(c_print, precision=2, formatter={'float_kind': lambda x: f'{x:.2f}'})} | "
                f"gradients={faulty_grads}"
            )
            break
        # compute mean training loss
        train_loss_epoch = train_loss_sum / len(batch_ids)

        # perform validation
        params_robot_softplus, A_flat, c = optimiz_params
        val_loss_epoch, _ = Loss(
            params_softplus=params_robot_softplus, 
            A_flat=A_flat, 
            c=c, 
            y_batch=val_set["y"],
            yd_batch=val_set["yd"], 
            ydd_batch=val_set["ydd"],
        )
        
        # print progress and save losses
        tqdm.write(
            f"Epoch {epoch:02d} | "
            f"L={params_robot_print[:n_pcs]} | "
            f"D={params_robot_print[n_pcs:]} | "
            f"A={onp.array2string(A_flat_print, precision=2, formatter={'float_kind': lambda x: f'{x:.2f}'})} | "
            f"c={onp.array2string(c_print, precision=2, formatter={'float_kind': lambda x: f'{x:.2f}'})} | "
            f"train loss={train_loss_epoch:.3e} | "
            f"val loss={val_loss_epoch:.3e}"
        )
        train_loss_ts[epoch] = train_loss_epoch
        val_loss_ts[epoch] = val_loss_epoch

jax.block_until_ready(optimiz_params)
end = time.perf_counter()
print(f'Elapsed time (optimization): {end-start} s')
    
# Print optimal parameters (and save them)
params_robot_softplus_opt, A_flat_opt, c_opt = optimiz_params
params_robot_opt = jax.nn.softplus(params_robot_softplus_opt)
L_opt = params_robot_opt[:n_pcs]
D_opt = jnp.diag(params_robot_opt[n_pcs:])
A_opt = jnp.diag(A_flat_opt)
print(f'L_opt={L_opt}\n'
      f'D_opt={onp.diag(D_opt)}\n'
      f'A_opt={onp.diag(A_opt)}\n'
      f'c_opt={c_opt}')

#onp.savez(data_folder/'optimal_data', L=onp.array(L_opt), D=onp.array(D_opt), A=onp.array(A_opt), c=onp.array(c_opt))

# Visualization
fig, ax1 = plt.subplots()

train_loss_line, = ax1.plot(range(n_iter), train_loss_ts[:n_iter], label='train loss')
val_loss_line, = ax1.plot(range(n_iter), val_loss_ts[:n_iter], label='validation loss')
ax1.set_yscale('log')
ax1.set_xlabel('iterations')
ax1.set_ylabel('loss', color='k')
ax1.tick_params(axis='y', labelcolor='k')

ax2 = ax1.twinx()
lr_line, = ax2.plot(range(n_iter), [lr(i) for i in range(n_iter)], linewidth=0.5, label='learning rate', color='gray')
ax2.set_ylabel('lr', color='gray')
ax2.set_yscale('log')
ax2.tick_params(axis='y', labelcolor='gray')

lines = [train_loss_line, val_loss_line, lr_line]
labels = ['train Loss', 'validation loss', 'learning rate']
ax1.legend(lines, labels, loc='center right')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title('Loss curve')
plt.figtext(0.5, -0.05, f"L_opt={onp.array(L_opt)} m\n D_opt={onp.diag(D_opt)} Pa*s\n A_opt={onp.diag(A_opt)}\n c_opt={onp.array(c_opt)}", ha="center", va="top")
plt.tight_layout()
plt.savefig(plots_folder/'Loss', bbox_inches='tight')
plt.show()


# =====================================================
# Robot after optimization
# =====================================================
print('\n--- AFTER OPTIMIZATION ---')

# # Load optimal parameters
# data_opt = onp.load(data_folder/'optimal_data.npz')
# L_opt = jnp.array(data_opt['L'])
# D_opt = jnp.array(data_opt['D'])
# A_opt = jnp.array(data_opt['A'])
# c_opt = jnp.array(data_opt['c'])

# Update robot with optimal parameters
robot_opt = robot.update_params({"L": L_opt, "D": D_opt})

if show_simulations:
    # Simulate robot
    print('Simulating robot...')
    _, q_ts, _ = robot_opt.resolve_upon_time(
        q0 = q0, 
        qd0 = qd0,
        u = u, 
        t0 = t0, 
        t1 = t1, 
        dt = dt, 
        saveat_ts = saveat,
        solver = solver,
        stepsize_controller = step_size,
        max_steps = max_steps
    )

    animate_robot_matplotlib(
        robot = robot_opt,
        t_list = saveat,
        q_list = q_ts,
        interval = 1e-2, 
        slider = True,
        animation = False,
        show = True
    )
else:
    print('[simulation skipped]')

# Test on the test dataset after optimization
_, metrics = Loss(
    params_softplus=params_robot_softplus_opt, 
    A_flat=A_flat_opt, 
    c=c_opt, 
    y_batch=test_set["y"],
    yd_batch=test_set["yd"], 
    ydd_batch=test_set["ydd"],
)
RMSE = onp.sqrt(metrics["MSE"])
pred = onp.array(metrics["predictions"])
print(f'Test accuracy (after training): RMSE = {RMSE:.6e}')
print(f'Example:\n'
      f'    (y, yd) = ({onp.array(test_set["y"][69])}, {onp.array(test_set["yd"][69])})\n'
      f'    (q, qd) = (A*y+c, A*yd) = ({onp.array(test_set["y"][69]@onp.diag(A_flat_opt).T+c_opt)}, {onp.array(test_set["yd"][69]@onp.diag(A_flat_opt).T)})\n'
      f'    pred: qdd = {pred[69]}\n'
      f'    lab: A*ydd = {onp.array(test_set["ydd"][69]@onp.diag(A_flat_opt).T)}\n'
      f'    |error|: A*ydd = {onp.abs( onp.array(test_set["ydd"][69]@onp.diag(A_flat_opt).T) - pred[69] )}'
)
