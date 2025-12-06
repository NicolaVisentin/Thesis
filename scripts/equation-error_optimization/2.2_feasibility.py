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
from diffrax import Tsit5, ConstantStepSize

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.ticker import MaxNLocator

from pathlib import Path
from tqdm import tqdm
import time
import sys

from soromox.systems.my_systems import PlanarPCS_simple
from soromox.systems.system_state import SystemState

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

# Functions for plotting
def draw_robot(
        robot: PlanarPCS_simple, 
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
    robot: PlanarPCS_simple,
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
show_simulations = True # choose whether to perform time simulations of the approximator (and comparison with RON)


# =====================================================
# Functions for optimization
# =====================================================

# Loss function
@jax.jit
def Loss(
        params_optimiz : Sequence, 
        data_batch : Dict, 
) -> Tuple[float, Dict]:
    """
    Computes loss function over a batch of data for certain parameters. In this case:
    Takes a batch of datapoints (y, yd), computes the forward dynamics ydd_hat = f_approximator(y,yd)
    and computes the loss as the MSE in the batch between predictions ydd_hat and labels ydd. In particular,
    our approximator here is: ydd_hat = qdd_hat = f_pcs(y,yd,tau(y)), with tau(y) a certain feedback control law. 

    !! This works only for n_ron = 3*n_pcs !!

    Args
    ----
    params_optimiz : Sequence
        Parameters for the opimization. In this case a list with:
        - **phi**: Tuple with pcs params (L_softplus, D_softplus). L.shape=(n_pcs,), D.shape=(3*n_pcs,)
        - **Phi**: Tuple with controller params.
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

    phi, Phi = params_optimiz

    L_softplus, D_softplus = phi
    W, b = Phi

    # convert parameters
    L = jax.nn.softplus(L_softplus)
    D = jnp.diag(jax.nn.softplus(D_softplus))

    # update robot
    robot_updated = robot.update_params({"L": L, "D": D})

    # predictions
    q_batch = y_batch
    qd_batch = yd_batch
    z_batch = jnp.concatenate([q_batch, qd_batch], axis=1) # state z=[q^T, qd^T]. Shape (batch_size, 2*3*n_pcs)
    tau_batch = jnp.tanh( q_batch @ W.T + b )
    actuation_arg = (tau_batch,)

    forward_dynamics_vmap = jax.vmap(robot_updated.forward_dynamics, in_axes=(None,0,0))
    zd_batch = forward_dynamics_vmap(0, z_batch, actuation_arg) # state derivative zd=[qd^T, qdd^T]. Shape (batch_size, 2*3*n_pcs)
    _, qdd_batch = jnp.split(zd_batch, 2, axis=1) 

    # compute loss (compare predictions ydd_hat=qdd with labels ydd)
    ydd_hat_batch = qdd_batch
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
dataset = onp.load(dataset_folder/'soft robot optimization/dataset_m1e5_N6_simplified.npz') # ! 6 decoupled oscillators
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
# Approximator before optimization
# =====================================================
print('--- BEFORE OPTIMIZATION ---')

# Initialize parameters
n_pcs = 2
key, key_W, key_b = jax.random.split(key, 3)

L0 = jnp.array([1.0e-1, 1.0e-1])
D0 = jnp.diag(jnp.array([1.0e-4, 1.0e-1, 1.0e-1,
                         1.0e-4, 1.0e-1, 1.0e-1]))
W0 = jax.random.uniform(key_W, shape=(3*n_pcs, 3*n_pcs), minval=-1e-6, maxval=1e-6)
b0 = jax.random.uniform(key_b, shape=(3*n_pcs,), minval=-1e-6, maxval=1e-6)

L_softplus = InverseSoftplus(L0)
D_softplus = InverseSoftplus(jnp.diag(D0))

phi = (L_softplus, D_softplus)
Phi = (W0, b0)
params_optimiz = [phi, Phi]

# Initialize robot
parameters = {
    "th0": jnp.array(jnp.pi/2),
    "L": L0,
    "r": jnp.array([2e-2, 2e-2]),
    "rho": jnp.array([1070, 1070]),
    "g": jnp.array([0.0, 9.81]), # !! gravity UP !!
    "E": jnp.array([2e3, 2e3]),
    "G": 1e3 * jnp.ones((n_pcs,)),
    "D": D0
}

robot = PlanarPCS_simple(
    num_segments = n_pcs,
    params = parameters,
    order_gauss = 5
)

# If required, simulate robot and compare its behaviour in time with the RON's one
if show_simulations:
    # Load simulation results from RON
    RON_evolution_data = onp.load(saved_data_folder/'RON_evolution_N6_simplified_a.npz')
    time_RONsaved = jnp.array(RON_evolution_data['time'])
    y_RONsaved = jnp.array(RON_evolution_data['y'])
    yd_RONsaved = jnp.array(RON_evolution_data['yd'])

    # Define controller function
    def tau_law(system_state, W, b):
        """Implements user-defined, time-varying, feedback control tau(t) = tanh( W*q(t) + b )."""
        q, _ = jnp.split(system_state.y, 2)
        tau = jnp.tanh( W @ q + b )
        return tau, None
    tau_fb = jax.jit(partial(tau_law, W=W0, b=b0)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox

    # Simulation parameters
    t0 = time_RONsaved[0]
    t1 = time_RONsaved[-1]
    dt = 1e-4
    saveat = np.arange(t0, t1, 1e-2)
    solver = Tsit5()
    step_size = ConstantStepSize()
    max_steps = int(1e6)

    # Simulate robot
    q0 = y_RONsaved[0]
    qd0 = yd_RONsaved[0]
    initial_state = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))

    print('Simulating robot...')
    start = time.perf_counter()
    sim_out = robot.rollout_closed_loop_to(
        initial_state = initial_state,
        controller = tau_fb,
        t1 = t1, 
        solver_dt = dt, 
        save_ts = saveat,
        solver = solver,
        stepsize_controller = step_size,
        max_steps = max_steps
    )
    end = time.perf_counter()
    print(f'Elapsed time (simulation): {end-start} s')

    # Extract results
    timePCS = sim_out.t
    q_PCS, qd_PCS = jnp.split(sim_out.y, 2, axis=1)

    y_hat = q_PCS
    yd_hat = qd_PCS
    
    # Plot y(t) and y_hat(t)
    fig, axs = plt.subplots(3,2, figsize=(12,9))
    for i, ax in enumerate(axs.flatten()):
        ax.plot(timePCS, y_hat[:,i], 'b', label=r'$\hat{y}(t)$')
        ax.plot(time_RONsaved, y_RONsaved[:,i], 'b--', label=r'$y_{RON}(t)$')
        ax.grid(True)
        ax.set_xlabel('t [s]')
        ax.set_ylabel('y, q')
        ax.set_title(f'Component {i+1}')
        ax.legend()
    plt.tight_layout()
    plt.savefig(plots_folder/'RONvsPCS_time_before', bbox_inches='tight')
    #plt.show()

    # Plot phase planes
    fig, axs = plt.subplots(3,2, figsize=(12,9))
    for i, ax in enumerate(axs.flatten()):
        lin_pcs = ax.plot(y_hat[:, i], yd_hat[:, i], 'b', label=r'PCS $(\hat{y}, \, \hat{\dot{y}})$')
        lin_ron = ax.plot(y_RONsaved[:, i], yd_RONsaved[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
        ax.grid(True)
        ax.set_xlabel(r'$y$')
        ax.set_ylabel(r'$\dot{y}$')
        ax.set_title(f'Component {i+1}')
        ax.legend()
    plt.tight_layout()
    plt.savefig(plots_folder/'RONvsPCS_phaseplane_before', bbox_inches='tight')
    #plt.show()

    # Animate robot
    animate_robot_matplotlib(
        robot = robot,
        t_list = saveat,
        q_list = q_PCS,
        interval = 1e-2, 
        slider = True,
        animation = False,
        show = True
    )
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
      f'    error: |e| = {onp.abs( labels[69] - pred[69] )}'
)

############################################################################
##### SOME CHECKS ##########################################################
"""
# !! Check (jitted) loss computation !!
print('\n\nCHECK: (jitted) loss computation')
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
print('\n\nCHECK: (jitted) loss + gradients computation')
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
    n_iter = 200 # number of epochs
    batch_size = 2**6

    key, subkey = jax.random.split(key)
    batches_per_epoch = batch_indx_generator(subkey, train_size, batch_size).shape[0]

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
            optimiz_state=optimiz_state,
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
                    loss_fn=Loss,
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

    phi_opt, Phi_opt = params_optimiz_opt
    L_softplus_opt, D_softplus_opt = phi_opt
    W_opt, b_opt = Phi_opt

    L_opt = jax.nn.softplus(L_softplus_opt)
    D_opt = jnp.diag(jax.nn.softplus(D_softplus_opt))

    print(f'L_opt={L_opt}\n'
        f'D_opt={onp.diag(D_opt)}\n'
        f'W_opt={W_opt}\n'
        f'b_opt={b_opt}')

    # onp.savez(
    #     data_folder/'optimal_data.npz', 
    #     L=onp.array(L_opt), 
    #     D=onp.array(D_opt),
    #     W=onp.array(W_opt),
    #     b=onp.array(b_opt),
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
        f"L_opt={onp.array(L_opt)} m\n"
        f"D_opt={onp.diag(D_opt)} Pa*s\n"
        f"W_opt={W_opt}\n"
        f"b_opt={b_opt}",
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
# L_opt = jnp.array(data_opt['L'])
# D_opt = jnp.array(data_opt['D'])
# W_opt = jnp.array(data_opt['W'])
# b_opt = jnp.array(data_opt['b'])
# L_softplus_opt = InverseSoftplus(L_opt)
# D_softplus_opt = InverseSoftplus(jnp.diag(D_opt))
# phi_opt = (L_softplus_opt, D_softplus_opt)
# Phi_opt = (W_opt, b_opt)
# params_optimiz_opt = [phi_opt, Phi_opt]

# Update robot with optimal parameters
robot_opt = robot.update_params({"L": L_opt, "D": D_opt})
tau_fb_opt = jax.jit(partial(tau_law, W=W_opt, b=b_opt))

# If required, simulate robot and compare its behaviour in time with the RON's one
if show_simulations:
    # Simulate robot
    print('Simulating robot...')
    start = time.perf_counter()
    sim_out = robot_opt.rollout_closed_loop_to(
        initial_state = initial_state,
        controller = tau_fb_opt,
        t1 = t1, 
        solver_dt = dt, 
        save_ts = saveat,
        solver = solver,
        stepsize_controller = step_size,
        max_steps = max_steps
    )
    end = time.perf_counter()
    print(f'Elapsed time (simulation): {end-start} s')

    # Extract results
    timePCS = sim_out.t
    q_PCS, qd_PCS = jnp.split(sim_out.y, 2, axis=1)

    y_hat = q_PCS
    yd_hat = qd_PCS
    
    # Plot y(t) and y_hat(t)
    fig, axs = plt.subplots(3,2, figsize=(12,9))
    for i, ax in enumerate(axs.flatten()):
        ax.plot(timePCS, y_hat[:,i], 'b', label=r'$\hat{y}(t)$')
        ax.plot(time_RONsaved, y_RONsaved[:,i], 'b--', label=r'$y_{RON}(t)$')
        ax.grid(True)
        ax.set_xlabel('t [s]')
        ax.set_ylabel('y, q')
        ax.set_title(f'Component {i+1}')
        ax.legend()
    plt.tight_layout()
    plt.savefig(plots_folder/'RONvsPCS_time_after', bbox_inches='tight')
    #plt.show()

    # Plot phase planes
    fig, axs = plt.subplots(3,2, figsize=(12,9))
    for i, ax in enumerate(axs.flatten()):
        lin_pcs = ax.plot(y_hat[:, i], yd_hat[:, i], 'b', label=r'PCS $(\hat{y}, \, \hat{\dot{y}})$')
        lin_ron = ax.plot(y_RONsaved[:, i], yd_RONsaved[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
        ax.grid(True)
        ax.set_xlabel(r'$y$')
        ax.set_ylabel(r'$\dot{y}$')
        ax.set_title(f'Component {i+1}')
        ax.legend()
    plt.tight_layout()
    plt.savefig(plots_folder/'RONvsPCS_phaseplane_after', bbox_inches='tight')
    #plt.show()

    # Animate robot
    animate_robot_matplotlib(
        robot = robot_opt,
        t_list = saveat,
        q_list = q_PCS,
        interval = 1e-2, 
        slider = True,
        animation = False,
        show = True
    )
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
      f'    error: |e| = {onp.abs( labels[69] - pred[69] )}'
)