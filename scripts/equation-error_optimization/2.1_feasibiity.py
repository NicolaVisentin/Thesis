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
plots_folder = main_folder/'plots and videos'/curr_folder.stem/Path(__file__).stem/'T15' # folder for plots and videos
dataset_folder = main_folder/'datasets'                                            # folder with the dataset
data_folder = main_folder/'saved data'/curr_folder.stem/Path(__file__).stem/'T15'        # folder for saving data

data_folder.mkdir(parents=True, exist_ok=True)
plots_folder.mkdir(parents=True, exist_ok=True)

# Functions for plotting robot
def draw_robot(
        robot: PlanarPCS_simple, 
        q: Array, 
        num_points: int = 50
):
    L_max = jnp.sum(robot.L)
    s_ps = jnp.linspace(0, L_max, num_points)

    chi_ps = robot.forward_kinematics_batched(q, s_ps)  # (N,3)
    curve = jnp.array(chi_ps[:, 1:], dtype=jnp.float64) # (N,2)
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
    fps: float = None,
    duration: float = None,
    save_path: Path = None,
):
    if slider is None and animation is None:
        raise ValueError("Either 'slider' or 'animation' must be set to True.")
    if animation and slider:
        raise ValueError(
            "Cannot use both animation and slider at the same time. Choose one."
        )

    _, pos_tip_list = jax.vmap(draw_robot, in_axes=(None,0,None))(robot, q_list, num_points)
    width = onp.max(pos_tip_list)
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
        n_frames = len(q_list)
        default_fps = 1000 / interval
        default_duration = n_frames * (interval / 1000)
        if fps is None and duration is None:
            final_fps = default_fps
            final_duration = default_duration
            frame_skip = 1
        elif fps is not None and duration is None:
            final_fps = fps
            final_duration = n_frames / save_fps
            frame_skip = 1
        elif fps is None and duration is not None:
            final_duration = duration
            final_fps = n_frames / duration
            frame_skip = 1
        else:
            final_fps = fps
            final_duration = duration
            n_required_frames = int(final_duration * final_fps)
            n_required_frames = max(1, n_required_frames)
            frame_skip = max(1, n_frames // n_required_frames)

        frame_indices = list(range(0, n_frames, frame_skip))

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
            frames=frame_indices,
            init_func=init,
            blit=False,
            interval=1000/final_fps,
        )
        if save_path is not None:
            ani.save(save_path, writer=PillowWriter(fps=final_fps))

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
use_scan = True # choose whether to use normal for loop or lax.scan
show_simulations = True # choose whether to perform time simulations of the approximator (and comparison with RON)
coupled_case = False # use dataset of coupled RON network
reconstruction = 'ydd' # reconstruction loss on y and optionally on yd and ydd. Choose 'y', 'yd', or 'ydd'
"""
Choose controller to train. Possibilities are:
    'tanh_simple': u = tanh(W*q + b)
    'tanh_complete': u = tanh(W*z + b), where z = [q^T, qd^T]^T
    'mlp': u = MLP(q,qd)
    'none'
"""
controller_to_train = 'tanh_complete'


# =====================================================
# Functions for optimization
# =====================================================

# Loss function
@jax.jit
def Loss(
        params_optimiz : Sequence, 
        data_batch : Dict, 
        robot : PlanarPCS_simple,
        controller : MLP,
        encoder : MLP,
        decoder : MLP,
) -> Tuple[float, Dict]:
    """
    Computes loss function over a batch of data for certain parameters. In this case:
    Takes a batch of datapoints (y, yd), computes the forward dynamics ydd_hat = f_approximator(y,yd)
    and computes the loss as the MSE in the batch between predictions ydd_hat and labels ydd.

    Args
    ----
    params_optimiz : Sequence
        Parameters for the opimization. In this case a list with:
        - **p_robot**: Tuple with pcs params (L_raw, D_raw, r_raw, rho_raw, E_raw, G_raw).
        - **p_map**: Tuple (p_encoder, p_decoder). Each of them is a tuple with MLP encoder/decoder
                     parameters (layers). 
        - **p_controller**: Tuple with MLP controller parameters (layers).              
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
    n_pcs = robot.num_segments

    # extract everything
    y_batch = data_batch["y"]
    yd_batch = data_batch["yd"]
    ydd_batch = data_batch["ydd"]

    p_robot, p_map, p_controller = params_optimiz

    L_raw, D_raw, r_raw, rho_raw, E_raw, G_raw = p_robot
    p_encoder, p_decoder = p_map

    # convert parameters
    L = jax.nn.softplus(L_raw)
    D = jnp.diag(jax.nn.softplus(D_raw))
    r = jax.nn.softplus(r_raw) * jnp.ones(n_pcs)
    rho = jax.nn.softplus(rho_raw) * jnp.ones(n_pcs)
    E = jax.nn.softplus(E_raw) * jnp.ones(n_pcs)
    G = jax.nn.softplus(G_raw) * jnp.ones(n_pcs)

    # update robot, map and controller
    robot_updated = robot.update_params({"L": L, "D": D, "r": r, "rho": rho, "E": E, "G": G})
    controller_updated = controller.update_params(p_controller)
    encoder_updated = encoder.update_params(p_encoder)
    decoder_updated = decoder.update_params(p_decoder)

    # compute q and qd: q = phi(y) and qd = J_phi(y)*yd
    q_batch, qd_batch = encoder_updated.forward_xd_batch(y_batch, yd_batch) # shape (batch_size, 3*n_pcs)

    # predictions
    z_batch = jnp.concatenate([q_batch, qd_batch], axis=1) # state z=[q^T, qd^T]. Shape (batch_size, 2*3*n_pcs)
    if controller_to_train == 'tanh_simple':
        tau_batch = controller_updated.forward_batch(q_batch)
    elif controller_to_train == 'none':
        tau_batch = jnp.zeros_like(q_batch)
    else:
        tau_batch = controller_updated.forward_batch(z_batch)
    actuation_arg = (tau_batch,)

    forward_dynamics_vmap = jax.vmap(robot_updated.forward_dynamics, in_axes=(None,0,0))
    zd_batch = forward_dynamics_vmap(0, z_batch, actuation_arg) # state derivative zd=[qd^T, qdd^T]. Shape (batch_size, 2*3*n_pcs)
    _, qdd_batch = jnp.split(zd_batch, 2, axis=1) 

    # comptue ydd_hat: ydd_hat = J_psi(q)*qdd + H_psi(q)*(qd qd^T)
    ydd_hat_batch = decoder_updated.forward_xdd_batch(q_batch, qd_batch, qdd_batch) # shape (batch_size, 3*n_pcs)

    # compute mse loss (compare predictions ydd_hat with labels ydd)
    MSE = jnp.mean(jnp.sum((ydd_hat_batch - ydd_batch)**2, axis=1))
    
    # compute reconstruction loss
    match reconstruction:
        case 'y':
            y_hat_batch_rec = decoder_updated.forward_batch(q_batch)
            reconstruction_loss = 1e2 * jnp.mean(jnp.sum((y_batch - y_hat_batch_rec)**2, axis=1))
        case 'yd':
            y_hat_batch_rec, yd_hat_batch_rec = decoder_updated.forward_xd_batch(q_batch, qd_batch)
            reconstruction_loss = 1e1 * jnp.mean(jnp.sum((y_batch - y_hat_batch_rec)**2, axis=1)) + 1e1 * jnp.mean(jnp.sum((yd_batch - yd_hat_batch_rec)**2, axis=1))
        case 'ydd':
            y_hat_batch_rec, yd_hat_batch_rec = decoder_updated.forward_xd_batch(q_batch, qd_batch)
            ydd_hat_batch_rec = decoder_updated.forward_xdd_batch(q_batch, qd_batch, encoder_updated.forward_xdd_batch(y_batch, yd_batch, ydd_batch))
            reconstruction_loss = 1e1 * jnp.mean(jnp.sum((y_batch - y_hat_batch_rec)**2, axis=1)) + 1e1 * jnp.mean(jnp.sum((yd_batch - yd_hat_batch_rec)**2, axis=1)) + 1e1 * jnp.mean(jnp.sum((ydd_batch - ydd_hat_batch_rec)**2, axis=1))
    
    # complete loss
    loss = MSE + reconstruction_loss

    # store metrics
    metrics = {
        "MSE": MSE,
        "predictions": ydd_hat_batch,
        "labels": ydd_batch,
        "reconstructionMSE": reconstruction_loss
    }

    return loss, metrics


# =====================================================
# Prepare datasets
# =====================================================

# Load dataset: m data from a RON with n_ron oscillators
if not coupled_case:
    dataset = onp.load(dataset_folder/'soft robot optimization/N6_simplified/dataset_m1e5_N6_simplified.npz')
else:
    dataset = onp.load(dataset_folder/'soft robot optimization/N6_noInput/dataset_m1e5_N6_noInput.npz')
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
key, key_encoder, key_decoder, key_controller = jax.random.split(key, 4)

# ...mapping
mlp_sizes_enc = [n_ron, 64, 64, 3*n_pcs]
mlp_sizes_dec = [3*n_pcs, 64, 64, n_ron]
encoder = MLP(key=key_encoder, layer_sizes=mlp_sizes_enc, scale_init=0.01)
decoder = MLP(key=key_decoder, layer_sizes=mlp_sizes_dec, scale_init=0.01)

p_encoder = tuple(encoder.params) # tuple of tuples with layers: ((W1, b1), (W2, b2), ...)
p_decoder = tuple(decoder.params) # tuple of tuples with layers: ((W1, b1), (W2, b2), ...)
p_map = (p_encoder, p_decoder)

# ...robot
L0 = 1e-1 * jnp.ones(n_pcs)
D0 = jnp.diag(jnp.tile(jnp.array([5e-6, 5e-3, 5e-3]), n_pcs))
r0 = 2e-2 * jnp.ones(n_pcs)
rho0 = 1070 * jnp.ones(n_pcs)
E0 = 2e3 * jnp.ones(n_pcs)
G0 = 1e3 * jnp.ones(n_pcs)

L_raw = InverseSoftplus(L0)
D_raw = InverseSoftplus(jnp.diag(D0))
r_raw = InverseSoftplus(r0[0])
rho_raw = InverseSoftplus(rho0[0])
E_raw = InverseSoftplus(E0[0])
G_raw = InverseSoftplus(G0[0])

p_robot = (L_raw, D_raw, r_raw, rho_raw, E_raw, G_raw)

# ...controller
match controller_to_train:
    case 'tanh_simple':
        mlp_sizes = [3*n_pcs, 3*n_pcs]
        scale_init = 0.00001
    case 'tanh_complete':
        mlp_sizes = [2*3*n_pcs, 3*n_pcs]
        scale_init = 0.00001
    case 'mlp':
        mlp_sizes = [2*3*n_pcs, 64, 64, 3*n_pcs]
        scale_init = 0.001
    case 'none':
        mlp_sizes = [2*3*n_pcs, 3*n_pcs]
        scale_init = 0.0
    case _:
        raise ValueError('Unknown controller')
mlp_controller = MLP(key=subkey, layer_sizes=mlp_sizes, scale_init=scale_init) # initialize MLP feedback control law

p_controller = tuple(mlp_controller.params) # tuple of tuples with layers: ((W1, b1), (W2, b2), ...)

# Collect parameters
params_optimiz = (p_robot, p_map, p_controller)

# Initialize robot
parameters = {
    "th0": jnp.array(jnp.pi/2),
    "L": L0,
    "r": r0,
    "rho": rho0,
    "g": jnp.array([0.0, 9.81]), # !! gravity UP !!
    "E": E0,
    "G": G0,
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
    if not coupled_case:
        RON_evolution_data = onp.load(dataset_folder/'soft robot optimization/N6_simplified/RON_evolution_N6_simplified_a.npz')
    else:
        RON_evolution_data = onp.load(dataset_folder/'soft robot optimization/N6_noInput/RON_evolution_N6_noInput.npz')
    time_RONsaved = jnp.array(RON_evolution_data['time'])
    y_RONsaved = jnp.array(RON_evolution_data['y'])
    yd_RONsaved = jnp.array(RON_evolution_data['yd'])

    # Define controller
    def tau_law(system_state: SystemState, controller: MLP):
        """Implements user-defined feedback control tau(t) = MLP(q(t),qd(t))."""
        if controller_to_train == 'tanh_simple':
            q, _ = jnp.split(system_state.y, 2)
            tau = controller(q)
        else:
            tau = controller(system_state.y)
        return tau, None
    tau_fb = jax.jit(partial(tau_law, controller=mlp_controller)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox

    # Simulation parameters
    t0 = time_RONsaved[0]
    t1 = time_RONsaved[-1]
    dt = 1e-4
    saveat = np.arange(t0, t1, 1e-2)
    solver = Tsit5()
    step_size = ConstantStepSize()
    max_steps = int(1e6)

    # Simulate robot
    q0 = encoder(y_RONsaved[0])
    qd0 = encoder.compute_jacobian(q0) @ yd_RONsaved[0]
    initial_state_pcs = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))

    print('Simulating robot...')
    start = time.perf_counter()
    sim_out_pcs = robot.rollout_closed_loop_to(
        initial_state = initial_state_pcs,
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
    timePCS = sim_out_pcs.t
    q_PCS, qd_PCS = jnp.split(sim_out_pcs.y, 2, axis=1)
    u_pcs = sim_out_pcs.u

    y_hat_pcs = decoder.forward_batch(q_PCS) # y_hat(t) = psi(q(t)). Shape (n_steps, n_ron)
    yd_hat_pcs = jnp.einsum("bij,bj->bi", jax.vmap(decoder.compute_jacobian)(q_PCS), qd_PCS) # yd_hat(t) = J_psi(q(t))*qd(t)

    # Plot PCS strains
    fig, axs = plt.subplots(3,1, figsize=(12,9))
    for i in range(n_pcs):
        axs[0].plot(timePCS, q_PCS[:,i], label=f'segment {i+1}')
        axs[0].grid(True)
        axs[0].set_xlabel('t [s]')
        axs[0].set_ylabel(r"$\kappa_\mathrm{be}$ [rad/m]")
        axs[0].set_title('Bending strain')
        axs[0].legend()
        axs[1].plot(timePCS, q_PCS[:,i+1], label=f'segment {i+1}')
        axs[1].grid(True)
        axs[1].set_xlabel('t [s]')
        axs[1].set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
        axs[1].set_title('Axial strain')
        axs[1].legend()
        axs[2].plot(timePCS, q_PCS[:,i+2], label=f'segment {i+1}')
        axs[2].grid(True)
        axs[2].set_xlabel('t [s]')
        axs[2].set_ylabel(r"$\sigma_\mathrm{sh}$ [-]")
        axs[2].set_title('Shear strain')
        axs[2].legend()
    plt.tight_layout()
    plt.savefig(plots_folder/'Strains_before', bbox_inches='tight')
    #plt.show()

    # Plot actuation power before training
    fig, axs = plt.subplots(3,1, figsize=(10,6))
    for i in range(n_pcs):
        axs[0].plot(timePCS, qd_PCS[:,i] * u_pcs[:,i], label=f'segment {i+1}')
        axs[0].grid(True)
        axs[0].set_xlabel('t [s]')
        axs[0].set_ylabel(r"$P_\mathrm{be}$ [W]")
        axs[0].set_title('Bending actuation power')
        axs[0].legend()
        axs[1].plot(timePCS, qd_PCS[:,i+1] * u_pcs[:,i+1], label=f'segment {i+1}')
        axs[1].grid(True)
        axs[1].set_xlabel('t [s]')
        axs[1].set_ylabel(r"$P_\mathrm{ax}$ [W]")
        axs[1].set_title('Axial actuation power')
        axs[1].legend()
        axs[2].plot(timePCS, qd_PCS[:,i+2] * u_pcs[:,i+2], label=f'segment {i+1}')
        axs[2].grid(True)
        axs[2].set_xlabel('t [s]')
        axs[2].set_ylabel(r"$P_\mathrm{sh}$ [W]")
        axs[2].set_title('Shear actuation power')
        axs[2].legend()
    plt.tight_layout()
    plt.savefig(plots_folder/'Power_before', bbox_inches='tight')
    #plt.show()

    # Plot y(t) and y_hat(t)
    fig, axs = plt.subplots(3,2, figsize=(12,9))
    for i, ax in enumerate(axs.flatten()):
        ax.plot(time_RONsaved, y_RONsaved[:,i], 'b--', label=r'$y_{RON}(t)$')
        ax.plot(timePCS, y_hat_pcs[:,i], 'b', label=r'$\hat{y}_{PCS}(t)$')
        ax.grid(True)
        ax.set_xlabel('t [s]')
        ax.set_ylabel('y, q')
        ax.set_title(f'Component {i+1}')
        ax.set_ylim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
        ax.legend()
    plt.tight_layout()
    plt.savefig(plots_folder/'RONvsPCS_time_before', bbox_inches='tight')
    #plt.show()

    # Plot phase planes
    fig, axs = plt.subplots(3,2, figsize=(12,9))
    for i, ax in enumerate(axs.flatten()):
        ax.plot(y_RONsaved[:, i], yd_RONsaved[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
        ax.plot(y_hat_pcs[:, i], yd_hat_pcs[:, i], 'b', label=r'$(\hat{y}_{PCS}, \, \hat{\dot{y}}_{PCS})$')
        ax.grid(True)
        ax.set_xlabel(r'$y$')
        ax.set_ylabel(r'$\dot{y}$')
        ax.set_title(f'Component {i+1}')
        ax.set_xlim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
        ax.set_ylim([onp.min(yd_RONsaved[:,i])-1, onp.max(yd_RONsaved[:,i])+1])
        ax.legend()
    plt.tight_layout()
    plt.savefig(plots_folder/'RONvsPCS_phaseplane_before', bbox_inches='tight')
    #plt.show()

    # Animate robot
    animate_robot_matplotlib(
        robot = robot,
        t_list = saveat,
        q_list = q_PCS,
        interval = 1e-3, 
        slider = False,
        animation = True,
        show = True,
        duration = 10,
        fps = 30,
        save_path = plots_folder/'animation_before.gif',
    )
else:
    print('[simulation skipped]')

# Correct signature for loss function
Loss = partial(Loss, robot=robot, controller=mlp_controller, encoder=encoder, decoder=decoder)

# Test RMSE on the test set before optimization
_, metrics = Loss(
    params_optimiz=params_optimiz, 
    data_batch=test_set,
)
RMSE = onp.sqrt(metrics["MSE"])
pred = onp.array(metrics["predictions"])
labels = onp.array(metrics["labels"])
reconstructionRMSE = onp.sqrt(metrics["reconstructionMSE"])

print(f'Test accuracy: RMSE = {RMSE:.6e}, reconstruction RMSE = {reconstructionRMSE:.6e}')
print(f'Example:\n'
      f'    (y, yd) = ({onp.array(test_set["y"][69])}, {onp.array(test_set["yd"][69])})\n'
      f'    prediction: ydd_hat = {pred[69]}\n'
      f'    label: ydd = {labels[69]}\n'
      f'    error: |e| = {onp.abs( labels[69] - pred[69] )}\n\n'
      f'    encoding: (q, qd) = ({onp.array(encoder(test_set["y"][69]))}, {onp.array(encoder.compute_jacobian(test_set["y"][69])@test_set["yd"][69])})\n'
      f'    reconstruction: (y_hat, yd_hat) = ({onp.array(decoder(encoder(test_set["y"][69])))}, {onp.array(decoder.compute_jacobian(encoder(test_set["y"][69])) @ (encoder.compute_jacobian(test_set["y"][69])@test_set["yd"][69]))})\n'
)


# =====================================================
# Optimization
# =====================================================

if True:
    print(F'\n--- OPTIMIZATION ---')

    # Optimization parameters
    n_iter = 9000 # number of epochs
    batch_size = 2**6

    key, subkey = jax.random.split(key)
    batches_per_epoch = batch_indx_generator(subkey, train_size, batch_size).shape[0]

    # Setup optimizer
    lr = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=1e-3,
        warmup_steps=100*batches_per_epoch,
        decay_steps=n_iter*batches_per_epoch,
    )       
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr),
    )
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
    elatime_optimiz = end - start
    print(f'Elapsed time (optimization): {elatime_optimiz} s')

    # Print optimal parameters
    params_optimiz_opt = params_optimiz
    p_robot_opt, p_map_opt, p_controller_opt = params_optimiz_opt

    L_raw_opt, D_raw_opt, r_raw_opt, rho_raw_opt, E_raw_opt, G_raw_opt = p_robot_opt
    p_encoder_opt, p_decoder_opt = p_map_opt

    L_opt = jax.nn.softplus(L_raw_opt)
    D_opt = jnp.diag(jax.nn.softplus(D_raw_opt))
    r_opt = jax.nn.softplus(r_raw_opt) * jnp.ones(n_pcs)
    rho_opt = jax.nn.softplus(rho_raw_opt) * jnp.ones(n_pcs)
    E_opt = jax.nn.softplus(E_raw_opt) * jnp.ones(n_pcs)
    G_opt = jax.nn.softplus(G_raw_opt) * jnp.ones(n_pcs)

    print(
        f'L_opt: \n{L_opt}\n'
        f'D_opt: \n{D_opt}\n'
        f'r_opt: \n{r_opt}\n'
        f'rho_opt: \n{rho_opt}\n'
        f'E_opt: \n{E_opt}\n'
        f'G_opt: \n{G_opt}'
    )

    # Update optimal controller, robot and map
    robot_opt = robot.update_params({"L": L_opt, "D": D_opt, "r": r_opt, "rho": rho_opt, "E": E_opt, "G": G_opt})
    mlp_controller_opt = mlp_controller.update_params(p_controller_opt)
    encoder_opt = encoder.update_params(p_encoder_opt)
    decoder_opt = decoder.update_params(p_decoder_opt)

    # Save optimal parameters
    onp.savez(
        data_folder/'optimal_data_robot.npz', 
        L=onp.array(L_opt), 
        D=onp.array(D_opt),
        r=onp.array(r_opt),
        rho=onp.array(rho_opt),
        E=onp.array(E_opt),
        G=onp.array(G_opt),
    )
    mlp_controller_opt.save_params(data_folder/'optimal_data_controller.npz')
    encoder_opt.save_params(data_folder/'optimal_data_encoder.npz')
    decoder_opt.save_params(data_folder/'optimal_data_decoder.npz')

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
    plt.title(f'Loss curve (elapsed time: {elatime_optimiz:.2f} s)')
    plt.tight_layout()
    plt.savefig(plots_folder/'Loss', bbox_inches='tight')
    plt.show()


# =====================================================
# Approximator after optimization
# =====================================================
print('\n--- AFTER OPTIMIZATION ---')

# Load optimal parameters
if False:
    # Choose prefix for data
    prefix_load = ''

    data_robot_opt = onp.load(data_folder/f'{prefix_load}optimal_data_robot.npz')
    p_encoder_opt = encoder.load_params(data_folder/f'{prefix_load}optimal_data_encoder.npz')
    p_decoder_opt = decoder.load_params(data_folder/f'{prefix_load}optimal_data_decoder.npz')
    p_controller_opt = mlp_controller.load_params(data_folder/f'{prefix_load}optimal_data_controller.npz')

    L_opt = jnp.array(data_robot_opt['L'])
    D_opt = jnp.array(data_robot_opt['D'])
    r_opt = jnp.array(data_robot_opt['r'])
    rho_opt = jnp.array(data_robot_opt['rho'])
    E_opt = jnp.array(data_robot_opt['E'])
    G_opt = jnp.array(data_robot_opt['G'])

    L_raw_opt = InverseSoftplus(L_opt)
    D_raw_opt = InverseSoftplus(jnp.diag(D_opt))
    r_raw_opt = InverseSoftplus(r_opt[0])
    rho_raw_opt = InverseSoftplus(rho_opt[0])
    E_raw_opt = InverseSoftplus(E_opt[0])
    G_raw_opt = InverseSoftplus(G_opt[0])

    p_robot_opt = (L_raw_opt, D_raw_opt, r_raw_opt, rho_raw_opt, E_raw_opt, G_raw_opt)
    p_map_opt = (p_encoder_opt, p_decoder_opt)

    robot_opt = robot.update_params({"L": L_opt, "D": D_opt, "r": r_opt, "rho": rho_opt, "E": E_opt, "G": G_opt})
    encoder_opt = encoder.update_params(p_encoder_opt)
    decoder_opt = decoder.update_params(p_decoder_opt)
    mlp_controller_opt = mlp_controller.update_params(p_controller_opt)

    params_optimiz_opt = (p_robot_opt, p_map_opt, p_controller_opt)

# If required, simulate robot and compare its behaviour in time with the RON's one
if show_simulations:
    # Update control law
    tau_fb_opt = jax.jit(partial(tau_law, controller=mlp_controller_opt)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox
    
    # Simulate robot
    q0 = encoder_opt(y_RONsaved[0])
    qd0 = encoder_opt.compute_jacobian(q0) @ yd_RONsaved[0]
    initial_state_pcs = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))

    print('Simulating robot...')
    start = time.perf_counter()
    sim_out_pcs = robot_opt.rollout_closed_loop_to(
        initial_state = initial_state_pcs,
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
    timePCS = sim_out_pcs.t
    q_PCS, qd_PCS = jnp.split(sim_out_pcs.y, 2, axis=1)
    u_pcs = sim_out_pcs.u

    y_hat_pcs = decoder_opt.forward_batch(q_PCS) # y_hat(t) = psi(q(t)). Shape (n_steps, n_ron)
    yd_hat_pcs = jnp.einsum("bij,bj->bi", jax.vmap(decoder_opt.compute_jacobian)(q_PCS), qd_PCS) # yd_hat(t) = J_psi(q(t))*qd(t)

    # Plot PCS strains
    fig, axs = plt.subplots(3,1, figsize=(12,9))
    for i in range(n_pcs):
        axs[0].plot(timePCS, q_PCS[:,i], label=f'segment {i+1}')
        axs[0].grid(True)
        axs[0].set_xlabel('t [s]')
        axs[0].set_ylabel(r"$\kappa_\mathrm{be}$ [rad/m]")
        axs[0].set_title('Bending strain')
        axs[0].legend()
        axs[1].plot(timePCS, q_PCS[:,i+1], label=f'segment {i+1}')
        axs[1].grid(True)
        axs[1].set_xlabel('t [s]')
        axs[1].set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
        axs[1].set_title('Axial strain')
        axs[1].legend()
        axs[2].plot(timePCS, q_PCS[:,i+2], label=f'segment {i+1}')
        axs[2].grid(True)
        axs[2].set_xlabel('t [s]')
        axs[2].set_ylabel(r"$\sigma_\mathrm{sh}$ [-]")
        axs[2].set_title('Shear strain')
        axs[2].legend()
    plt.tight_layout()
    plt.savefig(plots_folder/'Strains_after', bbox_inches='tight')
    #plt.show()

    # Plot actuation power after training
    fig, axs = plt.subplots(3,1, figsize=(10,6))
    for i in range(n_pcs):
        axs[0].plot(timePCS, qd_PCS[:,i] * u_pcs[:,i], label=f'segment {i+1}')
        axs[0].grid(True)
        axs[0].set_xlabel('t [s]')
        axs[0].set_ylabel(r"$P_\mathrm{be}$ [W]")
        axs[0].set_title('Bending actuation power')
        axs[0].legend()
        axs[1].plot(timePCS, qd_PCS[:,i+1] * u_pcs[:,i+1], label=f'segment {i+1}')
        axs[1].grid(True)
        axs[1].set_xlabel('t [s]')
        axs[1].set_ylabel(r"$P_\mathrm{ax}$ [W]")
        axs[1].set_title('Axial actuation power')
        axs[1].legend()
        axs[2].plot(timePCS, qd_PCS[:,i+2] * u_pcs[:,i+2], label=f'segment {i+1}')
        axs[2].grid(True)
        axs[2].set_xlabel('t [s]')
        axs[2].set_ylabel(r"$P_\mathrm{sh}$ [W]")
        axs[2].set_title('Shear actuation power')
        axs[2].legend()
    plt.tight_layout()
    plt.savefig(plots_folder/'Power_after', bbox_inches='tight')
    #plt.show()
    
    # Plot y(t) and y_hat(t)
    fig, axs = plt.subplots(3,2, figsize=(12,9))
    for i, ax in enumerate(axs.flatten()):
        ax.plot(time_RONsaved, y_RONsaved[:,i], 'b--', label=r'$y_{RON}(t)$')
        ax.plot(timePCS, y_hat_pcs[:,i], 'b', label=r'$\hat{y}_{PCS}(t)$')
        ax.grid(True)
        ax.set_xlabel('t [s]')
        ax.set_ylabel('y, q')
        ax.set_title(f'Component {i+1}')
        ax.set_ylim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
        ax.legend()
    plt.tight_layout()
    plt.savefig(plots_folder/'RONvsPCS_time_after', bbox_inches='tight')
    #plt.show()

    # Plot phase planes
    fig, axs = plt.subplots(3,2, figsize=(12,9))
    for i, ax in enumerate(axs.flatten()):
        ax.plot(y_RONsaved[:, i], yd_RONsaved[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
        ax.plot(y_hat_pcs[:, i], yd_hat_pcs[:, i], 'b', label=r'$(\hat{y}_{PCS}, \, \hat{\dot{y}}_{PCS})$')
        ax.grid(True)
        ax.set_xlabel(r'$y$')
        ax.set_ylabel(r'$\dot{y}$')
        ax.set_title(f'Component {i+1}')
        ax.set_xlim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
        ax.set_ylim([onp.min(yd_RONsaved[:,i])-1, onp.max(yd_RONsaved[:,i])+1])
        ax.legend()
    plt.tight_layout()
    plt.savefig(plots_folder/'RONvsPCS_phaseplane_after', bbox_inches='tight')
    #plt.show()

    # Animate robot
    animate_robot_matplotlib(
        robot = robot_opt,
        t_list = saveat,
        q_list = q_PCS,
        interval = 1e-3, 
        slider = False,
        animation = True,
        show = True,
        duration = 10,
        fps = 30,
        save_path = plots_folder/'animation_after.gif',
    )
else:
    print('[simulation skipped]')

# Correct signature for loss function
Loss = partial(Loss, robot=robot_opt, controller=mlp_controller_opt, encoder=encoder_opt, decoder=decoder_opt)

# Test RMSE on the test set after optimization
_, metrics = Loss(
    params_optimiz=params_optimiz_opt, 
    data_batch=test_set,
)
RMSE = onp.sqrt(metrics["MSE"])
pred = onp.array(metrics["predictions"])
labels = onp.array(metrics["labels"])
reconstructionRMSE = onp.sqrt(metrics["reconstructionMSE"])

print(f'Test accuracy: RMSE = {RMSE:.6e}, reconstruction RMSE = {reconstructionRMSE:.6e}')
print(f'Example:\n'
      f'    (y, yd) = ({onp.array(test_set["y"][69])}, {onp.array(test_set["yd"][69])})\n'
      f'    prediction: ydd_hat = {pred[69]}\n'
      f'    label: ydd = {labels[69]}\n'
      f'    error: |e| = {onp.abs( labels[69] - pred[69] )}\n\n'
      f'    encoding: (q, qd) = ({onp.array(encoder_opt(test_set["y"][69]))}, {onp.array(encoder_opt.compute_jacobian(test_set["y"][69])@test_set["yd"][69])})\n'
      f'    reconstruction: (y_hat, yd_hat) = ({onp.array(decoder_opt(encoder_opt(test_set["y"][69])))}, {onp.array(decoder_opt.compute_jacobian(encoder_opt(test_set["y"][69])) @ (encoder_opt.compute_jacobian(test_set["y"][69])@test_set["yd"][69]))})\n'
)

# Compute actuation power mean squared value on the test set after optimization
q_test_power, qd_test_power = encoder_opt.forward_xd_batch(test_set["y"], test_set["yd"]) # shape (testset_size, 3*n_pcs)
z_test_power = jnp.concatenate([q_test_power, qd_test_power], axis=1) # shape (testset_size, 2*3*n_pcs)
tau_test_power = mlp_controller_opt.forward_batch(z_test_power) # shape (testset_size, 3*n_pcs)
power = jnp.sum(tau_test_power * qd_test_power, axis=1) # shape (testset_size,)
power_msv_after = jnp.mean(power**2) # scalar

# Save some metrics
with open(data_folder/'metrics.txt', 'w') as file:
    file.write(f"Final test RMS error:                {RMSE}\n")
    file.write(f"Final test RMS reconstruction error: {reconstructionRMSE}\n")
    file.write(f"Final test RMS power:                {onp.sqrt(power_msv_after)}\n")