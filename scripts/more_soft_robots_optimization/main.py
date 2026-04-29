# =====================================================
# Setup
# =====================================================

# Choose device (cpu or gpu)
import os
os.environ["JAX_PLATFORMS"] = "cpu"

# Imports
import numpy as onp
import jax
import jax.numpy as jnp
import optax
from diffrax import Tsit5, ConstantStepSize, LinearInterpolation, AbstractTerm

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
jax.config.update("jax_enable_x64", True) # double precision
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

# General
load_experiment = False # choose whether to load saved experiment or to perform training
experiment = 'TEST' # name of the experiment to perform/load
use_scan = False # choose whether to use normal for loop or lax.scan
show_simulations = True # choose whether to perform time simulations of the physical reservoir (and comparison with RON)
simulation_duration = jnp.inf # seconds of example simulation to perform. Choose simulation_duration=jnp.inf for the full simulation in ron_evolution_example

# Reference RON reservoir
ron_case = 'input' # 'simple' 'coupled' 'input'
ron_dataset = 'sMNIST_RON_N12_DT0.006_RHO0.99/dataset_m1e5_N12_DT0.006_RHO0.99' # name of the case to load from 'soft robot optimization' folder
ron_evolution_example = 'sMNIST_RON_N12_DT0.006_RHO0.99/RON_evolution_N12_DT0.006_RHO0.99_long' # name of the case to load from 'soft robot optimization' folder
#ron_dataset = 'MG_RON_N12_DT0.15/dataset_m1e5_N12_DT0.15' # name of the case to load from 'soft robot optimization' folder
#ron_evolution_example = 'MG_RON_N12_DT0.15/RON_evolution_N12_DT0.15' # name of the case to load from 'soft robot optimization' folder

# controller
train_unique_controller = False # if True, Tau = Tau_tot(Z, u), where Tau_tot is specified in fb_controller_to_train. 
                               # If False, Tau = Tau_fb(Z) + Tau_ff(u), where Tau_fb is specified in fb_controller_to_train and Tau_ff in ff_controller_to_train
fb_controller_to_train = 'tanh_complete' # 'linear_simple', 'linear_complete', 'tanh_simple', 'tanh_complete', 'mlp'
ff_controller_to_train = 'mlp' # (only applies to train_unique_controller = False). Choose 'linear', 'tanh', 'mlp'

# Mapping
map_to_train = 'norm_flow' # 'diag', 'svd', 'reconstruction', 'norm_flow'
reconstruction_type = 'ydd' # (only applies to 'reconstruction') reconstruction loss on y and optionally on yd and ydd. Choose 'y', 'yd', or 'ydd'

# Robots
n_robots = 2 # number of soft robots in the reservoir
n_pcs = 2 # number of segments for the single PCS
train_robots = True # if False, does not optimize the soft robot


# =====================================================
# Functions
# =====================================================

# Rename folders for plots/data
plots_folder = plots_folder/experiment
data_folder = data_folder/experiment
data_folder.mkdir(parents=True, exist_ok=True)
plots_folder.mkdir(parents=True, exist_ok=True)

# Convert map parameters if necessary
match map_to_train:
    case 'diag':
        # Converts A -> A_raw
        @partial(jax.jit, static_argnums=(1,))
        def A2Araw(A: Array, a_thresh: float=0.0) -> Tuple:
            A_flat = jnp.diag(A)
            A_raw = InverseSoftplus(A_flat - a_thresh) # convert singular values
            return A_raw

        # Converts A_raw -> A
        @partial(jax.jit, static_argnums=(1,))
        def Araw2A(A_raw: Tuple, a_thresh: float=0.0) -> Array:
            A_flat = jax.nn.softplus(A_raw) + a_thresh
            A = jnp.diag(A_flat)
            return A
        
    case 'svd':
        # Converts A -> A_raw
        @partial(jax.jit, static_argnums=(1,))
        def A2Araw(A: Array, s_thresh: float=0.0) -> Tuple:
            """A_raw is tuple (U,s,Vt) with SVD of A = U*diag(s)*Vt, where s vector is parametrized with softplus
            to ensure s_i > thresh >= 0 for all i."""
            U, s, Vt = jnp.linalg.svd(A)          # decompose A = U*S*V.T, with s=diag(S) and Vt=V^T
            s_raw = InverseSoftplus(s - s_thresh) # convert singular values
            A_raw = (U, s_raw, Vt)
            return A_raw

        # Converts A_raw -> A
        @partial(jax.jit, static_argnums=(1,))
        def Araw2A(A_raw: Tuple, s_thresh: float=0.0) -> Array:
            """A_raw is tuple (U,s,Vt) with SVD of A = U*diag(s)*V^T, where s vector is parametrized with softplus
            to ensure s_i > thresh >= 0 for all i."""
            U, s_raw, Vt = A_raw
            s = jax.nn.softplus(s_raw) + s_thresh
            A = U @ jnp.diag(s) @ Vt
            return A
    case _:
        pass

# Convert soft robots parameters
def convert2original(raw_params: ParamsRobots) -> ParamsRobots:
    """
    Transform raw parameters to physical parameters.

    Args
    ----
    raw_params : ParamsRobots
        Dictionary of raw parameters.

    Returns
    -------
    params : ParamsRobots
        Dictionary of physical parameters with same keys.
    """
    return {
        "L": jax.nn.softplus(raw_params["L"]),
        "r": jax.nn.softplus(raw_params["r"]),
        "rho": jax.nn.softplus(raw_params["rho"]),
        "E": jax.nn.softplus(raw_params["E"]),
        "G": jax.nn.softplus(raw_params["G"]),
        "D": jax.vmap(jnp.diag)(jax.nn.softplus(raw_params["D"])),
    }

def convert2raw(params: ParamsRobots) -> ParamsRobots:
    """
    Transform physical parameters to raw parameters.

    Args
    ----
    params : ParamsRobots
        Dictionary of physical parameters.

    Returns
    -------
    raw_params : ParamsRobots
        Dictionary of raw parameters.
    """
    return {
        "L": InverseSoftplus(params["L"]),
        "r": InverseSoftplus(params["r"]),
        "rho": InverseSoftplus(params["rho"]),
        "E": InverseSoftplus(params["E"]),
        "G": InverseSoftplus(params["G"]),
        "D": InverseSoftplus(jax.vmap(jnp.diag)(params["D"])),
    }

# Loss function
@jax.jit
def Loss(
        params_optimiz : Sequence, 
        data_batch : Dict, 
        robots : MultiPcsSystem,
        controller : MLP | Tuple[MLP, MLP],
        map: RealNVP | Tuple[MLP, MLP] = None,
) -> Tuple[float, Dict]:
    """
    Computes loss function over a batch of data for certain parameters. In this case:
    Takes a batch of datapoints (y, yd, u), computes the forward dynamics ydd_hat = f_approximator(y,yd,u)
    and computes the loss as the MSE in the batch between predictions ydd_hat and labels ydd.

    Args
    ----
    params_optimiz : Sequence
        (Raw) parameters for the opimization. In this case a tuple with:
        - **p_robots**: Dictionary with pcs params for all robots {"L":L_raw,"D":D_raw, ...} with L_raw of shape (n_robots,n_pcs), etc.
        - **p_map**: Trainable parameters of the mapping function.
        - **p_controller**: Trainable parameters of the controller.              
    data_batch : Dict
        Dictionary with datapoints and labels to compute the loss. In this case has keys:
        - **"y"**: Batch of datapoints y. Shape (batch_size, n_ron)
        - **"yd"**: Batch of datapoints yd. Shape (batch_size, n_ron)
        - **"u"**: Batch of datapoints u Shape (batch_size, n_input)
        - **"ydd"**: Batch of labels ydd. Shape (batch_size, n_ron)
    robots : MultiPcsSystem
        "Dummy" instance of the soft robots system class (actual parameters are in params_optimiz).
    controller : MLP | Tuple[MLP, MLP]
        "Dummy" instance of the controller (actual parameters are in params_optimiz). Can be either a unique controller
        Tau = MLP(Z, u) or a tuple with two controllers Tau_fb = MLP(Z) and Tau_ff = MLP(u) such that Tau = Tau_fb + Tau_ff.
    map : RealNVP | Tuple[MLP, MLP] | None
        "Dummy" instance of the map (actual parameters are in params_optimiz). Can be a RealNVP class, a tuple 
        containing encoder/decoder or None if linear mapping is used (default: None).

    Returns
    -------
    loss : float
        Scalar loss computed as MSE in the batch between predictions and labels.
    metrics : Dict[float, Dict]
        Dictionary of useful metrics.
    """
    # extract dataset
    y_batch = data_batch["y"]
    yd_batch = data_batch["yd"]
    ydd_batch = data_batch["ydd"]
    u_batch = data_batch["u"]

    # extract parameters
    p_robots_raw, p_map, p_controller = params_optimiz

    # convert parameters and update instances
    # ...robot
    p_robots = convert2original(p_robots_raw)
    if train_robots:
        robots_updated = robots.update_params(p_robots)
    else:
        robots_updated = robots

    # ...map
    match map_to_train:
        case 'diag' | 'svd':
            A_raw, c = p_map
            A = Araw2A(A_raw, A_thresh)
        case 'reconstruction':
            p_encoder, p_decoder = p_map
            encoder, decoder = map
            encoder_updated = encoder.update_params(p_encoder)
            decoder_updated = decoder.update_params(p_decoder)
        case 'norm_flow':
            map_updated = map.update_params(p_map)

    # ...controller
    if train_unique_controller:
        controller_updated = controller.update_params(p_controller)
    else:
        fb_controller, ff_controller = controller
        p_fb_controller, p_ff_controller = p_controller
        fb_controller_updated = fb_controller.update_params(p_fb_controller)
        ff_controller_updated = ff_controller.update_params(p_ff_controller)

    # compute Q and Qd -> direct map
    match map_to_train:
        case 'diag' | 'svd':
            Q_batch = y_batch @ jnp.transpose(A) + c # shape (batch_size, 3*n_pcs*n_robots)
            Qd_batch = yd_batch @ jnp.transpose(A) # shape (batch_size, 3*n_pcs*n_robots)
        case 'reconstruction':
            Q_batch, Qd_batch = encoder_updated.forward_xd_batch(y_batch, yd_batch) # shape (batch_size, 3*n_pcs*n_robots)
        case 'norm_flow':
            Q_batch, Qd_batch = map_updated.forward_with_derivatives_batch(y_batch, yd_batch) # shape (batch_size, 3*n_pcs*n_robots)

    # actuation
    Z_batch = jnp.concatenate([Q_batch, Qd_batch], axis=1) # state Z=[Q^T, Qd^T]^T. Shape (batch_size, 2*3*n_pcs*n_robots)

    if fb_controller_to_train == 'tanh_simple' or fb_controller_to_train == 'linear_simple':
        fb_contr_inp = Q_batch # shape (batch_size, 3*n_pcs*n_robots)
    else:
        fb_contr_inp = Z_batch # shape (batch_size, 2*3*n_pcs*n_robots)

    if ron_case == 'input':
        contr_inp = jnp.concatenate([fb_contr_inp, u_batch], axis=1) # shape (batch_size, 3*n_pcs*n_robots+1) or (batch_size, 2*3*n_pcs*n_robots+1)
    else:
        contr_inp = fb_contr_inp # shape (batch_size, 3*n_pcs*n_robots) or (batch_size, 2*3*n_pcs*n_robots)

    if train_unique_controller:
        Tau_batch = controller_updated.forward_batch(contr_inp) # shape (batch_size, 3*n_pcs*n_robots)
    else:
        Tau_batch = fb_controller_updated.forward_batch(fb_contr_inp) + ff_controller_updated.forward_batch(u_batch) # shape (batch_size, 3*n_pcs*n_robots)

    # predictions
    forward_dynamics_vmap = jax.vmap(robots_updated.forward_dynamics, in_axes=(None,0,0))
    Zd_batch = forward_dynamics_vmap(0, Z_batch, (Tau_batch,)) # state derivative Zd=[qd^T, qdd^T]. Shape (batch_size, 2*3*n_pcs*n_robots)
    _, Qdd_batch = jnp.split(Zd_batch, 2, axis=1) # shape (batch_size, 3*n_pcs*n_robots)

    # comptue ydd_hat -> inverse map
    match map_to_train:
        case 'diag' | 'svd':
            ydd_hat_batch = jnp.linalg.solve(A, Qdd_batch.T).T # shape (batch_size, n_ron)
        case 'reconstruction':
            ydd_hat_batch = decoder_updated.forward_xdd_batch(Q_batch, Qd_batch, Qdd_batch) # shape (batch_size, n_ron)
        case 'norm_flow':
            _, _, ydd_hat_batch = map_updated.inverse_with_derivatives_batch(Q_batch, Qd_batch, Qdd_batch) # shape (batch_size, n_ron)

    # compute MSE loss (compare predictions ydd_hat with labels ydd)
    MSE = jnp.mean(jnp.sum((ydd_hat_batch - ydd_batch)**2, axis=1))

    # add reconstruction loss if required
    if map_to_train == 'reconstruction':
        match reconstruction_type:
            case 'y':
                y_hat_batch_rec = decoder_updated.forward_batch(Q_batch)
                reconstruction_loss = 1e2 * jnp.mean(jnp.sum((y_batch - y_hat_batch_rec)**2, axis=1))
            case 'yd':
                y_hat_batch_rec, yd_hat_batch_rec = decoder_updated.forward_xd_batch(Q_batch, Qd_batch)
                reconstruction_loss = 1e1 * jnp.mean(jnp.sum((y_batch - y_hat_batch_rec)**2, axis=1)) + 1e1 * jnp.mean(jnp.sum((yd_batch - yd_hat_batch_rec)**2, axis=1))
            case 'ydd':
                y_hat_batch_rec, yd_hat_batch_rec = decoder_updated.forward_xd_batch(Q_batch, Qd_batch)
                ydd_hat_batch_rec = decoder_updated.forward_xdd_batch(Q_batch, Qd_batch, encoder_updated.forward_xdd_batch(y_batch, yd_batch, ydd_batch))
                reconstruction_loss = 1e1 * jnp.mean(jnp.sum((y_batch - y_hat_batch_rec)**2, axis=1)) + 1e1 * jnp.mean(jnp.sum((yd_batch - yd_hat_batch_rec)**2, axis=1)) + 1e1 * jnp.mean(jnp.sum((ydd_batch - ydd_hat_batch_rec)**2, axis=1))
        loss = MSE + reconstruction_loss
    else:
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
dataset = onp.load(dataset_folder/'soft robot optimization'/f'{ron_dataset}.npz')
y = dataset["y"] # position samples of the RON oscillators. Shape (m, n_ron)
yd = dataset["yd"] # velocity samples of the RON oscillators. Shape (m, n_ron)
ydd = dataset["ydd"] # accelerations of the RON oscillators. Shape (m, n_ron)
u = dataset["u"] # input. Shape (m, 1)

# Convert into jax
y_dataset = jnp.array(y, dtype=jnp.float64)
yd_dataset = jnp.array(yd, dtype=jnp.float64)
ydd_dataset = jnp.array(ydd, dtype=jnp.float64)
u_dataset = jnp.array(u, dtype=jnp.float64)

dataset = {
    "y": y_dataset,
    "yd": yd_dataset,
    "u": u_dataset,
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
# Physical reservoir before optimization
# =====================================================
print('--- BEFORE OPTIMIZATION ---')

# Initialize parameters and instances
key, key_map, key_controller = jax.random.split(key, 3)

# ...mapping
match map_to_train:
    case 'diag':
        A_thresh = 1e-4 # threshold on the singular values
        A0 = jnp.diag(jnp.tile(jnp.array([1e0, 1e-1, 1e-1]), n_robots*n_pcs)) # shape (3*n_pcs*n_robots, n_ron)
        c0 = jnp.zeros(3*n_pcs*n_robots)

        map = None
        A_raw = A2Araw(A0, A_thresh)
        p_map = (A_raw, c0)

    case 'svd':
        A_thresh = 1e-4 # threshold on the singular values
        s_init = 3e-3
        A0 = init_A_svd(key_map, n_ron, s_init, s_init + 1e-6)
        c0 = jnp.zeros(3*n_pcs*n_robots)

        map = None
        A_raw = A2Araw(A0, A_thresh)
        p_map = (A_raw, c0)

    case 'reconstruction':
        key, key_encoder, key_decoder = jax.random.split(key, 3)
        mlp_sizes_enc = [n_ron, 32, 32, 3*n_pcs*n_robots]
        mlp_sizes_dec = [3*n_pcs*n_robots, 32, 32, n_ron]
        encoder = MLP(key=key_encoder, layer_sizes=mlp_sizes_enc, scale_init=0.01)
        decoder = MLP(key=key_decoder, layer_sizes=mlp_sizes_dec, scale_init=0.01)

        map = (encoder, decoder)
        p_encoder = tuple(encoder.params) # tuple of tuples with layers: ((W1, b1), (W2, b2), ...)
        p_decoder = tuple(decoder.params) # tuple of tuples with layers: ((W1, b1), (W2, b2), ...)
        p_map = (p_encoder, p_decoder)

    case 'norm_flow':
        n_coupling_layers = 4 # number of coupling layers
        nets_hidden_dim = 32 # dimension of the MLPs (all of them have 2 hidden layers)
        activation_fn_map = 'tanh' # activation function for the MLPs ('tanh' or 'relu')

        masks = create_alternating_masks(input_dim=n_ron, num_layers=n_coupling_layers) # list of length num_layers. Each element is a (input_dim,) binary array 
        map = RealNVP(
            key_map,
            masks=masks,
            hidden_dim=nets_hidden_dim,
            activation_fn=activation_fn_map,
            scale_init_t_net=0.01,
            scale_init_scale_factor=0.01
        )
        p_map = tuple(map.params)

# ...robots
L0 = jnp.tile(1e-1 * jnp.ones(n_pcs), (n_robots,1))
D0 = jnp.tile(jnp.diag(jnp.tile(jnp.array([5e-6, 5e-3, 5e-3]), n_pcs)), (n_robots,1,1))
r0 = jnp.tile(2e-2 * jnp.ones(n_pcs),(n_robots,1))
rho0 = jnp.tile(1070 * jnp.ones(n_pcs),(n_robots,1))
E0 = jnp.tile(2e3 * jnp.ones(n_pcs),(n_robots,1))
G0 = jnp.tile(1e3 * jnp.ones(n_pcs),(n_robots,1))

p_robots_original = {
    "th0": jnp.tile(jnp.array(jnp.pi/2), n_robots),
    "L": L0,
    "r": r0,
    "rho": rho0,
    "g": jnp.tile(jnp.array([0.0, 9.81]), (n_robots,1)), # !! gravity UP !!
    "E": E0,
    "G": G0,
    "D": D0,
}
robots_system = MultiPcsSystem(
    n_robots = n_robots,
    n_pcs = n_pcs,
    params_robots = p_robots_original
)

p_robots = convert2raw(p_robots_original)

# ...controller
if train_unique_controller:
    match fb_controller_to_train:
        case 'tanh_simple':
            scale_init = 0.00001
            last_layer_activation = 'tanh'
            if ron_case == 'input':
                mlp_sizes = [3*n_pcs*n_robots + 1, 3*n_pcs*n_robots]
            else:
                mlp_sizes = [3*n_pcs*n_robots, 3*n_pcs*n_robots]

        case 'linear_simple':
            scale_init = 0.00001
            last_layer_activation = 'linear'
            if ron_case == 'input':
                mlp_sizes = [3*n_pcs*n_robots + 1, 3*n_pcs*n_robots]
            else:
                mlp_sizes = [3*n_pcs*n_robots, 3*n_pcs*n_robots]

        case 'tanh_complete':
            scale_init = 0.00001
            last_layer_activation = 'tanh'
            if ron_case == 'input':
                mlp_sizes = [2*3*n_pcs*n_robots + 1, 3*n_pcs*n_robots]
            else:
                mlp_sizes = [2*3*n_pcs*n_robots, 3*n_pcs*n_robots]

        case 'linear_complete':
            scale_init = 0.00001
            last_layer_activation = 'linear'
            if ron_case == 'input':
                mlp_sizes = [2*3*n_pcs*n_robots + 1, 3*n_pcs*n_robots]
            else:
                mlp_sizes = [2*3*n_pcs*n_robots, 3*n_pcs*n_robots]
            
        case 'mlp':
            scale_init = 0.001
            last_layer_activation = 'linear'
            if ron_case == 'input':
                mlp_sizes = [2*3*n_pcs*n_robots + 1, 64, 64, 3*n_pcs*n_robots]
            else:
                mlp_sizes = [2*3*n_pcs*n_robots, 64, 64, 3*n_pcs*n_robots]

        case _:
            raise ValueError('Unknown controller')
        
    mlp_controller = MLP(key=subkey, layer_sizes=mlp_sizes, scale_init=scale_init, last_layer=last_layer_activation) # initialize MLP feedback control law
    p_controller = tuple(mlp_controller.params) # tuple of tuples with layers: ((W1, b1), (W2, b2), ...)

else:
    match fb_controller_to_train:
        case 'tanh_simple':
            fb_scale_init = 0.00001
            fb_last_layer_activation = 'tanh'
            fb_mlp_sizes = [3*n_pcs*n_robots, 3*n_pcs*n_robots]

        case 'linear_simple':
            fb_last_layer_activation = 'linear'
            fb_scale_init = 0.00001
            fb_mlp_sizes = [3*n_pcs*n_robots, 3*n_pcs*n_robots]

        case 'tanh_complete':
            fb_scale_init = 0.00001
            fb_last_layer_activation = 'tanh'
            fb_mlp_sizes = [2*3*n_pcs*n_robots, 3*n_pcs*n_robots]

        case 'linear_complete':
            fb_scale_init = 0.00001
            fb_last_layer_activation = 'linear'
            fb_mlp_sizes = [2*3*n_pcs*n_robots, 3*n_pcs*n_robots]
            
        case 'mlp':
            fb_scale_init = 0.001
            fb_last_layer_activation = 'linear'
            fb_mlp_sizes = [2*3*n_pcs*n_robots, 64, 64, 3*n_pcs*n_robots]

        case _:
            raise ValueError('Unknown fb controller')
    
    match ff_controller_to_train:
        case 'tanh':
            ff_scale_init = 0.00001
            ff_last_layer_activation = 'tanh'
            ff_mlp_sizes = [1, 3*n_pcs*n_robots]

        case 'mlp':
            ff_scale_init = 0.001
            ff_last_layer_activation = 'linear'
            ff_mlp_sizes = [1, 64, 64, 3*n_pcs*n_robots]

        case 'linear':
            ff_scale_init = 0.00001
            ff_last_layer_activation = 'linear'
            ff_mlp_sizes = [1, 3*n_pcs*n_robots]

        case _:
            raise ValueError('Unknown fb controller')
        
    key, key_fb, key_ff = jax.random.split(key, 3)
    fb_mlp_controller = MLP(key=key_fb, layer_sizes=fb_mlp_sizes, scale_init=fb_scale_init, last_layer=fb_last_layer_activation) # initialize MLP feedback control law
    ff_mlp_controller = MLP(key=key_ff, layer_sizes=ff_mlp_sizes, scale_init=ff_scale_init, last_layer=ff_last_layer_activation) # initialize MLP feedforward control law
    mlp_controller = (fb_mlp_controller, ff_mlp_controller)

    p_fb_controller = tuple(fb_mlp_controller.params) # tuple of tuples with layers: ((W1, b1), (W2, b2), ...)
    p_ff_controller = tuple(ff_mlp_controller.params) # tuple of tuples with layers: ((W1, b1), (W2, b2), ...)
    p_controller = (p_fb_controller, p_ff_controller)

# Collect parameters
params_optimiz = (p_robots, p_map, p_controller)

# If required, simulate robot and compare its behaviour in time with the RON's one
if show_simulations:
    # Load simulation results from RON
    RON_evolution_data = onp.load(dataset_folder/'soft robot optimization'/f'{ron_evolution_example}.npz')
    time_RONsaved = jnp.array(RON_evolution_data['time'], dtype=jnp.float64)
    idx_max_time = jnp.searchsorted(time_RONsaved, simulation_duration) - 1 # get index of element < 'simulation_duration' s
    time_RONsaved = time_RONsaved[:idx_max_time] # only first ~'simulation_duration' s
    y_RONsaved = jnp.array(RON_evolution_data['y'], dtype=jnp.float64)[:idx_max_time] # only first ~'simulation_duration' s
    yd_RONsaved = jnp.array(RON_evolution_data['yd'], dtype=jnp.float64)[:idx_max_time] # only first ~'simulation_duration' s
    u_RONsaved = jnp.array(RON_evolution_data['u'], dtype=jnp.float64)[:idx_max_time] # only first ~'simulation_duration' s
    if len(u_RONsaved.shape) == 1: # !!! THIS IS HERE BECAUSE OF HOW DATA WERE SAVED: sMINST HAS (n_steps, 1) WHILE M-G HAS (n_steps,) !!!
        u_RONsaved = u_RONsaved[:, None] # ...and we want to work with (n_steps, n_u)

    # Define controller
    min_len = jnp.min(jnp.array([len(time_RONsaved), len(u_RONsaved)]))
    u_interpolator = LinearInterpolation(
        ts=time_RONsaved[:min_len],
        ys=u_RONsaved[:min_len]
    )

    def tau_law(system_state: SystemState, controller: MLP | Tuple[MLP, MLP], u_interp_fn: AbstractTerm):
        """Implements user-defined control Tau(t) = Tau_fn(Q(t),Qd(t),u(t))."""
        u = u_interp_fn.evaluate(system_state.t)
        Q, Qd = jnp.split(system_state.y, 2)
        Z = system_state.y

        if not train_unique_controller:
            fb_controller, ff_controller = controller

        if fb_controller_to_train == 'tanh_simple' or fb_controller_to_train == 'linear_simple':
            fb_contr_inp = Q
        else:
            fb_contr_inp = Z

        if ron_case == 'input':
            contr_inp = jnp.concatenate([fb_contr_inp, u])
        else:
            contr_inp = fb_contr_inp

        if train_unique_controller:
            Tau = controller(contr_inp)
        else:
            Tau = fb_controller(fb_contr_inp) + ff_controller(u)

        return Tau, None
    
    tau_fb = jax.jit(partial(tau_law, controller=mlp_controller, u_interp_fn=u_interpolator)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox

    # Simulation parameters
    t0 = time_RONsaved[0]
    t1 = time_RONsaved[-1]
    dt = 1e-4
    saveat = np.arange(t0, t1, 1e-2)
    solver = Tsit5()
    step_size = ConstantStepSize()
    max_steps = int(1e6)

    # Convert initial condition RON -> latent
    match map_to_train:
        case 'diag' | 'svd':
            Q0 = A0 @ y_RONsaved[0] + c0
            Qd0 = A0 @ yd_RONsaved[0]
        case 'reconstruction':
            Q0 = encoder(y_RONsaved[0])
            Qd0 = encoder.compute_jacobian(Q0) @ yd_RONsaved[0]
        case 'norm_flow':
            Q0, Qd0 = map.forward_with_derivatives(y_RONsaved[0], yd_RONsaved[0])
    initial_state_pcs = SystemState(t=t0, y=jnp.concatenate([Q0, Qd0]))

    # Show max 15 DOFs in the plots
    if 3*n_pcs*n_robots > 15:
        n_show = 15
    else:
        n_show = 3*n_pcs*n_robots

    n_cols = min(3, n_show)
    n_rows = int(np.ceil(n_show / n_cols))

    if True:
        # Simulate robot
        print('Simulating robot...')
        start = time.perf_counter()
        sim_out_pcs = robots_system.rollout_closed_loop_to(
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
        Z_ts = sim_out_pcs.y # shape (n_steps, 3*2*n_pcs*n_robots)
        Q_ts, Qd_ts = jnp.split(Z_ts, 2, axis=1) # shape (n_steps, 3*n_pcs*n_robots) each
        _, q_ts, qd_ts = jax.vmap(robots_system.transform_Z)(Z_ts) # shape (n_steps, n_robots, 3*n_pcs) each
        Tau_ts = sim_out_pcs.u # shape (n_steps, 3*n_pcs*n_robots)
        tau_ts = jax.vmap(robots_system.transform_Tau)(Tau_ts) # shape (n_steps, n_robots, 3*n_pcs)

        # Convert output latent -> RON
        match map_to_train:
            case 'diag' | 'svd':
                y_hat_pcs = jnp.linalg.solve(A0, (Q_ts - c0).T).T # y_hat(t) = inv(A) * ( Q(t) - c )
                yd_hat_pcs = jnp.linalg.solve(A0, Qd_ts.T).T # yd_hat(t) = inv(A) * Qd(t)
            case 'reconstruction':
                y_hat_pcs = decoder.forward_batch(Q_ts) # y_hat(t) = psi(Q(t)). Shape (n_steps, n_ron)
                yd_hat_pcs = jnp.einsum("bij,bj->bi", jax.vmap(decoder.compute_jacobian)(Q_ts), Qd_ts) # yd_hat(t) = J_psi(Q(t))*Qd(t)
            case 'norm_flow':
                y_hat_pcs, yd_hat_pcs = map.inverse_with_derivatives_batch(Q_ts, Qd_ts) # shape (n_steps, n_ron)

        # Plot PCS strains
        for n in range(n_robots):
            fig, axs = plt.subplots(3,1, figsize=(12,9))
            for i in range(int(n_show/3/n_robots)):
                axs[0].plot(timePCS, q_ts[:,n,3*i], label=f'segment {i+1}')
                axs[0].grid(True)
                axs[0].set_xlabel('t [s]')
                axs[0].set_ylabel(r"$\kappa_\mathrm{be}$ [rad/m]")
                axs[0].set_title('Bending strain')
                axs[0].legend()
                axs[1].plot(timePCS, q_ts[:,n,3*i+1], label=f'segment {i+1}')
                axs[1].grid(True)
                axs[1].set_xlabel('t [s]')
                axs[1].set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
                axs[1].set_title('Axial strain')
                axs[1].legend()
                axs[2].plot(timePCS, q_ts[:,n,3*i+2], label=f'segment {i+1}')
                axs[2].grid(True)
                axs[2].set_xlabel('t [s]')
                axs[2].set_ylabel(r"$\sigma_\mathrm{sh}$ [-]")
                axs[2].set_title('Shear strain')
                axs[2].legend()
            plt.tight_layout()
            plt.savefig(plots_folder/f'Strains_before_robot_{n+1}', bbox_inches='tight')
            #plt.show()
            plt.close()

            # Plot actuation power
            fig, axs = plt.subplots(3,1, figsize=(12,9))
            for i in range(int(n_show/3/n_robots)):
                axs[0].plot(timePCS, qd_ts[:,n,3*i] * tau_ts[:,n,3*i], label=f'segment {i+1}')
                axs[0].grid(True)
                axs[0].set_xlabel('t [s]')
                axs[0].set_ylabel(r"$P_\mathrm{be}$ [W]")
                axs[0].set_title('Bending actuation power')
                axs[0].legend()
                axs[1].plot(timePCS, qd_ts[:,n,3*i+1] * tau_ts[:,n,3*i+1], label=f'segment {i+1}')
                axs[1].grid(True)
                axs[1].set_xlabel('t [s]')
                axs[1].set_ylabel(r"$P_\mathrm{ax}$ [W]")
                axs[1].set_title('Axial actuation power')
                axs[1].legend()
                axs[2].plot(timePCS, qd_ts[:,n,3*i+2] * tau_ts[:,n,3*i+2], label=f'segment {i+1}')
                axs[2].grid(True)
                axs[2].set_xlabel('t [s]')
                axs[2].set_ylabel(r"$P_\mathrm{sh}$ [W]")
                axs[2].set_title('Shear actuation power')
                axs[2].legend()
            plt.tight_layout()
            plt.savefig(plots_folder/f'Power_before_robot_{n+1}', bbox_inches='tight')
            #plt.show()
            plt.close()

        # Plot y(t) and y_hat(t)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 9))
        if n_show == 1:
            axs = np.array([axs])
        else:
            axs = axs.flatten()

        for i in range(n_show):
            axs[i].plot(time_RONsaved, y_RONsaved[:,i], 'b--', label=r'$y(t)$')
            axs[i].plot(timePCS, y_hat_pcs[:,i], 'b', label=r'$\hat{y}(t)$')
            axs[i].grid(True)
            axs[i].set_xlabel('t [s]')
            axs[i].set_ylabel('y')
            axs[i].set_title(f'Component {i+1}')
            axs[i].set_ylim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            axs[i].legend()

        for i in range(3*n_pcs*n_robots, len(axs)):
            axs[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(plots_folder/'RONvsPCS_time_before', bbox_inches='tight')
        #plt.show()
        plt.close()

        # Plot phase planes
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 9))
        if n_show == 1:
            axs = np.array([axs])
        else:
            axs = axs.flatten()

        for i in range(n_show):
            axs[i].plot(y_RONsaved[:, i], yd_RONsaved[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
            axs[i].plot(y_hat_pcs[:, i], yd_hat_pcs[:, i], 'b', label=r'$(\hat{y}, \, \hat{\dot{y}})$')
            axs[i].grid(True)
            axs[i].set_xlabel(r'$y$')
            axs[i].set_ylabel(r'$\dot{y}$')
            axs[i].set_title(f'Component {i+1}')
            axs[i].set_xlim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
            axs[i].set_ylim([onp.min(yd_RONsaved[:,i])-1, onp.max(yd_RONsaved[:,i])+1])
            axs[i].legend()

        for i in range(3*n_pcs*n_robots, len(axs)):
            axs[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(plots_folder/'RONvsPCS_phaseplane_before', bbox_inches='tight')
        #plt.show()
        plt.close()

        # Plot total actuation
        for n in range(n_robots):
            fig, axs = plt.subplots(3,1, figsize=(12,9))
            for i in range(int(n_show/3/n_robots)):
                axs[0].plot(timePCS, tau_ts[:,n,3*i], label=f'segment {i+1}')
                axs[0].grid(True)
                axs[0].set_xlabel('t [s]')
                axs[0].set_ylabel(r"$\tau_{be}$ [$N\cdot m^2$]")
                axs[0].set_title('Bending actuation')
                axs[0].legend()
                axs[1].plot(timePCS, tau_ts[:,n,3*i+1], label=f'segment {i+1}')
                axs[1].grid(True)
                axs[1].set_xlabel('t [s]')
                axs[1].set_ylabel(r"$\tau_{ax}$ [$N\cdot m$]")
                axs[1].set_title('Axial actuation')
                axs[1].legend()
                axs[2].plot(timePCS, tau_ts[:,n,3*i+2], label=f'segment {i+1}')
                axs[2].grid(True)
                axs[2].set_xlabel('t [s]')
                axs[2].set_ylabel(r"$\tau_{sh}$ [$N\cdot m$]")
                axs[2].set_title('Shear actuation')
                axs[2].legend()
            plt.tight_layout()
            plt.savefig(plots_folder/f'Actuation_before_robot_{n+1}', bbox_inches='tight')
            #plt.show()
            plt.close()

        # Plot feedforward and feedback terms (if not unique controller)
        if not train_unique_controller:
            Tau_ff_component_ts = ff_mlp_controller.forward_batch(u_RONsaved[:min_len]) # shape (n_steps, 3*n_pcs*n_robots)
            tau_ff_component_ts = jax.vmap(robots_system.transform_Tau)(Tau_ff_component_ts) # shape (n_steps, n_robots, 3*n_pcs)
            if fb_controller_to_train == 'linear_simple' or fb_controller_to_train == 'tanh_simple':
                Tau_fb_component_ts = fb_mlp_controller.forward_batch(Q_ts) # shape (n_steps, 3*n_pcs*n_robots)
            else:
                Tau_fb_component_ts = fb_mlp_controller.forward_batch(Z_ts) # shape (n_steps, 3*n_pcs*n_robots)
            tau_fb_component_ts = jax.vmap(robots_system.transform_Tau)(Tau_fb_component_ts) # shape (n_steps, n_robots, 3*n_pcs)

            for n in range(n_robots):
                # Plot feedforward actuation
                fig, axs = plt.subplots(3,1, figsize=(12,9))
                for i in range(int(n_show/3/n_robots)):
                    axs[0].plot(time_RONsaved[:min_len], tau_ff_component_ts[:,n,3*i], label=f'segment {i+1}')
                    axs[0].grid(True)
                    axs[0].set_xlabel('t [s]')
                    axs[0].set_ylabel(r"$\tau_{be}$ [$N\cdot m^2$]")
                    axs[0].set_title('Bending ff actuation')
                    axs[0].legend()
                    axs[1].plot(time_RONsaved[:min_len], tau_ff_component_ts[:,n,3*i+1], label=f'segment {i+1}')
                    axs[1].grid(True)
                    axs[1].set_xlabel('t [s]')
                    axs[1].set_ylabel(r"$\tau_{ax}$ [$N\cdot m$]")
                    axs[1].set_title('Axial ff actuation')
                    axs[1].legend()
                    axs[2].plot(time_RONsaved[:min_len], tau_ff_component_ts[:,n,3*i+2], label=f'segment {i+1}')
                    axs[2].grid(True)
                    axs[2].set_xlabel('t [s]')
                    axs[2].set_ylabel(r"$\tau_{sh}$ [$N\cdot m$]")
                    axs[2].set_title('Shear ff actuation')
                    axs[2].legend()
                plt.tight_layout()
                plt.savefig(plots_folder/f'Actuation_ff_before_robot_{n+1}', bbox_inches='tight')
                #plt.show()
                plt.close()

                # Plot feedback actuation
                fig, axs = plt.subplots(3,1, figsize=(12,9))
                for i in range(int(n_show/3/n_robots)):
                    axs[0].plot(timePCS, tau_fb_component_ts[:,n,3*i], label=f'segment {i+1}')
                    axs[0].grid(True)
                    axs[0].set_xlabel('t [s]')
                    axs[0].set_ylabel(r"$\tau_{be}$ [$N\cdot m^2$]")
                    axs[0].set_title('Bending fb actuation')
                    axs[0].legend()
                    axs[1].plot(timePCS, tau_fb_component_ts[:,n,3*i+1], label=f'segment {i+1}')
                    axs[1].grid(True)
                    axs[1].set_xlabel('t [s]')
                    axs[1].set_ylabel(r"$\tau_{ax}$ [$N\cdot m$]")
                    axs[1].set_title('Axial fb actuation')
                    axs[1].legend()
                    axs[2].plot(timePCS, tau_fb_component_ts[:,n,3*i+2], label=f'segment {i+1}')
                    axs[2].grid(True)
                    axs[2].set_xlabel('t [s]')
                    axs[2].set_ylabel(r"$\tau_{sh}$ [$N\cdot m$]")
                    axs[2].set_title('Shear fb actuation')
                    axs[2].legend()
                plt.tight_layout()
                plt.savefig(plots_folder/f'Actuation_fb_before_robot_{n+1}', bbox_inches='tight')
                #plt.show()
                plt.close()

        # Animate robots
        for n in range(n_robots):
            flag = True if n == n_robots - 1 else False
            animate_robot_matplotlib(
                robot = robots_system.get_robot(n),
                t_list = saveat,
                q_list = q_ts[:,n],
                interval = 1e-3, 
                slider = False,
                animation = True,
                #show = flag,
                show = False,
                duration = 10,
                fps = 30,
                save_path = plots_folder/f'animation_before_robot_{n+1}.gif',
            )
    else:
        print('[simulation skipped]')
else:
    print('[simulation skipped]')

# Correct signature for loss function
Loss = partial(Loss, robots=robots_system, controller=mlp_controller, map=map)

# Test RMSE on the test set before optimization
_, metrics = Loss(
    params_optimiz=params_optimiz, 
    data_batch=test_set,
)
RMSE = onp.sqrt(metrics["MSE"])
pred = onp.array(metrics["predictions"])
labels = onp.array(metrics["labels"])
match map_to_train:
    case 'reconstruction':
        Q_encoded = encoder(test_set["y"][69])
        Qd_encoded = encoder.compute_jacobian(test_set["y"][69]) @ test_set["yd"][69]
        Qdd_encoded = encoder.forward_xdd(test_set["y"][69], test_set["yd"][69], test_set["ydd"][69])
        y_decoded = decoder(Q_encoded)
        yd_decoded = decoder.compute_jacobian(Q_encoded) @ Qd_encoded
        ydd_decoded = decoder.forward_xdd(Q_encoded, Qd_encoded, Qdd_encoded)
    case 'norm_flow':
        Q_encoded, Qd_encoded, Qdd_encoded = map.forward_with_derivatives(test_set["y"][69], test_set["yd"][69], test_set["ydd"][69])
        y_decoded, yd_decoded, ydd_decoded = map.inverse_with_derivatives(Q_encoded, Qd_encoded, Qdd_encoded)
    case _:
        Q_encoded = A0 @ test_set["y"][69] + c0
        Qd_encoded = A0 @ test_set["yd"][69]
        Qdd_encoded = A0 @ test_set["ydd"][69]
        y_decoded = jnp.linalg.solve(A0, (Q_encoded - c0).T).T
        yd_decoded = jnp.linalg.solve(A0, Qd_encoded.T).T
        ydd_decoded = jnp.linalg.solve(A0, Qdd_encoded.T).T

print(f'Test accuracy: RMSE = {RMSE:.6e}')
print(f'Example:\n'
      f'    datapoint (y, yd, ydd):   ({onp.array(test_set["y"][69])}, {onp.array(test_set["yd"][69])}, {onp.array(test_set["ydd"][69])})\n'
      f'    encoding (Q, Qd, Qdd):    ({onp.array(Q_encoded)}, {onp.array(Qd_encoded)}, {onp.array(Qdd_encoded)})\n'
      f'    decoding (y_, yd_, ydd_): ({onp.array(y_decoded)}, {onp.array(yd_decoded)}, {onp.array(ydd_decoded)})\n\n'
      f'    prediction ydd_hat: {pred[69]}\n'
      f'    label ydd:          {labels[69]}\n'
      f'    error |e|:          {onp.abs( labels[69] - pred[69] )}\n'
)


# =====================================================
# Optimization
# =====================================================

if not load_experiment:
    print(F'\n--- OPTIMIZATION ---')

    # Optimization parameters
    n_iter = 1500 # number of epochs
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

    # Optimal parameters
    params_optimiz_opt = params_optimiz
    p_robots_opt, p_map_opt, p_controller_opt = params_optimiz_opt

    # Print optimal robot parameters
    p_robots_opt_original = convert2original(p_robots_opt)
    L_opt = p_robots_opt_original["L"]
    D_opt = p_robots_opt_original["D"]
    r_opt = p_robots_opt_original["r"]
    rho_opt = p_robots_opt_original["rho"]
    E_opt = p_robots_opt_original["E"]
    G_opt = p_robots_opt_original["G"]

    print(
        f'L_opt: \n{L_opt}\n'
        f'D_opt: \n{D_opt}\n'
        f'r_opt: \n{r_opt}\n'
        f'rho_opt: \n{rho_opt}\n'
        f'E_opt: \n{E_opt}\n'
        f'G_opt: \n{G_opt}'
    )

    # Update optimal controller, robot and map
    robots_system_opt = robots_system.update_params(p_robots_opt_original)

    if train_unique_controller:
        mlp_controller_opt = mlp_controller.update_params(p_controller_opt)
    else:
        p_fb_controller_opt, p_ff_controller_opt = p_controller_opt
        fb_mlp_controller_opt = fb_mlp_controller.update_params(p_fb_controller_opt)
        ff_mlp_controller_opt = ff_mlp_controller.update_params(p_ff_controller_opt)
        mlp_controller_opt = (fb_mlp_controller_opt, ff_mlp_controller_opt)

    match map_to_train:
        case 'reconstruction':
            p_encoder_opt, p_decoder_opt = p_map_opt
            encoder_opt = encoder.update_params(p_encoder_opt)
            decoder_opt = decoder.update_params(p_decoder_opt)
            map_opt = (encoder_opt, decoder_opt)
        case 'norm_flow':
            map_opt = map.update_params(p_map_opt)
        case _:
            A_raw_opt, c_opt = p_map_opt
            A_opt = Araw2A(A_raw_opt, A_thresh)  
            map_opt = None          

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

    if train_unique_controller:
        mlp_controller_opt.save_params(data_folder/'optimal_data_controller.npz')
    else:
        fb_mlp_controller_opt.save_params(data_folder/'optimal_data_fb_controller.npz')
        ff_mlp_controller_opt.save_params(data_folder/'optimal_data_ff_controller.npz')

    match map_to_train:
        case 'reconstruction':
            encoder_opt.save_params(data_folder/'optimal_data_encoder.npz')
            decoder_opt.save_params(data_folder/'optimal_data_decoder.npz')
        case 'norm_flow':
            map_opt.save_params(data_folder/'optimal_data_map.npz')
        case _:
            onp.savez(
                data_folder/'optimal_data_map.npz', 
                A=onp.array(A_opt), 
                c=onp.array(c_opt)
            )

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
    #plt.show()
    plt.close()


# =====================================================
# Physical reservoir after optimization
# =====================================================
print('\n--- AFTER OPTIMIZATION ---')

# Load optimal parameters
if load_experiment:
    # Choose prefix for data
    prefix_load = ''

    # Load parameters robot
    data_robot_opt = onp.load(data_folder/f'{prefix_load}optimal_data_robot.npz')
    
    L_opt = jnp.array(data_robot_opt['L'], dtype=jnp.float64)
    D_opt = jnp.array(data_robot_opt['D'], dtype=jnp.float64)
    r_opt = jnp.array(data_robot_opt['r'], dtype=jnp.float64)
    rho_opt = jnp.array(data_robot_opt['rho'], dtype=jnp.float64)
    E_opt = jnp.array(data_robot_opt['E'], dtype=jnp.float64)
    G_opt = jnp.array(data_robot_opt['G'], dtype=jnp.float64)

    p_robots_opt_original = {
        "L": L_opt,
        "D": D_opt,
        "r": r_opt,
        "rho": rho_opt,
        "E": E_opt,
        "G": G_opt,
    }
    p_robots_opt = convert2raw(p_robots_opt_original)
    robots_system_opt = robots_system.update_params(p_robots_opt_original)
    
    # Load parameters controller
    if train_unique_controller:
        p_controller_opt = mlp_controller.load_params(data_folder/f'{prefix_load}optimal_data_controller.npz')
        mlp_controller_opt = mlp_controller.update_params(p_controller_opt)
    else:
        p_fb_controller_opt = fb_mlp_controller.load_params(data_folder/f'{prefix_load}optimal_data_fb_controller.npz')
        p_ff_controller_opt = ff_mlp_controller.load_params(data_folder/f'{prefix_load}optimal_data_ff_controller.npz')
        fb_mlp_controller_opt = fb_mlp_controller.update_params(p_fb_controller_opt)
        ff_mlp_controller_opt = ff_mlp_controller.update_params(p_ff_controller_opt)
        p_controller_opt = (p_fb_controller_opt, p_ff_controller_opt)
        mlp_controller_opt = (fb_mlp_controller_opt, ff_mlp_controller_opt)
    
    # Load parameters map
    match map_to_train:
        case 'reconstruction':
            p_encoder_opt = encoder.load_params(data_folder/f'{prefix_load}optimal_data_encoder.npz')
            p_decoder_opt = decoder.load_params(data_folder/f'{prefix_load}optimal_data_decoder.npz')
            p_map_opt = (p_encoder_opt, p_decoder_opt)
            encoder_opt = encoder.update_params(p_encoder_opt)
            decoder_opt = decoder.update_params(p_decoder_opt)
            map_opt = (encoder_opt, decoder_opt)
        case 'norm_flow':
            p_map_opt = map.load_params(data_folder/f'{prefix_load}optimal_data_map.npz')
            map_opt = map.update_params(p_map_opt)
        case _:
            data_map_opt = onp.load(data_folder/f'{prefix_load}optimal_data_map.npz')
            A_opt = jnp.array(data_map_opt['A'], dtype=jnp.float64)
            c_opt = jnp.array(data_map_opt['c'], dtype=jnp.float64)
            A_raw_opt = A2Araw(A_opt, A_thresh)
            p_map_opt = (A_raw_opt, c_opt)
            map_opt = None

    # Collect optimal parameters
    params_optimiz_opt = (p_robots_opt, p_map_opt, p_controller_opt)

# If required, simulate robot and compare its behaviour in time with the RON's one
if show_simulations:
    # Update control law
    tau_fb_opt = jax.jit(partial(tau_law, controller=mlp_controller_opt, u_interp_fn=u_interpolator)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox
    
    # Convert initial condition RON -> latent
    match map_to_train:
        case 'diag' | 'svd':
            Q0 = A_opt @ y_RONsaved[0] + c_opt
            Qd0 = A_opt @ yd_RONsaved[0]
        case 'reconstruction':
            Q0 = encoder_opt(y_RONsaved[0])
            Qd0 = encoder_opt.compute_jacobian(Q0) @ yd_RONsaved[0]
        case 'norm_flow':
            Q0, Qd0 = map_opt.forward_with_derivatives(y_RONsaved[0], yd_RONsaved[0])
    initial_state_pcs = SystemState(t=t0, y=jnp.concatenate([Q0, Qd0]))

    # Simulate robot
    print('Simulating robot...')
    start = time.perf_counter()
    sim_out_pcs = robots_system_opt.rollout_closed_loop_to(
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
    Z_ts = sim_out_pcs.y # shape (n_steps, 3*2*n_pcs*n_robots)
    Q_ts, Qd_ts = jnp.split(Z_ts, 2, axis=1) # shape (n_steps, 3*n_pcs*n_robots) each
    _, q_ts, qd_ts = jax.vmap(robots_system.transform_Z)(Z_ts) # shape (n_steps, n_robots, 3*n_pcs) each
    Tau_ts = sim_out_pcs.u # shape (n_steps, 3*n_pcs*n_robots)
    tau_ts = jax.vmap(robots_system.transform_Tau)(Tau_ts) # shape (n_steps, n_robots, 3*n_pcs) each

    # Convert output latent -> RON
    match map_to_train:
        case 'diag' | 'svd':
            y_hat_pcs = jnp.linalg.solve(A_opt, (Q_ts - c_opt).T).T # y_hat(t) = inv(A) * ( Q(t) - c )
            yd_hat_pcs = jnp.linalg.solve(A_opt, Qd_ts.T).T # yd_hat(t) = inv(A) * Qd(t)
        case 'reconstruction':
            y_hat_pcs = decoder_opt.forward_batch(Q_ts) # y_hat(t) = psi(Q(t)). Shape (n_steps, n_ron)
            yd_hat_pcs = jnp.einsum("bij,bj->bi", jax.vmap(decoder_opt.compute_jacobian)(Q_ts), Qd_ts) # yd_hat(t) = J_psi(Q(t))*Qd(t)
        case 'norm_flow':
            y_hat_pcs, yd_hat_pcs = map_opt.inverse_with_derivatives_batch(Q_ts, Qd_ts) # shape (n_steps, n_ron)

    # Plot PCS strains
    for n in range(n_robots):
        fig, axs = plt.subplots(3,1, figsize=(12,9))
        for i in range(int(n_show/3/n_robots)):
            axs[0].plot(timePCS, q_ts[:,n,3*i], label=f'segment {i+1}')
            axs[0].grid(True)
            axs[0].set_xlabel('t [s]')
            axs[0].set_ylabel(r"$\kappa_\mathrm{be}$ [rad/m]")
            axs[0].set_title('Bending strain')
            axs[0].legend()
            axs[1].plot(timePCS, q_ts[:,n,3*i+1], label=f'segment {i+1}')
            axs[1].grid(True)
            axs[1].set_xlabel('t [s]')
            axs[1].set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
            axs[1].set_title('Axial strain')
            axs[1].legend()
            axs[2].plot(timePCS, q_ts[:,n,3*i+2], label=f'segment {i+1}')
            axs[2].grid(True)
            axs[2].set_xlabel('t [s]')
            axs[2].set_ylabel(r"$\sigma_\mathrm{sh}$ [-]")
            axs[2].set_title('Shear strain')
            axs[2].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/f'Strains_after_robot_{n+1}', bbox_inches='tight')
        #plt.show()
        plt.close()

        # Plot actuation power
        fig, axs = plt.subplots(3,1, figsize=(12,9))
        for i in range(int(n_show/3/n_robots)):
            axs[0].plot(timePCS, qd_ts[:,n,3*i] * tau_ts[:,n,3*i], label=f'segment {i+1}')
            axs[0].grid(True)
            axs[0].set_xlabel('t [s]')
            axs[0].set_ylabel(r"$P_\mathrm{be}$ [W]")
            axs[0].set_title('Bending actuation power')
            axs[0].legend()
            axs[1].plot(timePCS, qd_ts[:,n,3*i+1] * tau_ts[:,n,3*i+1], label=f'segment {i+1}')
            axs[1].grid(True)
            axs[1].set_xlabel('t [s]')
            axs[1].set_ylabel(r"$P_\mathrm{ax}$ [W]")
            axs[1].set_title('Axial actuation power')
            axs[1].legend()
            axs[2].plot(timePCS, qd_ts[:,n,3*i+2] * tau_ts[:,n,3*i+2], label=f'segment {i+1}')
            axs[2].grid(True)
            axs[2].set_xlabel('t [s]')
            axs[2].set_ylabel(r"$P_\mathrm{sh}$ [W]")
            axs[2].set_title('Shear actuation power')
            axs[2].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/f'Power_after_robot_{n+1}', bbox_inches='tight')
        #plt.show()
        plt.close()
    
    # Plot y(t) and y_hat(t)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 9))
    if n_show == 1:
        axs = np.array([axs])
    else:
        axs = axs.flatten()

    for i in range(n_show):
        axs[i].plot(time_RONsaved, y_RONsaved[:,i], 'b--', label=r'$y_{RON}(t)$')
        axs[i].plot(timePCS, y_hat_pcs[:,i], 'b', label=r'$\hat{y}_{PCS}(t)$')
        axs[i].grid(True)
        axs[i].set_xlabel('t [s]')
        axs[i].set_ylabel('y')
        axs[i].set_title(f'Component {i+1}')
        #axs[i].set_ylim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
        axs[i].legend()

    for i in range(3*n_pcs*n_robots, len(axs)):
        axs[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(plots_folder/'RONvsPCS_time_after', bbox_inches='tight')
    #plt.show()
    plt.close()

    # Plot phase planes
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 9))
    if n_show == 1:
        axs = np.array([axs])
    else:
        axs = axs.flatten()

    for i in range(n_show):
        axs[i].plot(y_RONsaved[:, i], yd_RONsaved[:, i], 'b--', label=r'RON $(y, \, \dot{y})$')
        axs[i].plot(y_hat_pcs[:, i], yd_hat_pcs[:, i], 'b', label=r'$(\hat{y}_{PCS}, \, \hat{\dot{y}}_{PCS})$')
        axs[i].grid(True)
        axs[i].set_xlabel(r'$y$')
        axs[i].set_ylabel(r'$\dot{y}$')
        axs[i].set_title(f'Component {i+1}')
        #axs[i].set_xlim([onp.min(y_RONsaved[:,i])-1, onp.max(y_RONsaved[:,i])+1])
        #axs[i].set_ylim([onp.min(yd_RONsaved[:,i])-1, onp.max(yd_RONsaved[:,i])+1])
        axs[i].legend()

    for i in range(3*n_pcs*n_robots, len(axs)):
        axs[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(plots_folder/'RONvsPCS_phaseplane_after', bbox_inches='tight')
    #plt.show()
    plt.close()

    # Plot total actuation
    for n in range(n_robots):
        fig, axs = plt.subplots(3,1, figsize=(12,9))
        for i in range(int(n_show/3/n_robots)):
            axs[0].plot(timePCS, tau_ts[:,n,3*i], label=f'segment {i+1}')
            axs[0].grid(True)
            axs[0].set_xlabel('t [s]')
            axs[0].set_ylabel(r"$\tau_{be}$ [$N\cdot m^2$]")
            axs[0].set_title('Bending actuation')
            axs[0].legend()
            axs[1].plot(timePCS, tau_ts[:,n,3*i+1], label=f'segment {i+1}')
            axs[1].grid(True)
            axs[1].set_xlabel('t [s]')
            axs[1].set_ylabel(r"$\tau_{ax}$ [$N\cdot m$]")
            axs[1].set_title('Axial actuation')
            axs[1].legend()
            axs[2].plot(timePCS, tau_ts[:,n,3*i+2], label=f'segment {i+1}')
            axs[2].grid(True)
            axs[2].set_xlabel('t [s]')
            axs[2].set_ylabel(r"$\tau_{sh}$ [$N\cdot m$]")
            axs[2].set_title('Shear actuation')
            axs[2].legend()
        plt.tight_layout()
        plt.savefig(plots_folder/f'Actuation_after_robot_{n+1}', bbox_inches='tight')
        #plt.show()
        plt.close()

    # Plot feedforward and feedback terms (if not unique controller)
    if not train_unique_controller:
        Tau_ff_component_ts = ff_mlp_controller_opt.forward_batch(u_RONsaved[:min_len]) # shape (n_steps, 3*n_pcs*n_robots)
        tau_ff_component_ts = jax.vmap(robots_system_opt.transform_Tau)(Tau_ff_component_ts) # shape (n_steps, n_robots, 3*n_pcs)
        if fb_controller_to_train == 'linear_simple' or fb_controller_to_train == 'tanh_simple':
            Tau_fb_component_ts = fb_mlp_controller_opt.forward_batch(Q_ts) # shape (n_steps, 3*n_pcs*n_robots)
        else:
            Tau_fb_component_ts = fb_mlp_controller_opt.forward_batch(Z_ts) # shape (n_steps, 3*n_pcs*n_robots)
        tau_fb_component_ts = jax.vmap(robots_system_opt.transform_Tau)(Tau_fb_component_ts) # shape (n_steps, n_robots, 3*n_pcs)

        for n in range(n_robots):
            # Plot feedforward actuation
            fig, axs = plt.subplots(3,1, figsize=(12,9))
            for i in range(int(n_show/3/n_robots)):
                axs[0].plot(time_RONsaved[:min_len], tau_ff_component_ts[:,n,3*i], label=f'segment {i+1}')
                axs[0].grid(True)
                axs[0].set_xlabel('t [s]')
                axs[0].set_ylabel(r"$\tau_{be}$ [$N\cdot m^2$]")
                axs[0].set_title('Bending ff actuation')
                axs[0].legend()
                axs[1].plot(time_RONsaved[:min_len], tau_ff_component_ts[:,n,3*i+1], label=f'segment {i+1}')
                axs[1].grid(True)
                axs[1].set_xlabel('t [s]')
                axs[1].set_ylabel(r"$\tau_{ax}$ [$N\cdot m$]")
                axs[1].set_title('Axial ff actuation')
                axs[1].legend()
                axs[2].plot(time_RONsaved[:min_len], tau_ff_component_ts[:,n,3*i+2], label=f'segment {i+1}')
                axs[2].grid(True)
                axs[2].set_xlabel('t [s]')
                axs[2].set_ylabel(r"$\tau_{sh}$ [$N\cdot m$]")
                axs[2].set_title('Shear ff actuation')
                axs[2].legend()
            plt.tight_layout()
            plt.savefig(plots_folder/f'Actuation_ff_after_robot_{n+1}', bbox_inches='tight')
            #plt.show()
            plt.close()

            # Plot feedback actuation
            fig, axs = plt.subplots(3,1, figsize=(12,9))
            for i in range(int(n_show/3/n_robots)):
                axs[0].plot(timePCS, tau_fb_component_ts[:,n,3*i], label=f'segment {i+1}')
                axs[0].grid(True)
                axs[0].set_xlabel('t [s]')
                axs[0].set_ylabel(r"$\tau_{be}$ [$N\cdot m^2$]")
                axs[0].set_title('Bending fb actuation')
                axs[0].legend()
                axs[1].plot(timePCS, tau_fb_component_ts[:,n,3*i+1], label=f'segment {i+1}')
                axs[1].grid(True)
                axs[1].set_xlabel('t [s]')
                axs[1].set_ylabel(r"$\tau_{ax}$ [$N\cdot m$]")
                axs[1].set_title('Axial fb actuation')
                axs[1].legend()
                axs[2].plot(timePCS, tau_fb_component_ts[:,n,3*i+2], label=f'segment {i+1}')
                axs[2].grid(True)
                axs[2].set_xlabel('t [s]')
                axs[2].set_ylabel(r"$\tau_{sh}$ [$N\cdot m$]")
                axs[2].set_title('Shear fb actuation')
                axs[2].legend()
            plt.tight_layout()
            plt.savefig(plots_folder/f'Actuation_fb_after_robot_{n+1}', bbox_inches='tight')
            #plt.show()
            plt.close()

    # Animate robots
    for n in range(n_robots):
        flag = True if n == n_robots - 1 else False
        animate_robot_matplotlib(
            robot = robots_system_opt.get_robot(n),
            t_list = saveat,
            q_list = q_ts[:,n],
            interval = 1e-3, 
            slider = False,
            animation = True,
            #show = flag,
            show = False,
            duration = 10,
            fps = 30,
            save_path = plots_folder/f'animation_after_robot_{n+1}.gif',
        )
else:
    print('[simulation skipped]')

# Correct signature for loss function
Loss = partial(Loss, robots=robots_system_opt, controller=mlp_controller_opt, map=map_opt)

# Test RMSE on the test set after optimization
_, metrics = Loss(
    params_optimiz=params_optimiz_opt, 
    data_batch=test_set,
)
RMSE = onp.sqrt(metrics["MSE"])
pred = onp.array(metrics["predictions"])
labels = onp.array(metrics["labels"])
match map_to_train:
    case 'reconstruction':
        Q_encoded = encoder_opt(test_set["y"][69])
        Qd_encoded = encoder_opt.compute_jacobian(test_set["y"][69]) @ test_set["yd"][69]
        Qdd_encoded = encoder_opt.forward_xdd(test_set["y"][69], test_set["yd"][69], test_set["ydd"][69])
        y_decoded = decoder_opt(Q_encoded)
        yd_decoded = decoder_opt.compute_jacobian(Q_encoded) @ Qd_encoded
        ydd_decoded = decoder_opt.forward_xdd(Q_encoded, Qd_encoded, Qdd_encoded)
    case 'norm_flow':
        Q_encoded, Qd_encoded, Qdd_encoded = map_opt.forward_with_derivatives(test_set["y"][69], test_set["yd"][69], test_set["ydd"][69])
        y_decoded, yd_decoded, ydd_decoded = map_opt.inverse_with_derivatives(Q_encoded, Qd_encoded, Qdd_encoded)
    case _:
        Q_encoded = A_opt @ test_set["y"][69] + c_opt
        Qd_encoded = A_opt @ test_set["yd"][69]
        Qdd_encoded = A_opt @ test_set["ydd"][69]
        y_decoded = jnp.linalg.solve(A_opt, (Q_encoded - c_opt).T).T
        yd_decoded = jnp.linalg.solve(A_opt, Qd_encoded.T).T
        ydd_decoded = jnp.linalg.solve(A_opt, Qdd_encoded.T).T

print(f'Test accuracy: RMSE = {RMSE:.6e}')
print(f'Example:\n'
      f'    datapoint (y, yd, ydd):   ({onp.array(test_set["y"][69])}, {onp.array(test_set["yd"][69])}, {onp.array(test_set["ydd"][69])})\n'
      f'    encoding (Q, Qd, Qdd):    ({onp.array(Q_encoded)}, {onp.array(Qd_encoded)}, {onp.array(Qdd_encoded)})\n'
      f'    decoding (y_, yd_, ydd_): ({onp.array(y_decoded)}, {onp.array(yd_decoded)}, {onp.array(ydd_decoded)})\n\n'
      f'    prediction ydd_hat: {pred[69]}\n'
      f'    label ydd:          {labels[69]}\n'
      f'    error |e|:          {onp.abs( labels[69] - pred[69] )}\n'
)

# Compute actuation power mean squared value on the test set after optimization
match map_to_train:
    case 'reconstruction':
        Q_test_power, Qd_test_power = encoder_opt.forward_xd_batch(test_set["y"], test_set["yd"]) # shape (testset_size, 3*n_pcs*n_robots)
    case 'norm_flow':
        Q_test_power, Qd_test_power = map_opt.forward_with_derivatives_batch(test_set["y"], test_set["yd"]) # shape (testset_size, 3*n_pcs*n_robots)
    case _:
        Q_test_power = test_set["y"] @ jnp.transpose(A_opt) + c_opt # shape (testset_size, 3*n_pcs*n_robots)
        Qd_test_power = test_set["yd"] @ jnp.transpose(A_opt) # shape (testset_size, 3*n_pcs*n_robots)

Z_test_power = jnp.concatenate([Q_test_power, Qd_test_power], axis=1) # shape (testset_size, 2*3*n_pcs*n_robots)
if fb_controller_to_train == 'tanh_simple' or fb_controller_to_train == 'linear_simple':
    fb_contr_inp = Q_test_power # shape (testset_size, 3*n_pcs*n_robots)
else:
    fb_contr_inp = Z_test_power # shape (testset_size, 2*3*n_pcs*n_robots)

if ron_case == 'input':
    contr_inp = jnp.concatenate([fb_contr_inp, test_set["u"]], axis=1) # shape (testset_size, 3*n_pcs*n_robots+1) or (testset_size, 2*3*n_pcs*n_robots+1)
else:
    contr_inp = fb_contr_inp # shape (testset_size, 3*n_pcs*n_robots) or (testset_size, 2*3*n_pcs*n_robots)

if train_unique_controller:
    Tau_test_power = mlp_controller_opt.forward_batch(contr_inp) # shape (testset_size, 3*n_pcs*n_robots)
    power = jnp.sum(Tau_test_power * Qd_test_power, axis=1) # shape (testset_size,)
    power_msv_after = jnp.mean(power**2) # scalar
else:
    Tau_test_power_fb = fb_mlp_controller_opt.forward_batch(fb_contr_inp) # shape (testset_size, 3*n_pcs*n_robots)
    Tau_test_power_ff = ff_mlp_controller_opt.forward_batch(test_set["u"]) # shape (testset_size, 3*n_pcs*n_robots)
    Tau_test_power = Tau_test_power_fb + Tau_test_power_ff # shape (testset_size, 3*n_pcs*n_robots)
    power = jnp.sum(Tau_test_power * Qd_test_power, axis=1) # shape (testset_size,)
    power_msv_after = jnp.mean(power**2) # scalar

    power_fb = jnp.sum(Tau_test_power_fb * Qd_test_power, axis=1) # shape (testset_size,)
    power_ff = jnp.sum(Tau_test_power_ff * Qd_test_power, axis=1) # shape (testset_size,)
    power_msv_after_fb = jnp.mean(power_fb**2) # scalar
    power_msv_after_ff = jnp.mean(power_ff**2) # scalar

# Compute reconstruction error
match map_to_train:
    case 'reconstruction':
        Q_encoded, Qd_encoded = encoder_opt.forward_xd_batch(test_set["y"], test_set["yd"])
        Qdd_encoded = encoder_opt.forward_xdd_batch(test_set["y"], test_set["yd"], test_set["ydd"])
        y_decoded, yd_decoded = decoder_opt.forward_xd_batch(Q_encoded, Qd_encoded)
        ydd_decoded = decoder_opt.forward_xdd_batch(Q_encoded, Qd_encoded, Qdd_encoded)
    case 'norm_flow':
        Q_encoded, Qd_encoded, Qdd_encoded = map_opt.forward_with_derivatives_batch(test_set["y"], test_set["yd"], test_set["ydd"])
        y_decoded, yd_decoded, ydd_decoded = map_opt.inverse_with_derivatives_batch(Q_encoded, Qd_encoded, Qdd_encoded)
    case _:
        Q_encoded = test_set["y"] @ jnp.transpose(A_opt) + c_opt
        Qd_encoded = test_set["yd"] @ jnp.transpose(A_opt)
        Qdd_encoded = test_set["ydd"] @ jnp.transpose(A_opt)
        y_decoded = jnp.linalg.solve(A_opt, (Q_encoded - c_opt).T).T
        yd_decoded = jnp.linalg.solve(A_opt, Qd_encoded.T).T
        ydd_decoded = jnp.linalg.solve(A_opt, Qdd_encoded.T).T

reconstruction_rmse_y = jnp.sqrt(jnp.mean(jnp.sum((test_set["y"] - y_decoded)**2, axis=1)))
reconstruction_rmse_yd = jnp.sqrt(jnp.mean(jnp.sum((test_set["yd"] - yd_decoded)**2, axis=1)))
reconstruction_rmse_ydd = jnp.sqrt(jnp.mean(jnp.sum((test_set["ydd"] - ydd_decoded)**2, axis=1)))

# Save some metrics
suffix_log = ''
if load_experiment:
    elatime_optimiz = '(optimization not performed)'
    suffix_log = '_NoOptimization'

with open(data_folder/f'metrics{suffix_log}.txt', 'w') as file:
    file.write(f"SETUP\n")
    file.write(f"   Dataset:          {ron_dataset}\n")
    file.write(f"   RON case:         {ron_case}\n")
    file.write(f"   Latent dimension: {n_ron}\n")
    file.write(f"   Number of robots: {n_robots}\n")
    file.write(f"   PCS segments:     {n_pcs}\n")
    if not train_robots:
        file.write(f"   ! Robots were not optimized !\n")
    file.write(f"   Mapping:          {map_to_train}")
    if map_to_train == 'reconstruction':
        file.write(f", (reconstruction loss up to {reconstruction_type})\n")
    else:
        file.write(f"\n")
    if train_unique_controller:
        file.write(f"   Controller:       {fb_controller_to_train} (unique)\n\n")
    else:
        file.write(f"   Controller:       {fb_controller_to_train} (fb) + {ff_controller_to_train} (ff)\n\n")
    file.write(f"METRICS AFTER OPTIMIZATION\n")
    file.write(f"   Elapsed time for optimization:               {elatime_optimiz}\n")
    file.write(f"   Final test RMS error:                        {RMSE}\n")
    if train_unique_controller:
        file.write(f"   Final test RMS power:                        {onp.sqrt(power_msv_after)}\n")
    else:
        file.write(f"   Final test RMS power:                        {onp.sqrt(power_msv_after)} (fb: {onp.sqrt(power_msv_after_fb)}, ff: {onp.sqrt(power_msv_after_ff)})\n")
    file.write(f"   Final test RMSE reconstruction (y, yd, ydd): ({reconstruction_rmse_y}, {reconstruction_rmse_yd}, {reconstruction_rmse_ydd})\n")