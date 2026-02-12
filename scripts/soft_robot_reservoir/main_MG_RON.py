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
import copy

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

from pathlib import Path
from tqdm import tqdm
import sys
import time

from soromox.systems.my_systems import PlanarPCS_simple
from reservoir import pcsReservoir

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
data_folder = main_folder/'saved data'/curr_folder.stem/Path(__file__).stem        # folder for saving data
saved_data_folder = main_folder/'saved data'                                       # folder with saved data (trained architectures)

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
"""
This script:
    1.  Takes a certain pcs reservoir's architecture (robot + map + controller) specified by the user in `load_model_path`
        (! map type and controller type must be specified by hand in `map_type` and `fb_controller_type`, `ff_controller_type`).

    2.  Loads the Mackey-Glass dataset.

    3a. If `train` is True, output layer (scaler + predictor) is trained with the given reservoir. Trained scaler and 
        predictor are then saved in data folder named `experiment_name`. 
    3b. If `train` is False, loads a given output layer (scaler + predictor) from data folder named `experiment_name`.

    4.  The full architecture (reservoir + output layer) is tested on a test sequence.

    5.  Saves plots and metrics on the test sequence (and train sequence if training was performed) in `experiment_name` data and plots 
        folders. Also saves the dynamics of the reservoir during inference.
"""

# General
dt_u = 0.05 # time step for the input u. (in the RON paper dt = 0.17 s)
Nw = 200 # washout steps for the Mackey-Glass task
Nl = 84 # prediction lag for the Mackey-Glass task

# Output layer (scaler + predictor)
experiment_name = 'TEST' # name of the experiment to save/load
train = True # if True, perform training (output layer). Otherwise, test saved 'experiment_name' model

# Reservoir (robot + map + controller)
load_model_path = saved_data_folder/'equation-error_optimization'/'main_MG_RON'/'A9' # choose the reservoir to load (robot + map + controller)
map_type = 'none' # 'linear', 'encoder-decoder', 'bijective', 'none'
controller_type = 'ff' # if 'unique': tau = tau_tot(z,u). If 'fb+ff': tau = tau_fb(z) + tau_ff(u). If 'ff': tau = tau_ff(u) (randomly initialized tanh(V*u+d)) !!! If 'unique', the controller tau_tot is defined in fb_controller_type
fb_controller_type = 'tanh_simple' # 'linear_simple', 'linear_complete', 'tanh_simple', 'tanh_complete', 'mlp'
ff_controller_type = 'mlp' # 'linear', 'tanh', 'mlp'

# Rename folders for plots/data
plots_folder = plots_folder/experiment_name
data_folder = data_folder/experiment_name
data_folder.mkdir(parents=True, exist_ok=True)
plots_folder.mkdir(parents=True, exist_ok=True)


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
train_dataset = jnp.array([train_dataset], dtype=jnp.float64).squeeze() # sequence from k=0 to k=N-Nl-1. Shape (N-Nl,)
valid_dataset = jnp.array([valid_dataset], dtype=jnp.float64).squeeze() # sequence from k=0 to k=N-Nl-1. Shape (N-Nl,)
test_dataset = jnp.array([test_dataset], dtype=jnp.float64).squeeze() # sequence from k=0 to k=N-Nl-1. Shape (N-Nl,)

N_train = len(train_dataset) # N for the train set sequence
N_test = len(test_dataset) # N for the test set sequence


# =====================================================
# Define the reservoir
# =====================================================

# Define robot
data_robot_load = onp.load(load_model_path/'optimal_data_robot.npz')

L = jnp.array(data_robot_load['L'], dtype=jnp.float64)
D = jnp.array(data_robot_load['D'], dtype=jnp.float64)
r = jnp.array(data_robot_load['r'], dtype=jnp.float64)
rho = jnp.array(data_robot_load['rho'], dtype=jnp.float64)
E = jnp.array(data_robot_load['E'], dtype=jnp.float64)
G = jnp.array(data_robot_load['G'], dtype=jnp.float64)
n_pcs = len(L)

pcs_parameters = {
    "th0": jnp.array(jnp.pi/2),
    "L": L,
    "r": r,
    "rho": rho,
    "g": jnp.array([0.0, 9.81]), # !! gravity UP !!
    "E": E,
    "G": G,
    "D": D
}
robot = PlanarPCS_simple(
    num_segments = n_pcs,
    params = pcs_parameters,
    order_gauss = 5
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
        masks = create_alternating_masks(input_dim=3*n_pcs, num_layers=n_layers)
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
        if n_input == 3*n_pcs + 1:
            q, qd = jnp.split(z, 2)
            input_controller = jnp.concatenate([q, jnp.array([u])])
        else:
            input_controller = jnp.concatenate([z, jnp.array([u])])
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
        tau_ff = mlp_ff_controller(jnp.array([u]))
        if n_input_fb == 3*n_pcs:
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
    scal_input = jnp.tile(jnp.array([0.001, 0.1, 0.01]), n_pcs)
    V = scal_input[:,None] * jax.random.uniform(key_V, shape=(3*n_pcs,1), minval=0.0, maxval=1.0) # random input-to-hidden weights
    d = scal_input * jax.random.uniform(key_d, shape=(3*n_pcs,), minval=-1.0, maxval=1.0) # random input-to-hodden bias
    def controller(z, u, V, d):
        tau_ff = jnp.tanh(V @ jnp.array([u]) + d)
        return tau_ff
    controller = jax.jit(partial(controller, V=V, d=d))

# Instantiate the reservoir
reservoir = pcsReservoir(
    robot=robot,
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

if train:
    # Train the output layer (predictor) (1): pass the train input sequence to the model
    print(f'--- Generating activations for training ---')
    start = time.perf_counter()
    (
        time_ts,
        state_reservoir_ts, # reservoir's states evolution from k=0 to k=N-Nl-1. Shape (N-Nl, n_hid)
        state_pcs_ts, # pcs's states evolution from k=0 to k=N-Nl-1. Shape (N-Nl, 3*n_pcs)
        actuation_ts, # pcs actuation. Shape (N-Nl, 3*n_pcs)
        _
     ) = reservoir(train_dataset, time_u_train, saveat_train, dt_sim)
    activations = state_reservoir_ts[Nw:] # remove the initial washout steps. Shape (N-Nl-Nw, n_hid). It's the reservoir's states evolution from k=Nw to k=N-Nl-1
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
    joblib.dump(scaler, data_folder/'scaler.pkl')
    joblib.dump(predictor, data_folder/'predictor.pkl')

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
    scaler = joblib.load(data_folder/'scaler.pkl')
    predictor = joblib.load(data_folder/'predictor.pkl')

# Forward on the test set
print(f'--- Evaluating perfomances (test set) ---')
start = time.perf_counter()
(
    time_ts,
    state_reservoir_ts, # reservoir's states evolution from k=0 to k=N-Nl-1. Shape (N-Nl, n_hid)
    state_pcs_ts, # pcs's states evolution from k=0 to k=N-Nl-1. Shape (N-Nl, 3*n_pcs)
    actuation_ts, # pcs actuation. Shape (N-Nl, 3*n_pcs)
    _
) = reservoir(test_dataset, time_u_test, saveat_test, dt_sim)

activations = state_reservoir_ts[Nw:] # remove the initial washout steps. Shape (N-Nl-Nw, n_hid). It's the reservoir's states evolution from k=Nw to k=N-Nl-1
activations.block_until_ready()
stop = time.perf_counter() 
elatime_forward_pass_testing = stop - start
print(f'Elapsed time: {elatime_forward_pass_testing}')

# Prediction and test accuracy
activations = onp.array(activations)
activations = scaler.transform(activations)
pred = predictor.predict(activations)
rmse = jnp.sqrt(jnp.mean((pred - test_target) ** 2))
rms_target = jnp.sqrt(jnp.mean(test_target ** 2))
test_nrmse = (rmse / rms_target)
print(f'Test NRMSE: {test_nrmse}')


# =====================================================
# Show results of the test
# =====================================================

# Prepare variables
y_ts, yd_ts = jnp.split(state_reservoir_ts, 2, axis=1) # reservoir states
q_ts, qd_ts = jnp.split(state_pcs_ts, 2, axis=1) # robot states
full_time = dt_u * onp.arange(0, N_test + Nl)
full_sequence = onp.concatenate([onp.array(test_dataset), test_target[-Nl:]]) # full MG test sequence

# Show predicted sequence
fig, ax = plt.subplots(1,1, figsize=(20,6))
ax.plot(full_time, full_sequence, 'k--', label='full sequence')
ax.plot(full_time[Nw:N_test-Nl], full_sequence[Nw:N_test-Nl], 'k', label='test sequence')
ax.plot(full_time[Nw+Nl:], pred, 'r', label='predicted sequence')
ax.grid(True)
ax.set_xlabel('t [s]')
ax.set_ylabel('x')
ax.set_title('Mackey-Glass sequence')
ax.legend()

plt.tight_layout()
plt.savefig(plots_folder/'Prediction', bbox_inches='tight')
#plt.show()

# Show reservoir/robot evolution
fig, axs = plt.subplots(3,2, figsize=(12,9))
for i, ax in enumerate(axs.flatten()):
    ax.plot(time_ts, y_ts[:,i], 'b', label='reservoir')
    ax.plot(time_ts, q_ts[:,i], 'r', label='soft robot')
    ax.grid(True)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('y, q')
    ax.set_title(f'Component {i+1}')
    ax.legend()
plt.tight_layout()
plt.savefig(plots_folder/'Example_inference_evolution', bbox_inches='tight') 
#plt.show()

# Show actuation signal tau(t)
fig, axs = plt.subplots(3,2, figsize=(16,13))
for i, ax in enumerate(axs.flatten()):
    ax2 = ax.twinx()
    ax2.plot(time_ts, test_dataset, 'k', alpha=0.3, label=r'reservoir input $u(t)$')
    ax2.set_ylabel(r'$u$')
    ax2.set_ylim([-0.6, 0.4])

    ax.plot(time_ts, actuation_ts[:,i], 'r', label=r'robot actuation $\tau(t)$')
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
animate_robot_matplotlib(
    robot = robot,
    t_list = time_ts,
    q_list = q_ts,
    interval = 1e-3, 
    slider = False,
    animation = True,
    show = False,
    duration = 10,
    fps = 30,
    save_path = plots_folder/'Example_inference_animation.gif',
)


# =========================================================
# Save text file with performances and data
# =========================================================

if not train:
    N_train = '(training was not performed)'
    train_nrmse = '(training was not performed)'
    elatime_forward_pass_training = '(training was not performed)'
    elatime_train_output_layer = '(training was not performed)'

with open(data_folder/'performances.txt', 'w') as file:
    file.write(f"SETUP\n")
    file.write(f"   Train set size: {N_train}\n")
    file.write(f"   Test set size:  {N_test}\n\n")
    file.write(f"RESERVOIR PROPERTIES\n")
    file.write(f"   Model path: {load_model_path}\n")
    file.write(f"   Dimension:  {3*n_pcs}\n")
    file.write(f"   Map:        {map_type}\n")
    if controller_type == 'unique':
        file.write(f"   Controller: {fb_controller_type} (unique)\n\n")
    elif controller_type == 'fb+ff':
        file.write(f"   Controller: {fb_controller_type} (fb) + {ff_controller_type} (ff)\n\n")
    else:
        file.write(f"   Controller: no fb + random ff\n\n")
    file.write(f"METRICS\n")
    file.write(f"   Elapsed time forward pass (train set): {elatime_forward_pass_training}\n")
    file.write(f"   Elapsed time training output layer:    {elatime_train_output_layer}\n")
    file.write(f"   Elapsed time forward pass (test set):  {elatime_forward_pass_testing}\n")
    file.write(f"   NRMSE (train set): {train_nrmse}\n")
    file.write(f"   NRMSE (test set):  {test_nrmse}\n")


# =========================================================
# Show all plots
# =========================================================
plt.show()