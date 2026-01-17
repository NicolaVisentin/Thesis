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
        (! map type must be specified by hand in `map_type`).

    2.  Loads the MNIST dataset and extract part of it. In particular, a portion `train_set_portion` is extracted from the 
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

# General
example_idx = 2 # if it is an integer i, loads the i-th image from MNIST test set. Otherwise 'black' for black image
train_set_portion = 6000 # fraction (or number of images) of the original train set (60 000 images) to use. If 1: full dataset
test_set_portion = 6000 # fraction (or number of images) of the original test set (10 000 images) to use. If 1: full dataset
batch_size = 100 # batch size for training and testing. Should be as high as possible, consistently with pc memory and datasets sizes

# Output layer (scaler + classifier)
experiment_name = 'pre_trained_output_layer' # name of the experiment to save/load
train = False # if True, perform training (output layer). Otherwise, test saved 'experiment_name' model

# Reservoir (robot + map + controller)
load_model_path = saved_data_folder/'equation-error_optimization'/'main'/'T10' # choose the reservoir to load (robot + map + controller)
map_type = 'linear' # 'linear', 'encoder-decoder', 'bijective'

# Rename folders for plots/data
plots_folder = plots_folder/experiment_name
data_folder = data_folder/experiment_name
data_folder.mkdir(parents=True, exist_ok=True)
plots_folder.mkdir(parents=True, exist_ok=True)


# =====================================================
# Datasets
# =====================================================

# Load MNIST dataset
train_set, test_set = load_mnist_data(dataset_folder/'MNIST') # shape (n_images, 1, 28, 28)

# Convert (n_imgs, 1, 28, 28) --> (n_imgs, 784)
train_set["images"] = train_set["images"].reshape(train_set["images"].shape[0], -1)
test_set["images"] = test_set["images"].reshape(test_set["images"].shape[0], -1)

# Convert to jax
train_set["images"] = jnp.array(train_set["images"])
train_set["labels"] = jnp.array(train_set["labels"])

test_set["images"] = jnp.array(test_set["images"])
test_set["labels"] = jnp.array(test_set["labels"])

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

# Define robot
data_robot_load = onp.load(load_model_path/'optimal_data_robot.npz')

L = jnp.array(data_robot_load['L'])
D = jnp.array(data_robot_load['D'])
r = jnp.array(data_robot_load['r'])
rho = jnp.array(data_robot_load['rho'])
E = jnp.array(data_robot_load['E'])
G = jnp.array(data_robot_load['G'])
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
        A = jnp.array(data_map['A'])
        c = jnp.array(data_map['c'])

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

# Define controller
mlp_controller_loader = MLP(key, [1, 1]) # instance just for loading parameters
p_controller = mlp_controller_loader.load_params(load_model_path/'optimal_data_controller.npz') # tuple ((W1, b1), (W2, b2), ...)
layers_dim = []
for i, layer in enumerate(p_controller):
    W = layer[0] # shape (n_out_layer, n_in_layer)
    layers_dim.append(W.shape[1]) # n_in_layer
    if i == 0:
        n_input = W.shape[1] # save input dim for the controller
layers_dim.append(W.shape[0]) # last layer: add output dimension (i.e. n_out_layer for the last layer)

mlp_controller_dummy = MLP(key, layers_dim)
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

# Instantiate the reservoir
reservoir = pcsReservoir(
    robot=robot,
    map_direct=map_direct,
    map_inverse=map_inverse,
    controller=controller
)

# Other stuff
dt_u = 0.042 # time step for the input u. (in the RON paper dt = 0.042 s)
dt_sim = 1e-4 # time step for the simulation

reservoir_forward = jax.jit(jax.vmap(reservoir, in_axes=(0,None,None,None))) # vmap reservoir's forward
time_u = jnp.linspace(0, dt_u * (len(train_set["images"][0]) - 1), len(train_set["images"][0])) # define time vector for the input image 
saveat = jnp.arange(0, time_u[-1], dt_u) # for saving simulation results


# =====================================================
# Training
# =====================================================

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
    print(f'Elapsed time: {stop-start}')

    # Train the output layer (classifier) (2): logistic regression of the output layer
    print(f'\n--- Training the classifier (regression) ---')
    start = time.perf_counter()
    scaler = preprocessing.StandardScaler().fit(onp.array(last_states))
    activations = scaler.transform(onp.array(last_states))
    classifier = LogisticRegression(max_iter=1000).fit(onp.array(activations), onp.array(labels))
    stop = time.perf_counter()
    print(f'Elapsed time: {stop-start}')

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
print(f'Elapsed time: {stop-start}')

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
time_ts, state_reservoir_ts, state_pcs_ts, actuation_ts, last_states = reservoir(image, time_u, saveat, dt_sim)
y_ts, yd_ts = jnp.split(state_reservoir_ts, 2, axis=1)
q_ts, qd_ts = jnp.split(state_pcs_ts, 2, axis=1)
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
    ax2.plot(time_u, image, 'k', alpha=0.3, label=r'reservoir input $u(t)$')
    ax2.set_ylabel(r'$u$')
    ax2.set_ylim([-0.1, 2])

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
time_ts2, state_reservoir_ts2, state_pcs_ts2, actuation_ts2, last_states2 = reservoir(image2, time_u, saveat, dt_sim)
y_ts2, yd_ts2 = jnp.split(state_reservoir_ts2, 2, axis=1)
q_ts2, qd_ts2 = jnp.split(state_pcs_ts2, 2, axis=1)
stop = time.perf_counter()
print(f'Elapsed time (simulation): {stop-start}')

# Compare reservoir/robot evolutions
fig, axs = plt.subplots(3,2, figsize=(12,9))
for i, ax in enumerate(axs.flatten()):
    ax.plot(time_ts, y_ts[:,i], 'b', label='reservoir (ex. 1)')
    ax.plot(time_ts, q_ts[:,i], 'r', label='soft robot (ex. 1)')
    ax.plot(time_ts2, y_ts2[:,i], 'b--', label='reservoir (ex. 2)')
    ax.plot(time_ts2, q_ts2[:,i], 'r--', label='soft robot (ex. 2)')
    ax.grid(True)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('y, q')
    ax.set_title(f'Component {i+1}')
    ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(plots_folder/'Comparison_inference_evolution', bbox_inches='tight') 
#plt.show()

# Compare actuation signals tau(t)
fig, axs = plt.subplots(3,2, figsize=(16,13))
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

with open(data_folder/'performances.txt', 'w') as file:
    file.write(f"SETUP\n")
    file.write(f"   Train set size: {train_set_size}\n")
    file.write(f"   Test set size:  {test_set_size}\n\n")
    file.write(f"RESERVOIR PROPERTIES\n")
    file.write(f"   Model path: {load_model_path}\n")
    file.write(f"   Dimension:  {3*n_pcs}\n")
    file.write(f"   Map:        {map_type}\n\n")
    file.write(f"METRICS\n")
    file.write(f"   Accuracy (train set): {train_accuracy}\n")
    file.write(f"   Accuracy (test set):  {test_accuracy}\n")


# =========================================================
# Show all plots
# =========================================================
plt.show()