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
train = True # if True, perform training. Otherwise, test saved model


# =====================================================
# Datasets
# =====================================================

# Load MNIST dataset
train_set_raw, test_set_raw = load_mnist_data(dataset_folder/'MNIST') # shape (n_images, 1, 28, 28)

# Convert (n_imgs, 1, 28, 28) --> (n_imgs, 784)
train_set = copy.deepcopy(train_set_raw)
test_set  = copy.deepcopy(test_set_raw)
train_set["images"] = train_set["images"].reshape(train_set["images"].shape[0], -1)
test_set["images"] = test_set["images"].reshape(test_set["images"].shape[0], -1)

# Convert to jax
train_set["images"] = jnp.array(train_set["images"])
train_set["labels"] = jnp.array(train_set["labels"])
train_set_size = len(train_set["labels"])

test_set["images"] = jnp.array(test_set["images"])
test_set["labels"] = jnp.array(test_set["labels"])
test_set_size = len(test_set["labels"])


# =====================================================
# Define the reservoir
# =====================================================

# Define robot
n_pcs = 2
L0 = 1e-1 * jnp.ones(n_pcs)
D0 = jnp.diag(jnp.tile(jnp.array([5e-6, 5e-3, 5e-3]), n_pcs))
r0 = 2e-2 * jnp.ones(n_pcs)
rho0 = 1070 * jnp.ones(n_pcs)
E0 = 2e3 * jnp.ones(n_pcs)
G0 = 1e3 * jnp.ones(n_pcs)

pcs_parameters = {
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
    params = pcs_parameters,
    order_gauss = 5
)

# Define mapping
@jax.jit
def map_direct(y, yd):
    q, qd = y, yd
    return q, qd

@jax.jit
def map_inverse(q, qd):
    y, yd = q, qd
    return y, yd 

# Define controller
@jax.jit
def controller(z, u):
    W = jnp.zeros((int(len(z)/2), len(z)))
    b = jnp.zeros(int(len(z)/2))
    V = jnp.ones(int(len(z)/2))
    return jnp.tanh(W @ z + b + V * u)

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

reservoir_forward = jax.jit(jax.vmap(reservoir, in_axes=(0,None,None,None))) # vmap roservoir's forward
time_u = jnp.linspace(0, dt_u * (len(train_set["images"][0]) - 1), len(train_set["images"][0])) # define time vector for the input image 
saveat = jnp.arange(0, time_u[-1], dt_u) # for saving simulation results


# =====================================================
# Training
# =====================================================

if train:
    # Train the output layer (classifier) (1): pass all the inputs in the train set to the model
    print(f'--- Generating previsions for training ---')
    key, subkey = jax.random.split(key)
    batch_ids = batch_indx_generator(subkey, train_set_size, batch_size=100) # create indices for the batches
    activations, labels = [], []
    start = time.perf_counter()
    for i in tqdm(range(len(batch_ids)), 'Model forward'):
        batch_i_ids = batch_ids[i]
        train_batch = extract_batch(train_set, batch_i_ids)
        _, _, _, _, activations_batch = reservoir_forward(train_batch["images"], time_u, saveat, dt_sim) # shape (batch_size, num_hidden_units)
        activations.append(activations_batch)
        labels.append(train_batch["labels"])

    activations = jnp.concatenate(activations) # shape (num_train_images, num_hidden_units)
    labels = jnp.concatenate(labels) # shape (num_train_images,)
    stop = time.perf_counter() 
    print(f'Elapsed time: {stop-start}')

    # Train the output layer (classifier) (2): logistic regression of the output layer
    print(f'\n--- Training the classifier (regression) ---')
    start = time.perf_counter()
    scaler = preprocessing.StandardScaler().fit(onp.array(activations))
    activations = scaler.transform(onp.array(activations))
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
# Testing
# =====================================================
if train:
    print()
print(f'--- Evaluating perfomances ---')

# If training was not performed, load saved data
if not train:
    scaler = joblib.load(data_folder/'scaler.pkl')
    classifier = joblib.load(data_folder/'classifier.pkl')

# Forward on the test set
activations, labels = [], []
key, subkey = jax.random.split(key)
batch_ids = batch_indx_generator(subkey, test_set_size, batch_size=5000) # create indices for the batches
start = time.perf_counter()
for i in tqdm(range(len(batch_ids)), 'Model forward'):
    batch_i_ids = batch_ids[i]
    test_batch = extract_batch(test_set, batch_i_ids)
    _, _, _, _, activations_batch = reservoir_forward(test_batch["images"], time_u, saveat) # shape (batch_size, num_hidden_units)
    activations.append(activations_batch)
    labels.append(test_batch["labels"])

activations = jnp.concatenate(activations) # shape (num_test_images, num_hidden_units)
labels = jnp.concatenate(labels) # shape (num_test_images,)
stop = time.perf_counter() 
print(f'Elapsed time: {stop-start}')

# Accuracy
test_accuracy = classifier.score(activations, labels)
print(f'Accuracy on the test set: {test_accuracy}')


# =====================================================
# Visualization (example)
# =====================================================
print(f'\n--- Example ---')

# Take one example and try inference
example_idx = 1234
image = test_set["images"][example_idx] # shape (784,)
image_raw = test_set_raw["images"][example_idx,0] # shape (28, 28)
label = test_set["labels"][example_idx]

start = time.perf_counter()
time_ts, state_reservoir_ts, state_pcs_ts, actuation_ts, activations = reservoir(image, time_u, saveat)
y_ts, yd_ts = jnp.split(state_reservoir_ts, 2, axis=1)
q_ts, qd_ts = jnp.split(state_pcs_ts, 2, axis=1)
stop = time.perf_counter()
print(f'Elapsed time (simulation): {stop-start}')

activations = activations[None,:] # sklear requires input (n_inputs, dim_inputs)
activations = scaler.transform(activations)
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
fig, axs = plt.subplots(3,2, figsize=(12,9))
for i, ax in enumerate(axs.flatten()):
    ax.plot(time_ts, actuation_ts[:,i], 'r')
    ax.grid(True)
    ax.set_xlabel('t [s]')
    ax.set_ylabel(r'$\tau$')
    ax.set_title(f'Component {i+1}')
plt.tight_layout()
plt.savefig(plots_folder/'Example_inference_actuation', bbox_inches='tight') 
#plt.show()

# Show reservoir input u(t)
plt.figure()
plt.plot(time_u, image, 'b')
plt.grid(True)
plt.xlabel('t [s]')
plt.ylabel('u')
plt.title('Reservoir input')
plt.tight_layout()
plt.savefig(plots_folder/'Example_inference_input', bbox_inches='tight') 
plt.show()

# Show robot animation
animate_robot_matplotlib(
    robot = robot,
    t_list = time_ts,
    q_list = q_ts,
    interval = 1e-3, 
    slider = False,
    animation = True,
    show = True,
    duration = 10,
    fps = 30,
    save_path = plots_folder/'Example_inference_animation.gif',
)