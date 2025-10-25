# =====================================================
# Setup
# =====================================================

# Choose device (cpu or gpu)
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Imports
import jax
import jax.numpy as jnp
from diffrax import Tsit5, Euler, Heun, Midpoint, Ralston, Bosh3, Dopri5, Dopri8, ImplicitEuler, Kvaerno3
from diffrax import ConstantStepSize, PIDController
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pickle
from tqdm import tqdm
import time

from evosax.algorithms.distribution_based.cma_es import CMA_ES
from utilis import *

from soromox.systems.planar_pcs import PlanarPCS 
from soromox.systems.planar_pcs_simplified import PlanarPCS_simple

# Set random seed
seed = 123
key = jax.random.key(seed)

# jax settings
jax.config.update("jax_enable_x64", True)  # double precision
jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)

# Folders
curr_folder = Path(__file__).parent # current folder
main_folder = curr_folder.parent

plots_folder = main_folder/'plots and videos'/Path(__file__).stem # folder for plots and videos
plots_folder.mkdir(parents=True, exist_ok=True)

# data_folder = main_folder/'saved data'/Path(__file__).stem # folder for saving data
# data_folder.mkdir(parents=True, exist_ok=True)

# Functions for plotting
def draw_robot(robot: PlanarPCS | PlanarPCS_simple, q: Array, num_points: int = 50):
    L_max = jnp.sum(robot.L)
    s_ps = jnp.linspace(0, L_max, num_points)

    chi_ps = robot.forward_kinematics_batched(q, s_ps) # (N,3)
    curve = np.array(chi_ps[:, 1:], dtype=np.float64)  # (N,2)
    pos_tip = curve[-1]                                # [x_tip, y_tip]

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
        t_old = np.linspace(0, 1, len(target))
        t_new = np.linspace(0, 1, len(q_list))
        target = np.interp(t_new, t_old, target)

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
        ax_slider = fig.add_axes([0.2, 0.0, 0.6, 0.03])  # [left, bottom, width, height]
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
# User settings
# =====================================================

use_softplus = False          # use softplus to force positive parameters
use_lax_scan = False          # jax.lax.scan or for loop
RobotModel = PlanarPCS_simple # planar PCS model (with or without Coriolis effect): PlanarPCS, PlanarPCS_simple


# =====================================================
# Target creation
# =====================================================

# End effector target (x coordinate)
t0 = 0.0
t1 = 5.0
dt_target = 1e-3
t_target = jnp.arange(t0, t1, dt_target)

c = 0.5
k = 29
wd = jnp.sqrt(4*k-c**2)/2
target = 0.2 * jnp.exp(-c/2 * t_target) * jnp.cos(wd * t_target)


# =====================================================
# Robot before optimization
# =====================================================
print('--- INITIAL SIMULATION ---')

# Instantiate robot
N = 2 # number of segments

L_default = jnp.array([1.0e-1, 1.0e-1])
L_scale = 1.6
L = L_scale * L_default

D_default = jnp.diag(jnp.array([1.0e-4, 1.0e-1, 1.0e-1,
                                1.0e-4, 1.0e-1, 1.0e-1]))
D_scale = 2
D = D_scale * D_default

parameters = {
    "th0": jnp.array(jnp.pi/2),
    "L": L,
    "r": jnp.array([2e-2, 2e-2]),
    "rho": jnp.array([1070, 1070]),
    "g": jnp.array([0.0, 9.81]), # !! gravity UP !!
    "E": jnp.array([2e3, 2e3]),
    "G": 1e3 * jnp.ones((N,)),
    "D": D
}

robot = RobotModel(
    num_segments = N,
    params = parameters,
    order_gauss = 5
)

# Set simulation parameters
q0 = jnp.array([-jnp.pi * 7/11, jnp.pi * 7/13]) # initial configuration
qd0 = jnp.zeros_like(q0)                        # initial velocities
u = jnp.zeros_like(q0)                          # torques (actuation)

q0 = jnp.array([-5.0*jnp.pi, 0.2, 0.1,
                5.0*jnp.pi, 0.2, 0.1]) # k, S_x, S_y
qd0 = jnp.zeros_like(q0)
u = jnp.zeros_like(q0)

t0 = t0
t1 = t1
dt =1e-5
save_at = t_target
solver = Euler() # Tsit5(), Euler(), Heun(), Midpoint(), Ralston(), Bosh3(), Dopri5(), Dopri8()
#step_size = PIDController(rtol=1e-6, atol=1e-6, dtmin=1e-4, force_dtmin=True) # ConstantStepSize(), PIDController(rtol=, atol=)
step_size = ConstantStepSize()
max_steps = int(1e6)

# Simulation
print('Simulating robot...')
start = time.perf_counter()
ts, q_ts, _ = robot.resolve_upon_time(
    q0 = q0, 
    qd0 = qd0,
    u = u, 
    t0 = t0, 
    t1 = t1, 
    dt = dt, 
    saveat_ts = save_at,
    solver = solver,
    stepsize_controller = step_size,
    max_steps = max_steps
)
end = time.perf_counter()
print(f'Elapsed time (simulation): {end-start} s')

# Extract end effector x coordinate in time
chi_ee_ts = jax.vmap(robot.forward_kinematics, in_axes=(0,None))(q_ts, jnp.sum(L)) # chi = [th, x, y]. Shape (n_steps, 3)
x_ee_ts = chi_ee_ts[:,1]

# Compute index for discharging initial transient
idx_end_trans = np.argmax(ts>1.0)

# Error
MSE = np.mean((x_ee_ts[idx_end_trans:] - target[idx_end_trans:]) ** 2)
print(f'MSE before optimization: {1e4*MSE:.4f} cm^2')

# Plot results
plt.figure()
plt.plot(t_target, x_ee_ts, color='b', label='x end effector')
plt.plot(t_target, target, color='b', linestyle='--', label='target')
plt.xlabel('t [s]')
plt.ylabel('x [m]')
plt.grid(True)
plt.title(f'End effector position (before optimization)')
plt.figtext(0.5, -0.05, f"L={np.array(L)} m\n D={np.diag(D)} Pa*s", ha="center", va="top")
plt.legend()
plt.tight_layout()
plt.savefig(plots_folder/'End effector (before optimization)', bbox_inches='tight')
#plt.show()

animate_robot_matplotlib(
    robot = robot,
    t_list = t_target,
    q_list = q_ts,
    target = target,
    interval = 1e-2, 
    slider = True,
    animation = False,
    show = True
)


# =====================================================
# Define the problem (function to optimize)
# =====================================================

# Optimization parameters are scaling factors L_scale and D_scale for L and D. If use_softplus,
# they are forced to be positive and > than a certain threshold L_scale_min and D_scale_min.
L_scale_min = 1e-8
D_scale_min = 1e-8
params_min = jnp.array([L_scale_min, D_scale_min])

# Set cost function
@jax.jit
def Loss(params):
    # update robot
    # if use_softplus:
    #     params = params_min + jax.nn.softplus(params)
    L_scale = params[0]
    D_scale = params[1]
    L = L_default * L_scale
    D = D_default * D_scale
    robot_updated = robot.update_params({"L": L, "D": D})
    # simulation
    _, q_ts, _ = robot_updated.resolve_upon_time(
        q0 = q0, 
        qd0 = qd0,
        u = u, 
        t0 = t0, 
        t1 = t1, 
        dt = dt, 
        saveat_ts = save_at,
        solver = solver,
        stepsize_controller = step_size,
        max_steps = max_steps
    )
    # extract end effector x coordinate in time
    chi_ee_ts = jax.vmap(robot_updated.forward_kinematics, in_axes=(0,None))(q_ts, jnp.sum(L)) # chi = [th, x, y]. Shape (n_steps, 3)
    x_ee_ts = chi_ee_ts[:,1]
    # loss
    J = jnp.mean((x_ee_ts[idx_end_trans:] - target[idx_end_trans:]) ** 2) # ! discharge initial transient
    return J

# Set research bounds
min_vals = jnp.array([0.6, 0.8])
max_vals = jnp.array([2.5, 10])

if use_softplus:
    min_vals = (InverseSoftplus(min_vals[0] - L_scale_min).item(), InverseSoftplus(min_vals[1] - L_scale_min).item())
    max_vals = (InverseSoftplus(max_vals[0] - D_scale_min).item(), InverseSoftplus(max_vals[1] - D_scale_min).item())

# Instantiate the problem
problem = MyOptimizationProblem(
    cost_fn=Loss,
    lower_research_bound=min_vals,
    upper_research_bound=max_vals,
)


# =====================================================
# Define the optimization strategy (CMA-ES)
# =====================================================

# Set parameters
num_generations = 12 # number of iterations
population_size = 24 # dimension of the population

# Set info to save during the iterations
def metrics_fn(key, population, fitness, state, cma_params):
    idx = jnp.argmin(fitness)
    metrics = {
        "best_fitness": fitness[idx],   # best fitness among all individuals
        "means": state.mean,            # mean of the distribution (solution)
        "populations": population,      # population
        "fitness_populations": fitness, # fitness of all individuals
    }
    if use_softplus:
        metrics["means"] = params_min + jax.nn.softplus(metrics["means"])
        metrics["populations"] = params_min + jax.nn.softplus(metrics["populations"])

    return metrics

# Instantiate the search strategy and initialize a solution and state of the optimizer
solution0 = jnp.array([L_scale, D_scale])
if use_softplus:
    solution0 = InverseSoftplus(solution0-params_min)

cma = CMA_ES(
    population_size=population_size, 
    solution=solution0,
    metrics_fn=metrics_fn,
)
cma_params = cma.default_params # parameters for CMA-ES (e.g. learning rate for the mean, learning rate for the covariance...)

key, subkey = jax.random.split(key)
state0 = cma.init(subkey, solution0, cma_params) # e.g. current best solution, current best fitness, current generation counter...


# =====================================================
# Run the optimization
# =====================================================
print(F'\n--- OPTIMIZATION ---')
start = time.perf_counter()

if use_lax_scan:
    print('Optimizing...')
    # Single iteration step (for jax.lax.scan)
    def step(carry, key):
        # extract CMA-ES params and current CMA-ES state from carry
        state, cma_params = carry

        # sample a population basing on the current state
        key_ask, key_tell = jax.random.split(key, 2)
        population, state = cma.ask(key_ask, state, cma_params)
        population = jnp.clip(population, min_vals, max_vals) # clip population within the bounds
        
        # evaluate the new population (return fitness) and update the counter
        fitness = problem.eval(population)

        # update the state basing on the evaluation of the new population
        state, metrics = cma.tell(key_tell, population, fitness, state, cma_params)

        carry = (state, cma_params)
        out = metrics
        return carry, out

    # Execute the iterations
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num_generations)
    carry0 = (state0, cma_params)
    _, metrics = jax.lax.scan(
        step,
        carry0,
        keys,
    )

else:
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num_generations)
    state = state0
    metrics_list = []

    pbar = tqdm(range(num_generations), 'Optimization')
    for ii in pbar:
        # sample a population basing on the current state
        key_ask, key_tell = jax.random.split(keys[ii], 2)
        population, state = cma.ask(key_ask, state, cma_params)
        population = jnp.clip(population, min_vals, max_vals)  # clip bounds

        # evaluate the new population (return fitness) and update the counter
        fitness = problem.eval(population)

        # update the state basing on the evaluation of the new population
        state, metrics = cma.tell(key_tell, population, fitness, state, cma_params)

        # append metrics
        metrics_list.append(metrics)

        # print stuff
        L_scale_curr, D_scale_curr = metrics["means"]
        L_curr = L_scale_curr * L_default
        D_curr = D_scale_curr * D_default
        best_fitness_curr = metrics["best_fitness"]
        tqdm.write(
            f"Iter {ii:02d} | "
            f"L={L_curr} | "
            f"D={np.diag(D_curr)} | "
            f"best fitness={best_fitness_curr:.3e}"
        )
    
    # convert list of dicts into dict of arrays
    metrics = {
        key: jnp.stack([m[key] for m in metrics_list])
        for key in metrics_list[0]
    }

end = time.perf_counter()
print(f'Elapsed time (optimization): {end-start} s')

# Extract results
means = metrics["means"]                             # mean of the population. Shape (num_generations, num_dims)
fitness_means = problem.eval(means)                  # fitness of the mean. Shape (num_generations,)
populations = metrics["populations"]                 # population. Shape (num_generations, population_size, num_dims)
fitness_populations = metrics["fitness_populations"] # fitness of all individuals. Shape (num_generations, population_size)
best_fitness = metrics["best_fitness"]               # fitness of the best individual. Shape (num_generations,)

L_scale_opt, D_scale_opt = means[-1]
L_opt = L_scale_opt * L_default
D_opt = D_scale_opt * D_default
print(f"Optimal L: {L_opt} m | Optimal D: {np.diag(D_opt)} Pa*s")


# =====================================================
# Visualize results
# =====================================================
print('Plotting results...')

# Plot best fitness for each generation vs generations
plt.figure()
plt.plot(fitness_means, label='fitness of the solution')
plt.plot(best_fitness, label='best fitness in population')
plt.grid(True)
plt.xlabel('generation')
plt.ylabel('fitness')
plt.title('Fitness curve')
plt.legend()
plt.tight_layout()
plt.savefig(plots_folder/'fitness curve', bbox_inches='tight')
plt.show()

# Convert bounds if softplus is used
if use_softplus:
    low_bound_plot = params_min + jax.nn.softplus(problem.lower_research_bound)
    up_bound_plot = params_min + jax.nn.softplus(problem.upper_research_bound)
else:
    low_bound_plot = problem.lower_research_bound
    up_bound_plot = problem.upper_research_bound

# Plot 3D
x = np.linspace(low_bound_plot[0], up_bound_plot[0], 11)
y = np.linspace(low_bound_plot[1], up_bound_plot[1], 11)
X, Y = np.meshgrid(x, y)
points = np.stack([X.flatten(), Y.flatten()], axis=1)
Z = problem.eval(points).reshape(X.shape)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.75, zorder=1)
sol = ax.scatter(means[-1,0], means[-1,1], fitness_means[-1], color='r', s=30, zorder=2, label='solution')
cbar = fig.colorbar(surf, ax=ax, shrink=0.6)
cbar.set_label('fitness')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('fitness')
ax.set_xlim([low_bound_plot[0], up_bound_plot[0]])
ax.set_ylim([low_bound_plot[1], up_bound_plot[1]])
ax.legend()
plt.tight_layout()
plt.savefig(plots_folder/'solution', bbox_inches='tight')
with open(plots_folder/'solution', 'wb') as f:
          pickle.dump(plt.gcf(), f)
plt.show()

# Visualize and save the optimization animation
animate_evolution(
    problem=problem,
    means=means,
    populations=populations,
    fitness_populations=fitness_populations,
    duration=5,
    grid_res={"X": X, "Y": Y, "Z": Z},
    savepath=plots_folder/'animation.gif',
    show=True,
)


# =====================================================
# Robot simulation after optimization
# =====================================================
print('\n--- FINAL SIMULATION ---')

# Update robot with optimal parameters
robot_opt = robot.update_params({"L": L_opt, "D": D_opt})

# Simulation
print('Simulating robot...')
_, q_ts, _ = robot_opt.resolve_upon_time(
    q0 = q0, 
    qd0 = qd0,
    u = u, 
    t0 = t0, 
    t1 = t1, 
    dt = dt, 
    saveat_ts = save_at,
    solver = solver,
    stepsize_controller = step_size,
    max_steps = max_steps
)

# Extract end effector x coordinate in time
chi_ee_ts = jax.vmap(robot_opt.forward_kinematics, in_axes=(0,None))(q_ts, jnp.sum(L_opt)) # chi = [th, x, y]. Shape (n_steps, 3)
x_ee_ts = chi_ee_ts[:,1]

# Error
MSE = np.mean((x_ee_ts[idx_end_trans:] - target[idx_end_trans:]) ** 2)
print(f'MSE after optimization: {1e4*MSE:.4f} cm^2')

# Plot results
plt.figure()
plt.plot(t_target, x_ee_ts, color='b', label='x end effector')
plt.plot(t_target, target, color='b', linestyle='--', label='target')
plt.xlabel('t [s]')
plt.ylabel('x [m]')
plt.grid(True)
plt.title(f'End effector position (after optimization)')
plt.figtext(0.5, -0.05, f"L={L_opt} m\n D={np.diag(D_opt)} Pa*s", ha="center", va="top")
plt.legend()
plt.tight_layout()
plt.savefig(plots_folder/'End effector (after optimization)', bbox_inches='tight')
#plt.show()

animate_robot_matplotlib(
    robot = robot_opt,
    t_list = t_target,
    q_list = q_ts,
    target = target,
    interval = 1e-2, 
    slider = True,
    animation = False,
    show = True
)