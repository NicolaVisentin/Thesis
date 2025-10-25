# =====================================================
# Setup
# =====================================================

# Choose device (cpu or gpu)
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Imports
import jax
import jax.numpy as jnp
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import time

from evosax.algorithms.distribution_based.cma_es import CMA_ES
from utilis import *

from soromox.systems.pendulum import Pendulum

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


# =====================================================
# User settings
# =====================================================

use_softplus = False
use_lax_scan = False


# =====================================================
# Pendulum before optimization
# =====================================================
print('--- INITIAL SIMULATION ---')

# Instantiate robot
num_links = 2
params = {
    "m": jnp.array([10.0, 6.0]),
    "I": jnp.array([3.0, 2.0]),
    "L": jnp.array([2.0, 1.0]),
    "Lc": jnp.array([1.0, 0.5]),
    "g": jnp.array([0.0, -9.81]),
    "D": jnp.diag(jnp.array([50, 20]))
}
robot = Pendulum(params)

# Set target (for optimization)
target = jnp.array([0.0, -2.0])

# Set simulation parameters
q0 = jnp.array([-jnp.pi * 7/11, jnp.pi * 7/13]) # initial configuration
qd0 = jnp.zeros_like(q0)                        # initial velocities
u = jnp.zeros_like(q0)                          # torques (actuation)

dt = 1e-3
t0 = 0.0
t1 = 15.0
save_dt = 100*dt # time resolution for saving results

# Simulation
print('Simulating robot...')
start = time.perf_counter()
ts_out, q_ts, _ = robot.resolve_upon_time(
    q0=q0,
    qd0=qd0,
    u=u,
    t0=t0,
    t1=t1,
    dt=dt,
    save_dt=save_dt,
)
end = time.perf_counter()
print(f'Elapsed time (simulation): {end-start} s')

# Extract end effector coordinates chi = [th, x, y]
chi_ee_ts = jax.vmap(robot.forward_kinematics_tips,)(q_ts)[:, -1, :] # shape (n_steps, 3)

# Error
dist = np.linalg.norm((chi_ee_ts[-1,1:] - target) ** 2)
print(f'MSE before optimization: {dist:.3f} m')

# Plot results
plt.figure()
plt.plot(ts_out, chi_ee_ts[:, 1], label="end-effector x", color='b')
plt.plot(ts_out, chi_ee_ts[:, 2], label="end-effector y", color='r')
plt.axhline(target[0], label='target x', linestyle='--', color='b')
plt.axhline(target[1], label='target y', linestyle='--', color='r')
plt.xlabel("t [s]")
plt.ylabel("pos [m]")
plt.title(f'End-effector position (before optimization)\nL={params["L"]} m')
plt.legend()
plt.grid(True)
plt.box(True)
plt.tight_layout()
plt.savefig(plots_folder/'end effector (beofre optimization)', bbox_inches='tight')
#plt.show()

plt.figure()
plt.scatter(chi_ee_ts[:, 1], chi_ee_ts[:, 2], c=ts_out, cmap="viridis")
plt.scatter(target[0], target[1], color='r', label='target')
plt.axis("equal")
plt.grid(True)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()
plt.colorbar(label="t [s]")
plt.title(f'End-effector position (beofre optimization)\nL={params["L"]} m')
plt.tight_layout()
plt.show()


# =====================================================
# Define the problem (function to optimize)
# =====================================================
    
# Set cost function
@jax.jit
def Loss(params):
    # update robot
    if use_softplus:
        params = jax.nn.softplus(params)
    robot_updated = robot.update_params({"L": params})
    # simulation
    _, q_ts, _ = robot_updated.resolve_upon_time(
        q0=q0,
        qd0=qd0,
        u=u,
        t0=t0,
        t1=t1,
        dt=dt,
        save_dt=save_dt,
        max_steps=int(1e5)
    )
    ee_final = robot_updated.forward_kinematics_tips(q_ts[-1,:])[-1, 1:]
    # loss
    J = (ee_final-target).T @ (ee_final-target)
    return J
     
# Set research bounds
min_vals = jnp.array([1e-1, 1e-1]) # min L1 and L2 in [m]
max_vals = jnp.array([4e0, 4e0])   # max L1 and L2 in [m]

if use_softplus:
    min_vals = InverseSoftplus(min_vals)
    max_vals = InverseSoftplus(max_vals)

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
num_generations = 12  # number of iterations
population_size = 64 # dimension of the population

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
        metrics["means"] = jax.nn.softplus(metrics["means"])
        metrics["populations"] = jax.nn.softplus(metrics["populations"])

    return metrics

# Instantiate the search strategy and initialize a solution and state of the optimizer
key, subkey = jax.random.split(key)
#solution0 = problem.sample(subkey) # initial point (mean) for the research
solution0 = robot.L
if use_softplus:
    solution0 = InverseSoftplus(solution0)

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
        L_curr = metrics["means"]
        best_fitness_curr = metrics["best_fitness"]
        tqdm.write(
            f"Iter {ii:02d} | "
            f"L={L_curr} | "
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

L_opt = means[-1]
print(f"Optimal L: {L_opt} m")


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
#plt.show()

# Convert bounds if softplus is used
if use_softplus:
    low_bound_plot = jax.nn.softplus(problem.lower_research_bound)
    up_bound_plot = jax.nn.softplus(problem.upper_research_bound)
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

# Update robot
robot_opt = robot.update_params({"L": L_opt})

# Simulation
print('Simulating robot...')
ts_out, q_ts, _ = robot_opt.resolve_upon_time(
    q0=q0,
    qd0=qd0,
    u=u,
    t0=t0,
    t1=t1,
    dt=dt,
    save_dt=save_dt,
)

# Extract end effector coordinates chi = [th, x, y]
chi_ee_ts = jax.vmap(robot_opt.forward_kinematics_tips,)(q_ts)[:, -1, :] # shape (n_steps, 3)

# Error
dist = np.linalg.norm((chi_ee_ts[-1,1:] - target) ** 2)
print(f'MSE after optimization: {dist:.3f} m')

# Plot results
plt.figure()
plt.plot(ts_out, chi_ee_ts[:, 1], label="end-effector x", color='b')
plt.plot(ts_out, chi_ee_ts[:, 2], label="end-effector y", color='r')
plt.axhline(target[0], label='target x', linestyle='--', color='b')
plt.axhline(target[1], label='target y', linestyle='--', color='r')
plt.xlabel("t [s]")
plt.ylabel("pos [m]")
plt.title(f'End-effector position (after optimization)\nL={L_opt} m')
plt.legend()
plt.grid(True)
plt.box(True)
plt.tight_layout()
plt.savefig(plots_folder/'end effector (after optimization)', bbox_inches='tight')
#plt.show()

plt.figure()
plt.scatter(chi_ee_ts[:, 1], chi_ee_ts[:, 2], c=ts_out, cmap="viridis")
plt.scatter(target[0], target[1], color='r', label='target')
plt.axis("equal")
plt.grid(True)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()
plt.colorbar(label="t [s]")
plt.title(f'End-effector position (after optimization)\nL={L_opt} m')
plt.tight_layout()
plt.show()