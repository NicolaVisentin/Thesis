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

from evosax.algorithms.distribution_based.cma_es import CMA_ES
from utilis import *

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

# Folder for plots and videos
curr_folder = Path(__file__).parent
plots_folder = curr_folder/'plots and videos'/Path(__file__).stem
plots_folder.mkdir(parents=True, exist_ok=True)

# # Folder for saving data
# data_folder = curr_folder/'saved data'/Path(__file__).stem
# data_folder.mkdir(parents=True, exist_ok=True)


# =====================================================
# Define the problem (function to optimize)
# =====================================================
    
# Set cost function
@jax.jit
def cost(x):
    J = jnp.sin(0.05*x[0]) * jnp.cos(0.05*x[1]) + jnp.sin(0.02*x[0]) * jnp.cos(0.02*x[1])
    return J

# Set research bounds
min_vals = jnp.array([-70, -80])
max_vals = jnp.array([70, 60])

# Instantiate the problem
problem = MyOptimizationProblem(
    cost_fn=cost,
    lower_research_bound=min_vals,
    upper_research_bound=max_vals,
)


# =====================================================
# Define the optimization strategy (CMA-ES)
# =====================================================

# Set parameters
num_generations = 24  # number of iterations
population_size = 128 # dimension of the population

# Set info to save during the iterations
def metrics_fn(key, population, fitness, state, params):
    idx = jnp.argmin(fitness)
    metrics = {
        "fitness": fitness,           # fitness of all individuals
        "best_fitness": fitness[idx], # best fitness among all individuals
        "mean": state.mean,           # mean of the distribution (solution)
        "populations": population     # population
    }
    return metrics

# Instantiate the search strategy and initialize a solution and state of the optimizer
key, subkey = jax.random.split(key)
#solution0 = problem.sample(subkey) # initial point (mean) for the research
solution0 = jnp.array([20.0, -20.0])

cma = CMA_ES(
    population_size=population_size, 
    solution=solution0,
    metrics_fn=metrics_fn,
)
params = cma.default_params # parameters for CMA-ES (e.g. learning rate for the mean, learning rate for the covariance...)

key, subkey = jax.random.split(key)
state0 = cma.init(subkey, solution0, params) # e.g. current best solution, current best fitness, current generation counter...


# =====================================================
# Run the optimization
# =====================================================

# Single iteration step (for jax.lax.scan)
def step(carry, key):
    # extract CMA-ES params and current CMA-ES state from carry
    state, params = carry
    key_ask, key_tell = jax.random.split(key, 2)

    # sample a population basing on the current state
    population, state = cma.ask(key_ask, state, params)
    population = jnp.clip(population, min_vals, max_vals) # clip population within the bounds
    
    # evaluate the new population (return fitness) and update the counter
    fitness = problem.eval(population)

    # update the state basing on the evaluation of the new population
    state, metrics = cma.tell(key_tell, population, fitness, state, params)

    carry = (state, params)
    out = metrics
    return carry, out

# Execute the iterations
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num_generations)
carry0 = (state0, params)
_, metrics = jax.lax.scan(
    step,
    carry0,
    keys,
)

# Extract results
means = metrics["mean"]                  # mean of the population. Shape (num_generations, num_dims)
fitness_means = problem.eval(means)      # fitness of the mean. Shape (num_generations,)
populations = metrics["populations"]     # population. Shape (num_generations, population_size, num_dims)
fitness_populations = metrics["fitness"] # fitness of all individuals. Shape (num_generations, population_size)
best_fitness = metrics["best_fitness"]   # fitness of the best individual. Shape (num_generations,)


# =====================================================
# Visualize results
# =====================================================

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

# Plot 3D
x = np.linspace(problem.lower_research_bound[0], problem.upper_research_bound[0], 101)
y = np.linspace(problem.lower_research_bound[1], problem.upper_research_bound[1], 101)
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
ax.set_xlim([problem.lower_research_bound[0], problem.upper_research_bound[0]])
ax.set_ylim([problem.lower_research_bound[1], problem.upper_research_bound[1]])
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
    grid_res=101,
    savepath=plots_folder/'animation.gif',
    show=True,
)
