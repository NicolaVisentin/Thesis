# =====================================================
# Setup
# =====================================================

# Choose device (cpu or gpu)
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Add path to libraries
import sys
from pathlib import Path

# Imports
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from evosax.problems import BBOBProblem
from evosax.algorithms.distribution_based.cma_es import CMA_ES

# Set random seed
seed = 0
key = jax.random.key(seed)


# =====================================================
# Define the problem (function to optimize)
# =====================================================

# Number of parameters
num_dims = 2

# Instantiate BBOB "sphere" problem
problem = BBOBProblem(
    fn_name='schaffers_f7',      # choose shape of the loss
    num_dims=num_dims,
    x_opt=jnp.array([0.0, 0.0]), # choose position of the minimum
    f_opt=0,                     # choose value of the minimum
    sample_rotations=False,
    seed=seed,
)

# Visualize the loss function
key, subkey = jax.random.split(key)
fig = plt.figure(figsize=(10, 5))

ax3D = fig.add_subplot(121, projection='3d')
problem.visualize_3d(subkey, ax=ax3D, logscale=True) # use logscale for rosenbrock, rosenbrock_rotated, different_powers, schaffers_f7, schaffers_f7_ill_cond, schwefel, katsuura

ax2D = fig.add_subplot(122)
problem.visualize_2d(subkey, ax=ax2D, logscale=True) # use logscale for rosenbrock, rosenbrock_rotated, different_powers, schaffers_f7, schaffers_f7_ill_cond, schwefel, katsuura

plt.show()
exit()

# =====================================================
# Define the optimization strategy (CMA-ES)
# =====================================================

num_generations = 64           # number of iterations
population_size = 128          # dimension of the population
solution0 = jnp.array([4.0, -3.0]) # initial point (mean) for the research
def metrics_fn(key, population, fitness, state, params):
    """
    Gives the best solution and corresponding fitness
    """
    idx = jnp.argmin(fitness)
    metrics = {"best_fitness": fitness[idx], "mean": state.mean}
    return metrics

# Instantiate the search strategy
es = CMA_ES(
    population_size=population_size, 
    solution=solution0,
    metrics_fn=metrics_fn,
)

params = es.default_params # parameters for CMA-ES (e.g. learning rate for the mean, learning rate for the covariance...)

# Initialize the state of the research strategy
key, subkey = jax.random.split(key)
state0 = es.init(subkey, solution0, params) # e.g. current best solution, current best fitness, current generation counter...


# =====================================================
# Run the optimization
# =====================================================
key, subkey = jax.random.split(key)
problem_state0 = problem.init(subkey)   # it's simply the counter initialized as 0

# Single iteration step (for jax.lax.scan)
def step(carry, key):
    # extract CMA-ES params, current CMA-ES state and current problem state (counter) from carry
    state, params, problem_state = carry
    key_ask, key_eval, key_tell = jax.random.split(key, 3)

    # sample a population basing on the current state
    population, state = es.ask(key_ask, state, params)
    population = jnp.clip(population, -5, 5)
    
    # evaluate the new population (return fitness) and update the counter
    fitness, problem_state, _ = problem.eval(key_eval, population, problem_state)

    # update the state basing on the evaluation of the new population
    state, metrics = es.tell(key_tell, population, fitness, state, params)

    return (state, params, problem_state), metrics

# Execute the iterations
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num_generations)
_, metrics = jax.lax.scan(
    step,
    (state0, params, problem_state0),
    keys,
)

# Extract results
means = metrics["mean"]             # shape (num_generations, num_dims)
fitnesses = metrics["best_fitness"] # shape (num_generations,)

# Plot best fitness for each generation vs generations
plt.figure()
plt.plot(fitnesses)
plt.grid(True)
plt.xlabel('generation')
plt.ylabel('best fitness')
plt.title('Fitness curve')
plt.show()

# Animation
fig, ax = plt.subplots(figsize=(6, 5))
key, subkey = jax.random.split(key)
problem.visualize_2d(subkey, ax=ax)
scatter = ax.scatter([], [], color="red", marker="x", linewidth=2)
title = ax.set_title("Generation: 0")

def init():
    scatter.set_offsets(jnp.empty((0, 2)))
    return scatter, title

def update(frame):
    scatter.set_offsets(means[frame].reshape(1, -1)) # update position of the mean
    title.set_text(f"Generation: {frame}")           # update title
    return scatter, title

anim = FuncAnimation(fig, update, frames=len(means), init_func=init, blit=True)
plt.close()

# Create a writer
path = "anim.gif"
anim.save(path, writer=PillowWriter())