import numpy as np
import jax
import jax.numpy as jnp
from jax import Array

from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# =====================================================
# Evolutionary algorithms
# =====================================================

# Class for optimization problems to run with evosax package
class MyOptimizationProblem:
    """
    Class for defining custom optimization problem to use with evosax.

    Parameters
    ----------
    cost_fn : Callable
        Cost function J = f(x), with x n-dimensional array.
    lower_research_bound : Array
        Lower bound for each research parameter (x1, x2, ..., xn). Shape (n,)
    upper_research_bound : Array
        Upper bound for each research parameter (x1, x2, ..., xn). Shape (n,)

    Attributes
    ----------
    cost_fn : Callable
        Cost function J = f(x), with x n-dimensional array.
    lower_research_bound : Array
        Lower bound for each research parameter (x1, x2, ..., xn). Shape (n,)
    upper_research_bound : Array
        Upper bound for each research parameter (x1, x2, ..., xn). Shape (n,)
    dim : int
        Dimension of the problem (n).
    """
    def __init__(
            self, 
            cost_fn: Callable,
            lower_research_bound: Array, 
            upper_research_bound: Array,
    ):
        if len(lower_research_bound) != len(upper_research_bound):
            raise ValueError('lower_research_bound and upper_research_bound must have same length.')
        self.lower_research_bound = lower_research_bound
        self.upper_research_bound = upper_research_bound
        self.dim = len(self.lower_research_bound)
        self.cost_fn = jax.vmap(cost_fn, in_axes=0) # vmap cost function to accept batch of inputs

    @partial(jax.jit, static_argnames=("self",))
    def eval(self, points_batch: Array) -> Array:
        """
        Evaluates J over a batch of points x.

        Args
        ----
        points_batch : Array
            Batch of points. Shape (batch_size, n)
        
        Returns
        -------
        fitness_batch : Array
            Value of J for the given points. Shape (batch_size,)
        """
        if len(points_batch.shape) == 1:
            points_batch = points_batch[None,:]
        fitness_batch = self.cost_fn(points_batch)
        return fitness_batch

    @partial(jax.jit, static_argnames=("self",))
    def sample(self, key) -> Array:
        """
        Samples a random point x in the research space.
        
        Args
        ----
        key : Array
            Random key.
        
        Returns
        -------
        point : Array
            Random point within the research bounds. Shape (n,)
        """
        point = jax.random.uniform(
            key=key, 
            shape=(self.dim,), 
            minval=self.lower_research_bound, 
            maxval=self.upper_research_bound
        )
        return point
    

# Function to animate the progress of the optimization (! ONLY FOR 2D CASES !)
def animate_evolution(
        problem,
        means : Array,
        fitness_means: Array = None,
        populations: Array = None,
        fitness_populations: Array = None,
        duration: int = 10,
        grid_res: int | dict = 0,
        savepath = None,
        show: bool = True
):    
    """
    Function to animate evolution of the evolutionary algorithm.

    Args
    ----
    problem
        Optimization problem defined by the user. It's a Problem object from evosax.
    means : Array
        Solution (mean of the population) for each iteration. Shape (num_generations, num_dims)
    fitness_means
        Fitness of the solution (mean of the population) for each iteration. If not provided, it is
        computed by calling problem.eval on means. Shape (num_generations,)
    populations : Array, optional
        Population for each iteration. Shape (num_generations, population_size, num_dims)
    fitness_populations: Array, optional
        Fitness of each individual of the population for each iteration. If not provided, it is computed 
        by calling problem.eval on populations. Shape (num_generations, population_size)
    duration : int, optional
        Duration of the animation in seconds (default: 10). 
    grid_res : int | dict, optional
        Resolution of the grid for plotting the surface (default: 0). If zero, does not plot the surface.
        If dict, must be a dictionary containing the data to plot the surface, i.e. grid_res["X"], 
        grid_res["Y"], grid_res["Z"].
    savepath : optional
        Path to save the animation, in form /my_path/'name_animation.gif'. Animation will be saved
        in /my_path as 'name_amination.gif'. If None, animation will not be saved.
    show : bool, optional
        True to print the animation (default: True).
    """
    num_generations = len(means)

    # compute fitness of the mean at each iteration, if not provided
    if fitness_means is None:
        fitness_means = problem.eval(means)
    
    # compute fitness of all individuals at each iteration, if not provided
    if populations is not None:
        pop = populations
        if fitness_populations is None:
            eval_pop = jax.vmap(problem.eval, in_axes=0)
            fitness_populations = eval_pop(pop)
    else:
        pop = np.full((num_generations,1,2), np.nan)
        fitness_populations = np.full((num_generations,1), np.nan)
    
    # compute fitness surface in the research space
    if grid_res == 0:
        if populations is not None:
            z_min = np.min([np.min(fitness_means), np.min(fitness_populations)])
            z_max = np.max([np.max(fitness_means), np.max(fitness_populations)])
        else:
            z_min = np.min(fitness_means)
            z_max = np.max(fitness_means)
    else:
        if isinstance(grid_res, int):
            x = np.linspace(problem.lower_research_bound[0], problem.upper_research_bound[0], grid_res)
            y = np.linspace(problem.lower_research_bound[1], problem.upper_research_bound[1], grid_res)
            X, Y = np.meshgrid(x, y)
            points = np.stack([X.flatten(), Y.flatten()], axis=1)
            Z = problem.eval(points).reshape(X.shape)
        else:
            X = grid_res["X"]
            Y = grid_res["Y"]
            Z = grid_res["Z"]

    # create figure
    fig = plt.figure(figsize=(6, 10))

    # add 3D plot
    ax3D = fig.add_subplot(211, projection='3d')
    if grid_res != 0:
        ax3D.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, zorder=1)
    else:
        ax3D.set_zlim([z_min, z_max])
    sol3D = ax3D.scatter([], [], [], color='r', s=20, zorder=3, label='mean')
    pop3D = ax3D.scatter([], [], [], color='orange', s=20, zorder=2)
    if populations is not None:
        pop3D.set_label('population')
    ax3D.set_xlabel('x')
    ax3D.set_ylabel('y')
    ax3D.set_zlabel('fitness')
    title = ax3D.set_title("Generation: 0")
    ax3D.set_xlim([problem.lower_research_bound[0], problem.upper_research_bound[0]])
    ax3D.set_ylim([problem.lower_research_bound[1], problem.upper_research_bound[1]])
    ax3D.legend()

    # add 2D plot
    ax2D = fig.add_subplot(212, )
    if grid_res != 0:
        plt2d = ax2D.contourf(X, Y, Z, cmap='viridis', zorder=1)
        cbar = fig.colorbar(plt2d, ax=ax2D, shrink=1.0)
        cbar.set_label('fitness')
    sol2D = ax2D.scatter([], [], color='r', s=20, zorder=3, label='mean')
    pop2D = ax2D.scatter([], [], color='orange', s=20, zorder=2)
    if populations is not None:
        pop2D.set_label('population')
    ax2D.set_xlabel('x')
    ax2D.set_ylabel('y')
    ax2D.grid(True)
    ax2D.set_xlim([problem.lower_research_bound[0], problem.upper_research_bound[0]])
    ax2D.set_ylim([problem.lower_research_bound[1], problem.upper_research_bound[1]])
    ax2D.legend()

    def init():
        sol3D._offsets3d = ([], [], [])
        pop3D._offsets3d = ([], [], [])
        sol2D.set_offsets(np.empty((0, 2)))
        pop2D.set_offsets(np.empty((0, 2)))
        title.set_text("Generation: 0")
        return sol3D, pop3D, sol2D, pop2D, title

    def update(frame):
        x_mean = means[frame, 0]
        y_mean = means[frame, 1]
        z_mean = fitness_means[frame]
        x_pop = pop[frame, :, 0]
        y_pop = pop[frame, :, 1]
        z_pop = fitness_populations[frame]

        title.set_text(f"Generation: {frame}") 
        sol3D._offsets3d = ([x_mean], [y_mean], [z_mean])
        pop3D._offsets3d = (x_pop, y_pop, z_pop)
        sol2D.set_offsets(np.column_stack((x_mean, y_mean)))
        pop2D.set_offsets(np.column_stack((x_pop, y_pop)))

        return sol3D, pop3D, sol2D, pop2D, title
    
    plt.tight_layout()

    # generate animation
    n_frames = len(means)
    fps = n_frames / duration
    ani = FuncAnimation(
        fig, update, init_func=init,
        frames=n_frames, interval=1000/fps, blit=False, repeat_delay=1500,
    )

    # save animation in specified path
    if savepath is not None:
        ani.save(savepath, writer=PillowWriter(fps=fps))

    # show animation
    if show:
        plt.show()
    else:
        plt.close()


# =====================================================
# Others
# =====================================================

# Inverse softplus function
def InverseSoftplus(x):
    """
    Inverse softplus function.
    """
    return jnp.log(jnp.exp(x)-1)


# Split dataset in train/validation/test
def split_dataset(
        key, 
        dataset : dict,
        train_ratio : float = 0.7, 
        test_ratio : float = 0.2,
    ) -> tuple[dict, dict, dict]:
    """
    Split datapoints and labels into train, validation, and test sets.

    Args
    ----
    key 
        jax.random.key.
    dataset : dict
        Dataset provided as a dictionary with all elements (m datapoints and m labels). Each key has shape (m, ...),
        where m in the size of the dataset.
    train_ratio : float
        Train set ratio between 0 and 1 (default: 0.7). train_ratio + val_ratio + test_ratio = 1.
    test_ratio : float
        Test set ratio between 0 and 1 (default: 0.2). train_ratio + val_ratio + test_ratio = 1.
    
    Returns
    -------
    train_set, val_set, test_set : dict
        Dictionaries for train set, validation set and test set, with the same keys of the original dataset.
    """
    # Dataset size
    first_key = next(iter(dataset.keys()))
    m = dataset[first_key].shape[0]

    # shuffle indices
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, m)

    # compute split indices
    train_end = int(train_ratio * m)
    test_end = m - int(test_ratio * m)

    # create splits for each item in dataset
    train_set = {k: v[perm[:train_end]] for k, v in dataset.items()}
    val_set   = {k: v[perm[train_end:test_end]] for k, v in dataset.items()}
    test_set  = {k: v[perm[test_end:]] for k, v in dataset.items()}

    return train_set, val_set, test_set