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

from evosax.types import Population, Solution, Fitness, Metrics


# =====================================================
# Sandbox
# =====================================================

from evosax.problems.problem import Problem, State
from evosax.problems.bbob.bbob import BBOBProblem

my_BBOBProblem = BBOBProblem(
    fn_name='sphere',
    num_dims=3,
    x_opt=jnp.array([1.0, 2, 3]),
    #f_opt=None,
    sample_rotations=True,
)

print(my_BBOBProblem.x_range)