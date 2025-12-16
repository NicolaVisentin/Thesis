# Imports
import jax
from jax import numpy as jnp
from jax import Array
from functools import partial


# =====================================================
# 
# =====================================================

# Mean kinetic energy ratio on a dataset
def mean_Ek_ratio(robot, dataset, map) -> float:
    """
    Args
    ----
    robot
        Robot instance. Must have method robot.kinetic_energy(q, qd)
    dataset
        Dictionary with keys "y" and "yd".
    map
        Mapping from r=[y,yd] to z=[q,qd]. In general, is defined as a function z = map(r)

    Returns
    -------
    Ek_ratio_metric
        Mean on the dataset of the ratio Ek_pcs / Ek_ron
    """
    y_batch, yd_batch = dataset["y"], dataset["yd"] # shape (batch_size, n_ron) each
    r_batch = jnp.concatenate([y_batch, yd_batch], axis=1) # shape (batch_size, 2*n_ron)
    z_batch = jax.vmap(map)(r_batch) # shape (batch_size, 2*3*n_pcs)
    q_batch, qd_batch = jnp.split(z_batch, 2, axis=1) # shape (batch_size, 3*n_pcs) each
    Ek_pcs = jax.vmap(robot.kinetic_energy, in_axes=(0,0))(q_batch, qd_batch) # shape (batch_size,)
    Ek_ron = jnp.sum(yd_batch**2, axis=1) # shape (batch_size,)
    Ek_ratio_metric = jnp.mean(Ek_ron / Ek_pcs) # scalar

    return Ek_ratio_metric

# Integrated absolute power
def compute_integrated_power(t_ts: Array, tau_ts: Array, qd_ts: Array):
    """
    Computes control effort as the integrated mean absolut power during a simulation.
    Power is P(t) = tau(t)^T * qd(t).

    Args
    ----
    t_ts : Array
        Time vector. Shape (time_steps,)
    tau_ts : Array
        Actuation time history. Shape (time_steps, 3*n_pcs)
    qd_ts : Array
        Strains velocity time history. Shape (time_steps, 3*n_pcs)

    Returns
    -------
    P_metric : float
        Integrated mean absolut value of the power.
    """
    # Instantaneous absolute power
    abs_power_ts = jnp.abs(jnp.sum(tau_ts * qd_ts, axis=-1)) # shape (time_steps,)

    # Integrated absolute power
    P_int = jnp.trapz(abs_power_ts, t_ts) # scalar

    # Time-normalized metric
    T = t_ts[-1] - t_ts[0]
    P_metric = P_int / T

    return P_metric
