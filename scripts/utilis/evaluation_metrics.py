# Imports
import jax
from jax import numpy as jnp
from jax import Array
from jax.scipy.integrate import trapezoid


# Root mean square error on a simulation
def compute_simulation_rmse(that_ts: Array, yhat_ts: Array, ty_ts: Array, y_ts: Array):
    """
    Computes accuracy as the root mean square error during a simulation. RMS( y(t), y_hat(t) )

    Args
    ----
    that_ts, yhat_ts : Array
        Approximation time and time history. Shape (time_steps, 3*n_pcs)
    ty_ts, y_ts : Array
        Reference time and time history. Shape (time_steps, 3*n_pcs)

    Returns
    -------
    rmse_metric : float
        Root mean square error.
    """
    # Interpolate to handle different dimensions
    interp_fn = lambda y_comp: jnp.interp(ty_ts, that_ts, y_comp)
    yhat_ts_interp = jax.vmap(interp_fn, in_axes=1, out_axes=1)(yhat_ts)

    # Compute RMSE
    error_ts = jnp.sum((y_ts - yhat_ts_interp)**2, axis=1) # shape (time_steps,)
    rmse_metric = jnp.sqrt(jnp.mean(error_ts)) # scalar

    return rmse_metric


# Root mean square power on a simulation
def compute_simulation_power(tau_ts: Array, qd_ts: Array):
    """
    Computes control effort as the root mean square power during a simulation.
    Power is P(t) = tau(t)^T * qd(t).

    Args
    ----
    tau_ts : Array
        Actuation time history. Shape (time_steps, 3*n_pcs)
    qd_ts : Array
        Strains velocity time history. Shape (time_steps, 3*n_pcs)

    Returns
    -------
    P_metric : float
        Root mean square value of the power.
    """
    power_ts = jnp.sum(tau_ts * qd_ts, axis=1) # shape (time_steps,)
    P_metric = jnp.sqrt(jnp.mean(power_ts**2)) # scalar

    return P_metric


# Kinetic energy ratio on a simulation
def compute_simulation_Ek_ratio(robot, tq_ts, q_ts, qd_ts, ty_ts, yd_ts):
    """
    Computes mapping effort as the Ek ratio during a simulation.
    Ek ratio is Ek_ron / Ek_pcs.

    Args
    ----
    robot
        Robot instance. Must have robot.kinetic_energy(q,qd) method.
    tq_ts : Array
        Time vector assicuated with q.
    q_ts : Array
        Strains time history. Shape (time_steps, 3*n_pcs)
    qd_ts : Array
        Strains velocity time history. Shape (time_steps, 3*n_pcs)
    ty_ts : Array
        Time vector assicuated with y.
    yd_ts : Array
        Reference velocity time history. Shape (time_steps, 3*n_pcs)

    Returns
    -------
    Ek_metric : float
        Ek_ratio.
    """
    E_k_pcs_ts = jax.vmap(robot.kinetic_energy, in_axes=(0,0))(q_ts, qd_ts) # shape (time_steps_pcs,)
    Ek_ron_ts = jnp.sum(yd_ts**2, axis=1) # shape (time_steps_ron,)
    Ek_metric = trapezoid(Ek_ron_ts, ty_ts) / trapezoid(E_k_pcs_ts, tq_ts) # scalar / scalar = scalar

    return Ek_metric


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
        Mapping from r=[y,yd] to z=[q,qd]. In general, is defined as a function z = map(r).

    Returns
    -------
    Ek_ratio_metric
        Mean on the dataset of the ratio Ek_ron / Ek_pcs.
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
    P_int = trapezoid(abs_power_ts, t_ts) # scalar

    # Time-normalized metric
    T = t_ts[-1] - t_ts[0]
    P_metric = P_int / T

    return P_metric