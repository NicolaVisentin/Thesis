# Imports
import jax
from jax import numpy as jnp
from jax import Array


# =====================================================
# 
# =====================================================

# Control integrated absolute power
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
