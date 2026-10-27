import jax
from jax import numpy as jnp, Array
import numpy as np
from typing import Callable, Tuple
from diffrax import Euler, Tsit5, ConstantStepSize, LinearInterpolation
from functools import partial

from soromox.systems.my_systems import PlanarPCS_simple
from soromox.systems.system_state import SystemState


class MultiPcsReservoir:
    def __init__(self, robots_system, map_direct: Callable, map_inverse: Callable, controller: Callable):
        """
        Initialize the reservoir.
        
        Args
        ----
        robots_system : MultiPcsSystem
            Soft robots system.
        map_direct : Callable
            Mapping between "virtual" space and physical soft robots space. Must have signature Q, Qd = map_direct(y, yd).
        map_inverse : Callable
            Mapping between physical soft robots space and "virtual" space. Must have signature y, yd = map_inverse(Q, Qd).
        controller : Callable
            (Feedback) control law on the PCS space. Must have signature Tau = controller(Z, u), where Z = [Q, Qd] is the 
            pcs system state.
        """
        self.robots = robots_system
        self.map_direct = map_direct
        self.map_inverse = map_inverse
        self.controller = controller
        self.hid_dim = self.robots.dim
        
    @partial(jax.jit, static_argnums=(0,))    
    def __call__(
        self, u: Array, time_u: Array, saveat: Array, dt: float=1e-4,
        key_noise=None, Sigma_Tau=None, Sigma_Q=None, Sigma_Qd=None, dt_noise=0.05
    ) -> Tuple[Array, Array, Array, Array, Array]:
        """
        Forward pass of the reservoir. Given a certain input in time, performs simulation of
        the reservoir and returns final activations.

        Args
        ----
        u : Array
            Reservoir's input time sequence. Shape (n_steps_input, dim_input)
        time_u : Array
            Time vector associated with u. Shape (n_steps_input,)
        saveat : Array
            Time vector for saving the solution of the simulation (use same t0 and t1 of time_u).
        dt : float
            Constant time step for the simulation (default: 0.0001 s).
        key_noise : Array | None
            JAX PRNG key for noise/disturbance generation. If None, no noise/disturbance is applied.
        Sigma_Tau, Sigma_Q, Sigma_Qd : Array | None
            Covariance matrices for disturbance on Tau and for noise on Q and Qd. Noise and disturbance are modeled
            as zero-mean Gaussian noise with given DIAGONAL covariance matrix. None deactivates the specific 
            noise/disturbance. Shape (3*n_pcs*n_robots, 3*n_pcs*n_robots).
        dt_noise : float
            Timestep for the noise (default: 0.05 s)

        Returns
        -------
        time_ts : Array
            Time vector of the simulation. Shape (n_steps,)
        state_reservoir_ts : Array
            Reservoir's state time history r(t) = [y(t), yd(t)], accounts for noise. Shape (n_steps, 2*n_y)
        state_pcs_ts : Array
            Soft robots system's state time history Z(t) = [Q(t), Qd(t)], accounts for noise. Shape (n_steps, 2*n_robots*3*n_pcs)
        actuation_ts : Array
            Soft robots system's control signal time history, accounts for disturbance. Shape (n_steps, n_robots*3*n_pcs)
        last_states : Array
            Final reservoir's state: last element of y_ts. Shape (n_y,)
        """
        # Simulation parameters
        t0 = time_u[0]
        t1 = time_u[-1]
        dt = dt #1e-4
        saveat = saveat
        solver = Euler() #Tsit5()
        step_size = ConstantStepSize()
        max_steps = int(1e8)

        # Noise setup
        if key_noise is None:
            key_Tau = key_Q = key_Qd = None
            Sigma_Tau = Sigma_Q = Sigma_Qd = None
        else:
            key_Tau, key_Q, key_Qd = jax.random.split(key_noise, 3)
            std_Tau = None if Sigma_Tau is None else jnp.sqrt(jnp.diag(Sigma_Tau))
            std_Q   = None if Sigma_Q   is None else jnp.sqrt(jnp.diag(Sigma_Q))
            std_Qd  = None if Sigma_Qd  is None else jnp.sqrt(jnp.diag(Sigma_Qd))

        def noise(key, t, stddev):
            """Zero-mean Gaussian noise N(0, Sigma), piecewise constant."""
            idx = jnp.floor((t - t0) / dt_noise + 1e-6).astype(jnp.uint32)
            subkey = jax.random.fold_in(key, idx) # allows for reproducing same noise signal after the simulation
            return stddev * jax.random.normal(subkey, stddev.shape)
        
        # Controller definition
        u_interp = LinearInterpolation(
            ts=time_u,
            ys=u
        )
        def tau_law(system_state: SystemState, controller: Callable, u_interp) -> Tuple:
            """Implements user-defined control law Tau(t, Q, Qd). Also includes noise/disturbance, if the case."""
            t = system_state.t
            u_t = u_interp.evaluate(t)
            Z = system_state.y

            if Sigma_Q is not None or Sigma_Qd is not None:
                Q, Qd = jnp.split(Z, 2)
                if Sigma_Q is not None:
                    Q = Q + noise(key_Q, t, std_Q)
                if Sigma_Qd is not None:
                    Qd = Qd + noise(key_Qd, t, std_Qd)
                Z = jnp.concatenate([Q, Qd])

            Tau = controller(Z, u_t)
            if Sigma_Tau is not None:
                Tau = Tau + noise(key_Tau, t, std_Tau)
            return Tau, None
        
        tau_diffrax = jax.jit(partial(tau_law, controller=self.controller, u_interp=u_interp)) # signature u(SystemState) -> (control_u, control_state_dot) required by soromox
        
        # Simulation
        Q0, Qd0 = self.map_direct(jnp.zeros(self.hid_dim), jnp.zeros(self.hid_dim)) # init cond for the reservoir is [0, 0]
        initial_state_pcs = SystemState(t=t0, y=jnp.concatenate([Q0, Qd0]))

        sim_out_pcs = self.robots.rollout_closed_loop_to(
            initial_state = initial_state_pcs,
            controller = tau_diffrax,
            t1 = t1, 
            solver_dt = dt, 
            save_ts = saveat,
            solver = solver,
            stepsize_controller = step_size,
            max_steps = max_steps
        )

        # Convert results and return outputs
        time_ts = sim_out_pcs.t
        Q_pcs, Qd_pcs = jnp.split(sim_out_pcs.y, 2, axis=1)
        if Sigma_Q is not None:
            Q_pcs = Q_pcs + jax.vmap(lambda t: noise(key_Q, t, std_Q))(time_ts)
        if Sigma_Qd is not None:
            Qd_pcs = Qd_pcs + jax.vmap(lambda t: noise(key_Qd, t, std_Qd))(time_ts)
        y, yd = self.map_inverse(Q_pcs, Qd_pcs)

        # Return outputs
        state_pcs_ts = jnp.concatenate((Q_pcs, Qd_pcs), axis=1) # also accounts for noise (measured state)
        actuation_ts = sim_out_pcs.u # also accounts for disturbance
        state_reservoir_ts = jnp.concatenate((y, yd), axis=1) # also accounts for noise/disturbance
        last_states = y[-1] # also accounts for noise/disturbance
        return time_ts, state_reservoir_ts, state_pcs_ts, actuation_ts, last_states