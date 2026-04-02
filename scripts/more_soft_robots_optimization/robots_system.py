import jax
from jax import numpy as jnp, Array
import equinox as eqx
from soromox.systems.my_systems import PlanarPCS_simple
from soromox.systems.dynamical_system import DynamicalSystem


# Custom types
ParamsRobots = dict[str,Array] # (batched) parameters of all robots: {"L":L, "D":D, ...}, where L has shape (n_robots, n_pcs), etc.


# Class
class MultiPcsSystem(DynamicalSystem):
    """
    Class for system of multiple soft robots.

    Attributes
    ----------
    n_robots : int
        Number of robots in the system.
    n_pcs : int
        Number of segments for the single robot.
    dim : int
        Dimension of the system.
    robots : PlanarPCS_simple
        `PlanarPCS_simple` pytree with batched leaves.
    """
    robots: PlanarPCS_simple
    n_robots: int = eqx.field(static=True)
    n_pcs: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    def __init__(self, n_robots: int, n_pcs: int, params_robots: ParamsRobots):
        """
        Initialize the system.
        
        Args
        ----
        n_robots : int
            Number of soft robots in the reservoir.
        n_pcs : int
            Number of segments for each PCS soft robot.
        params_robots: ParamsRobots
            Robots parameters, as a dictionary with batched values: {"L":L, "D":D, ...}, with 
            shape of L (n_robots, n_pcs), shape of D (n_robots, 3*n_pcs, 3*n_pcs), etc.
        """
        num_dofs = 3 * n_pcs * n_robots
        super().__init__(num_dofs=num_dofs, num_actuators=num_dofs)

        self.n_robots = n_robots
        self.n_pcs = n_pcs
        self.dim = num_dofs
        
        # Create list of robots
        robot_list = [
            PlanarPCS_simple(
                num_segments=n_pcs,
                params={k: v[i] for k, v in params_robots.items()},
            )
            for i in range(n_robots)
        ]
        # Convert list of robots into `PlanarPCS_simple` object with batched leaves.
        self.robots = jax.tree_util.tree_map(
            lambda *arrays: jnp.stack(arrays, axis=0),
            *robot_list,
        )

    def _get_robot(self, i: int) -> PlanarPCS_simple:
        """Extracts i-th robot from pytree. Returns the corresponding `PlanarPCS_simple` instance."""
        return jax.tree_util.tree_map(lambda x: x[i], self.robots)
    
    def transform_Z(self, Z: Array) -> Array:
        """
        Transforms system state Z = [Q^T, Qd^T]^T of shape (3*2*n_pcs*n_robots) into
        z_robots = [z1, z2, ..., zN] of shape (n_robots, 3*2*n_pcs).
        
        Args
        ----
        Z : Array
            System state in form Z = [Q^T, Qd^T]^T. Shape (3*2*n_pcs*n_robots)

        Returns
        -------
        z_robots : Array
            States of the robots in form z_robots = [z1, z2, ..., zN]. Shape (n_robots, 3*2*n_pcs)
        q_robots : Array
            Strains of the robots in form q_robots = [q1, q2, ..., qN]. Shape (n_robots, 3*n_pcs)
        qd_robots : Array
            Strains velocities of the robots in form qd_robots = [qd1, qd2, ..., qdN]. Shape (n_robots, 3*n_pcs)
        """
        Q, Qd = jnp.split(Z, 2) # extract Q and Qd from Z
        q_robots  = Q.reshape(self.n_robots, 3 * self.n_pcs) # Q shape (3*n_pcs*n_robots) -> (n_robots, 3*n_pcs)
        qd_robots = Qd.reshape(self.n_robots, 3 * self.n_pcs) # Qd shape (3*n_pcs*n_robots) -> (n_robots, 3*n_pcs)
        z_robots  = jnp.concatenate([q_robots, qd_robots], axis=-1)  # shape (n_robots, 2*3*n_pcs)
        return z_robots, q_robots, qd_robots
    
    @staticmethod
    def transform_z_robots(z_robots: Array) -> Array:
        """
        Transforms the states of the robots z_robots = [z1, z2, ..., zN] of shape (n_robots, 3*2*n_pcs) into
        Z = [Q^T, Qd^T]^T of shape (3*2*n_pcs*n_robots).
        
        Args
        ----
        z_robots : Array
            States of the robots in form z_robots = [z1, z2, ..., zN]. Shape (n_robots, 3*2*n_pcs)

        Returns
        -------
        Z : Array
            System state in form Z = [Q^T, Qd^T]^T. Shape (3*2*n_pcs*n_robots)
        q_robots : Array
            Strains of the robots in form q_robots = [q1, q2, ..., qN]. Shape (n_robots, 3*n_pcs)
        qd_robots : Array
            Strains velocities of the robots in form qd_robots = [qd1, qd2, ..., qdN]. Shape (n_robots, 3*n_pcs)
        """
        q_robots, qd_robots = jnp.split(z_robots, 2, axis=-1) # [qd1, ..., qdN] and [qdd1, ..., qddN]. Shape (n_robots, 3*n_pcs) each
        Q  = q_robots.reshape(-1) # [qd1^T, ..., qdN^T]^T. Shape (n_robots*3*n_pcs)
        Qd = qd_robots.reshape(-1)  # [qdd1^T, ..., qddN^T]^T. Shape (n_robots*3*n_pcs)
        Z = jnp.concatenate([Q, Qd])  # [Qd^T, Qdd^T]^T. Shape (2*n_robots*3*n_pcs,)
        return Z, q_robots, qd_robots
    
    def transform_Tau(self, Tau: Array) -> Array:
        """Transforms system actuation input Tau = [tau1^T, tau2^T, ..., tauN^T]^T of shape (3*n_pcs*n_robots) into
        tau_robots = [tau1, tau2, ..., tauN] of shape (n_robots, 3*n_pcs)."""
        tau_robots = Tau.reshape(self.n_robots, 3 * self.n_pcs)  # (3*n_pcs*n_robots) -> (n_robots, 3*n_pcs)
        return tau_robots
    
    @staticmethod
    def transform_tau_robots(tau_robots: Array) -> Array:
        """Transforms actuation inputs of the robots tau_robots = [tau1, tau2, ..., tauN] of shape (n_robots, 3*n_pcs) into
        Tau = [tau1^T, tau2^T, ..., tauN^T]^T of shape (3*n_pcs*n_robots)."""
        Tau = tau_robots.reshape(-1)
        return Tau
    
    @eqx.filter_jit
    def update_params(self, new_params: ParamsRobots) -> "MultiPcsSystem":
        """
        Update the parameters of all soft robots.

        Args
        ----
        new_params : ParamsRobots
            New robots parameters, as a dictionary with batched values: {"L":L, "D":D, ...}, with 
            shape of L (n_robots, n_pcs), shape of D (n_robots, 3*n_pcs, 3*n_pcs), etc.
        
        Returns
        -------
        updated_self : MultiPcsSystem
            New updated instance of the class.
        """
        # Update all robots
        new_robots = eqx.tree_at(
            lambda m: [getattr(m, k) for k in new_params],
            self.robots,
            [v for v in new_params.values()],
        )
        updated_self = eqx.tree_at(lambda m: m.robots, self, new_robots)
        return updated_self

    @eqx.filter_jit
    def forward_dynamics(self, t: Array, Z: Array, actuation_args: tuple | None = None) -> Array:
        """
        Computes forward dynamics for the entire system.

        Args
        ----
        t : Array
            Current time.
        Z : Array
            State of the entire system as Z = [Q^T, Qd^T]^T, where Q = [q1^T, ..., qN^T]^T
            and Qd = [qd1^T, ..., qdN^T]^T. Shape (2*3*n_pcs*n_robots)
        actuation_args : tuple | None
            Tuple of actuation inputs (Tau,) or (Tau, Tau_ext) where Tau is the actuation input
            with shape (3*n_pcs*n_robots,) and Tau_ext is the external force/torque. Optional, default: None.
        
        Returns
        -------
        Zd  : Array
            State derivative as Zd = [Qd^T, Qdd^T]^T. Shape (2*3*n_pcs*n_robots)
        """
        # Split the actuation arguments if provided
        if actuation_args is None:
            Tau = jnp.zeros(self.dim)
        elif len(actuation_args) == 1:
            Tau = actuation_args[0]
        elif len(actuation_args) == 2:
            Tau, _ = actuation_args
        else:
            raise ValueError("actuation_args must be a tuple of length 1 or 2.")

        # Reshape Z = [Q^T, Qd^T]^T = [q1^T, ..., qN^T, qd1^T, ..., qdN^T]^T to z_robots = [z1, ..., zN]. Shape (2*3*n_pcs*n_robots) -> (n_robots, 2*3*n_pcs)
        z_robots, _, _  = self.transform_Z(Z)  # shape (n_robots, 2*3*n_pcs)

        # Reshape Tau = [tau1^T, ..., tauN^T]^T to tau_robots = [tau1, ..., tauN]. Shape (3*n_pcs*n_robots) -> (n_robots, 3*n_pcs)
        tau_robots = self.transform_Tau(Tau)  # (3*n_pcs*n_robots) -> (n_robots, 3*n_pcs)

        # Apply forward dynamics to all robots with vmap
        def forward_single(robot: PlanarPCS_simple, z: Array, tau: Array) -> Array:
            """Forward dynamics for the single robot."""
            return robot.forward_dynamics(t, z, (tau,))
        zd_robots = jax.vmap(forward_single)(self.robots, z_robots, tau_robots) # zd_robots = [z1, ]. Shape (n_robots, 2*3*n_pcs)

        # Reshape zd_robots = [zd1, ..., zdN] to Zd = [Qd^T, Qdd^T]^T = [qd1^T, ..., qdN^T, qdd1^T, ..., qddN^T]^T. Shape (n_robots, 2*3*n_pcs) -> (2*3*n_pcs*n_robots)
        Zd, _, _ = self.transform_z_robots(zd_robots)  # [Qd^T, Qdd^T]^T. Shape (2*n_robots*3*n_pcs,)

        return Zd