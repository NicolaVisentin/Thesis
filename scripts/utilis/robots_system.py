import jax
from jax import numpy as jnp, Array
import equinox as eqx
from soromox.systems import LinkSpec
from soromox.systems.pcs import PlanarPCS, PlanarPCSStructure
from soromox.systems.dynamical_system import DynamicalSystem


# Custom types
ParamsRobots = dict[str,Array] # (batched) parameters of all robots: {"L":L, "D":D, ...}, where L has shape (n_robots, n_pcs), etc.


class PlanarPCSNoCoriolis(PlanarPCS):
    """Planar PCS model with the legacy scripts' simplified dynamics."""

    def dynamics_terms(
        self, q: Array, qd: Array, *, backend: str | None = None
    ) -> tuple[Array, Array, Array]:
        inertia, _, gravity = super().dynamics_terms(q, qd, backend=backend)
        return inertia, jnp.zeros_like(qd), gravity


def _segment_damping(damping: Array, num_segments: int) -> Array:
    damping = jnp.asarray(damping)
    if damping.ndim == 1 and damping.shape == (3 * num_segments,):
        damping = jnp.diag(damping)
    if damping.ndim == 2 and damping.shape == (num_segments, 3):
        return jax.vmap(jnp.diag)(damping)
    if damping.ndim == 2 and damping.shape == (3, 3) and num_segments == 1:
        return damping[None, ...]
    if damping.ndim == 2 and damping.shape == (3 * num_segments, 3 * num_segments):
        return jnp.stack(
            [
                damping[3 * index : 3 * index + 3, 3 * index : 3 * index + 3]
                for index in range(num_segments)
            ]
        )
    if damping.ndim == 3 and damping.shape == (num_segments, 3, 3):
        return damping
    raise ValueError(
        "D must be per-segment diagonal values, per-segment 3x3 matrices, "
        "or a (3*num_segments)x(3*num_segments) matrix."
    )


def build_planar_pcs(
    parameters: ParamsRobots,
    *,
    num_gauss_points: int = 5,
    coriolis: bool = False,
) -> PlanarPCS:
    """Build a current SoRoMoX planar PCS from the scripts' physical parameters."""
    lengths = jnp.asarray(parameters["L"])
    num_segments = lengths.shape[0]
    radii = jnp.broadcast_to(jnp.asarray(parameters["r"]), (num_segments,))
    densities = jnp.broadcast_to(jnp.asarray(parameters["rho"]), (num_segments,))
    young_moduli = jnp.broadcast_to(jnp.asarray(parameters["E"]), (num_segments,))
    shear_moduli = jnp.broadcast_to(jnp.asarray(parameters["G"]), (num_segments,))
    damping = _segment_damping(parameters["D"], num_segments)
    reference_strain = jnp.asarray(
        parameters.get("reference_strain", [0.0, 1.0, 0.0])
    )
    reference_strain = jnp.broadcast_to(reference_strain, (num_segments, 3))
    gravity = jnp.asarray(parameters.get("g", [0.0, -9.81]))
    theta = jnp.asarray(parameters.get("th0", 0.0))
    base_pose = jnp.array([theta, 0.0, 0.0])
    model_type = PlanarPCS if coriolis else PlanarPCSNoCoriolis
    links = [
        LinkSpec.circular(
            length=float(lengths[index]),
            radius=float(radii[index]),
            density=float(densities[index]),
            reference_strain=reference_strain[index],
            young_modulus=float(young_moduli[index]),
            shear_modulus=float(shear_moduli[index]),
            damping=damping[index],
        )
        for index in range(num_segments)
    ]
    return model_type.from_links(
        links,
        structure=PlanarPCSStructure(num_gauss_points=num_gauss_points),
        gravity=gravity,
        base_pose=base_pose,
    )


def update_planar_pcs(robot: PlanarPCS, parameters: ParamsRobots) -> PlanarPCS:
    """Update a PCS model from the physical parameter dictionaries used by scripts."""
    lengths = jnp.asarray(parameters["L"])
    num_segments = robot.num_segments
    radii = jnp.broadcast_to(jnp.asarray(parameters["r"]), (num_segments,))
    densities = jnp.broadcast_to(jnp.asarray(parameters["rho"]), (num_segments,))
    young_moduli = jnp.broadcast_to(jnp.asarray(parameters["E"]), (num_segments,))
    shear_moduli = jnp.broadcast_to(jnp.asarray(parameters["G"]), (num_segments,))
    damping = _segment_damping(parameters["D"], num_segments)

    area = jnp.pi * radii**2
    second_moment = jnp.pi * radii**4 / 4.0
    zeros = jnp.zeros_like(lengths)
    young_operator = lengths[:, None, None] * jax.vmap(jnp.diag)(
        jnp.stack([second_moment, area, zeros], axis=1)
    )
    shear_operator = lengths[:, None, None] * jax.vmap(jnp.diag)(
        jnp.stack([zeros, zeros, area], axis=1)
    )
    stiffness = (
        young_moduli[:, None, None] * young_operator
        + shear_moduli[:, None, None] * shear_operator
    )
    cross_section = robot.params.link.cross_section.replace(
        coefficients=radii[:, None]
    )
    links = robot.params.link.replace(
        length=lengths,
        density=densities,
        cross_section=cross_section,
        stiffness=stiffness,
        damping=damping,
    )
    return robot.with_params(robot.params.replace(link=links))


# Class
class MultiPcsSystem(DynamicalSystem):
    """
    Class for system of multiple soft robots. Basically is a batched PyTree representation of multiple
    `PlanarPCS` robots, where each leaf contains data for all robots stacked along the first dimension.

    Attributes
    ----------
    n_robots : int
        Number of robots in the system.
    n_pcs : int
        Number of segments for the single robot.
    dim : int
        Dimension of the system.
    robots : PlanarPCSNoCoriolis | PlanarPCS
        Batched representation of multiple planar PCS robots as a PyTree.
    """
    robots: PlanarPCSNoCoriolis | PlanarPCS
    n_robots: int = eqx.field(static=True)
    n_pcs: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    def __init__(self, n_robots: int, n_pcs: int, params_robots: ParamsRobots, coriolis: bool=False):
        """
        Initialize the system.
        
        Args
        ----
        n_robots : int
            Number of soft robots in the reservoir.
        n_pcs : int
            Number of segments for each PCS soft robot.
        params_robots : ParamsRobots
            Robots parameters, as a dictionary with batched values: {"L":L, "D":D, ...}, with 
            shape of L (n_robots, n_pcs), shape of D (n_robots, 3*n_pcs, 3*n_pcs), etc.
        coriolis : bool
            Whether to include or not Coriolis and centrifugal effects in the soft robot model
            (default: False).
        """
        num_dofs = 3 * n_pcs * n_robots
        super().__init__(num_actuators=num_dofs)

        self.n_robots = n_robots
        self.n_pcs = n_pcs
        self.dim = num_dofs
        
        # Create list of robots
        robot_list = [
            build_planar_pcs(
                {key: value[index] for key, value in params_robots.items()},
                coriolis=coriolis,
            )
            for index in range(n_robots)
        ]
        self.robots = jax.tree_util.tree_map(
            lambda *arrays: jnp.stack(arrays, axis=0),
            *robot_list,
        )

    def get_robot(self, i: int) -> PlanarPCSNoCoriolis | PlanarPCS:
        """Extract the i-th robot from the batched PyTree."""
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
        def update_single(robot, params_i):
            return update_planar_pcs(robot, params_i)
        
        # params_i is the dictionary with the values for the i-th robot
        new_robots = jax.vmap(update_single)(
            self.robots,
            {k: v for k, v in new_params.items()}
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
            Tau_ext = jnp.zeros(self.dim)
        elif len(actuation_args) == 1:
            Tau = actuation_args[0]
            Tau_ext = jnp.zeros(self.dim)
        elif len(actuation_args) == 2:
            Tau, Tau_ext = actuation_args
            if Tau_ext is None:
                Tau_ext = jnp.zeros(self.dim)
        else:
            raise ValueError("actuation_args must be a tuple of length 1 or 2.")

        # Reshape Z = [Q^T, Qd^T]^T = [q1^T, ..., qN^T, qd1^T, ..., qdN^T]^T to z_robots = [z1, ..., zN]. Shape (2*3*n_pcs*n_robots) -> (n_robots, 2*3*n_pcs)
        z_robots, _, _  = self.transform_Z(Z)  # shape (n_robots, 2*3*n_pcs)

        # Reshape Tau = [tau1^T, ..., tauN^T]^T to tau_robots = [tau1, ..., tauN]. Shape (3*n_pcs*n_robots) -> (n_robots, 3*n_pcs)
        tau_robots = self.transform_Tau(Tau)  # (3*n_pcs*n_robots) -> (n_robots, 3*n_pcs)
        tau_ext_robots = self.transform_Tau(Tau_ext)

        # Apply forward dynamics to all robots with vmap
        def forward_single(robot: PlanarPCSNoCoriolis | PlanarPCS, z: Array, tau: Array, tau_ext: Array) -> Array:
            """Forward dynamics for the single robot."""
            return robot.forward_dynamics(t, z, (tau, tau_ext))
        zd_robots = jax.vmap(forward_single)(self.robots, z_robots, tau_robots, tau_ext_robots) # zd_robots = [z1, ]. Shape (n_robots, 2*3*n_pcs)

        # Reshape zd_robots = [zd1, ..., zdN] to Zd = [Qd^T, Qdd^T]^T = [qd1^T, ..., qdN^T, qdd1^T, ..., qddN^T]^T. Shape (n_robots, 2*3*n_pcs) -> (2*3*n_pcs*n_robots)
        Zd, _, _ = self.transform_z_robots(zd_robots)  # [Qd^T, Qdd^T]^T. Shape (2*n_robots*3*n_pcs,)

        return Zd