import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx

from typing import Tuple, List

from .mlp import MLP, Params

# Custom types definition
ParamsRealNVP = List[Tuple[Params, Params, Array]]  # [(Params_translation_net1, Params_scale_net1, scale_factor1), (Params_translation_net2, Params_scale_net2, scale_factor2), ...] for all coupling layers

# Affine coupling layer
class AffineCoupling(eqx.Module):
    """
    Affine coupling layer for normalizing flows.
    
    Attributes
    ----------
    input_dim : int
        Input dimension.
    hidden_dim : int
        Hidden dimension for the neural networks.
    mask : Array
        Binary mask (1 = unchanged positions, 0 = transformed positions).
    scale_net : MLP
        MLP for computing the scale.
    translation_net : MLP
        MLP for computing the translation.
    scale_param : Array
        Learnable scaling parameter.
    activation_fn : str
        Activation function of the MLPs.
    """
    input_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    activation_fn : str = eqx.field(static=True)
    mask: Array
    scale_net: "MLP"
    translation_net: "MLP"
    scale_param: Array

    def __init__(self, key: jax.random.key, mask: Array, hidden_dim: int, activation_fn: str='relu'):
        """
        Initialize affine coupling layer.
        
        Args
        ----
        key : jax.random.key
            Random key for initialization.
        mask : Array
            Binary mask for splitting input.
        hidden_dim : int
            Hidden layer dimension.
        activation_fn : str
            Activation function for the MLPs. Either 'tanh' or 'relu' (default: 'relu').
        """
        self.input_dim = len(mask)
        self.hidden_dim = hidden_dim
        self.mask = mask
        self.activation_fn = activation_fn

        # Initialize scale and translation networks (MLPs)
        key_s, key_t, key_param = jax.random.split(key, 3)        
        
        layer_sizes_scale = [self.input_dim, self.hidden_dim, self.hidden_dim, self.input_dim]
        self.scale_net = MLP(key_s, layer_sizes_scale, activation_fn=self.activation_fn)

        layer_sizes_trans = [self.input_dim, self.hidden_dim, self.hidden_dim, self.input_dim]
        self.translation_net = MLP(key_t, layer_sizes_trans, activation_fn=self.activation_fn)
        
        # Initialize scale parameter with normal distribution
        self.scale_param = jax.random.normal(key_param, (self.input_dim,))

    @eqx.filter_jit
    def update_params(self, new_params : Tuple[Params, Params, Array]) -> "AffineCoupling":
        """
        Returns a new instance of the class with updated parameters.
        """
        # extract parameters
        t_net_new_params, s_net_new_params, scale_factor_new = new_params
        # copy istance
        updated_self = self
        # assign new parameters
        updated_self = eqx.tree_at(lambda m: m.translation_net.params, updated_self, t_net_new_params)
        updated_self = eqx.tree_at(lambda m: m.scale_net.params, updated_self, s_net_new_params)
        updated_self = eqx.tree_at(lambda m: m.scale_param, updated_self, scale_factor_new)
        return updated_self
    
    @eqx.filter_jit
    def _compute_scale(self, x: Array) -> Array:
        """Compute scaling using masked input."""
        x_a = x * self.mask # extract unchanged part of x
        s = self.scale_net(x_a)
        if self.scale_net.activation_fn == 'relu':
            s = jax.nn.relu(s) * self.scale_param # ! MLP class does not apply activation on the last layer, so we manually add it. Moreover, we add scaling factor
        else:
            s = jnp.tanh(s) * self.scale_param # ! MLP class does not apply activation on the last layer, so we manually add it. Moreover, we add scaling factor
        return s
    
    @eqx.filter_jit
    def _compute_translation(self, x: Array) -> Array:
        """Compute translation using masked input."""
        x_a = x * self.mask # extract unchanged part of x
        t = self.translation_net(x_a)
        return t
    
    @eqx.filter_jit
    def forward(self, x: Array) -> Tuple[Array, Array]:
        """Forward transformation x -> y for the single layer."""
        s = self._compute_scale(x)
        t = self._compute_translation(x)
        
        y = self.mask * x + (1 - self.mask) * (x * jnp.exp(s) + t)
        
        return y
    
    @eqx.filter_jit
    def inverse(self, y: Array) -> Tuple[Array, Array]:
        """Inverse transformation y -> x for the single layer."""
        s = self._compute_scale(y)
        t = self._compute_translation(y)
        
        x = self.mask * y + (1 - self.mask) * ((y - t) * jnp.exp(-s))
        
        return x


# RealNVP in JAX/Equinox
class RealNVP(eqx.Module):
    """
    RealNVP normalizing flow.
    
    Attributes
    ----------
    hidden_dim : int
        Hidden dimension for networks in coupling layers.
    masks : List[Array]
        List of masks defining coupling layers. It is a list of as many elements as the number of
        coupling layers, and each element is a shape (dim_input,) binary array.
    affine_couplings : List[AffineCoupling]
        List of affine coupling layers.
    params : ParamsRealNVP
        Parameters of the transformation. It is a list with as many elements as the number of coupling layers. 
        Each element is a tuple with parameters of one coupling layer, i.e. parameters of the two MLPs and the
        scaling factor.
    activation_fn : str
        Activation function for the MLPs in the coupling layers.
    """
    hidden_dim : int = eqx.field(static=True)
    activation_fn : str = eqx.field(static=True)
    masks : List[Array]
    affine_couplings : List[AffineCoupling]
    params : ParamsRealNVP
    
    def __init__(self, key: jax.random.key, masks: List[Array], hidden_dim: int, activation_fn: str='relu'):
        """
        Initialize RealNVP flow.
        
        Args
        ----
        key : jax.random.key
            Random key for initialization.
        masks : List[Array]
            List of masks defining coupling layers. It is a list of as many elements as the number of
            coupling layers, and each element is a shape (dim_input,) binary array.
        hidden_dim : int
            Hidden dimension for networks in coupling layers.
        activation_fn : str
            Activation function for the MLPs in the coupling layers (default: 'relu').
        """
        self.hidden_dim = hidden_dim
        self.masks = masks
        self.activation_fn = activation_fn
        
        # Initialize coupling layers
        keys = jax.random.split(key, len(masks))
        self.affine_couplings = [
            AffineCoupling(keys[i], masks[i], hidden_dim, activation_fn=self.activation_fn)
            for i in range(len(masks))
        ]
        
        self.params = [
            (layer.translation_net.params, layer.scale_net.params, layer.scale_param)
            for layer in self.affine_couplings
        ]

    @eqx.filter_jit
    def update_params(self, new_params: ParamsRealNVP) -> "RealNVP":
        affine_couplings_new = [
            coupling.update_params(layer_params)
            for coupling, layer_params in zip(self.affine_couplings, new_params)
        ]

        return eqx.tree_at(
            lambda m: m.affine_couplings,
            self,
            affine_couplings_new
    )

    @eqx.filter_jit
    def forward(self, x: Array) -> Tuple[Array, Array]:
        """
        Forward pass: latent -> observed.
        
        Args
        ----
        x : Array
            Input from latent space.
            
        Returns
        -------
        y : Array
            Output in observed space.
        """
        y = x
        # Apply coupling layers
        for coupling in self.affine_couplings:
            y = coupling.forward(y)  
        return y
    
    @eqx.filter_jit
    def inverse(self, y: Array) -> Tuple[Array, Array]:
        """
        Inverse pass: observed -> latent.
        
        Args
        ----
        y : Array
            Input from observed space.
            
        Returns
        -------
        x : Array
            Output in latent space.
        """
        x = y        
        # Apply inverse coupling layers in reverse order
        for coupling in reversed(self.affine_couplings):
            x = coupling.inverse(x)
        return x
    
    @eqx.filter_jit
    def forward_batch(self, x_batch: Array) -> Tuple[Array, Array]:
        """Batched version of `forward` method."""
        batched_forward = jax.vmap(self.forward)
        y_batch = batched_forward(x_batch)
        return y_batch
    
    @eqx.filter_jit
    def inverse_batch(self, y_batch: Array) -> Tuple[Array, Array]:
        """Batched version of `inverse` method."""
        batched_inverse = jax.vmap(self.inverse)
        x_batch = batched_inverse(y_batch)
        return x_batch
    
    @eqx.filter_jit
    def __call__(self, x: Array) -> Tuple[Array, Array]:
        """Call method implements forward pass."""
        return self.forward(x)
    
    @eqx.filter_jit
    def forward_with_derivatives(self, x: Array, xd: Array, xdd: Array=None) -> Array:
        """Given input x, input derivative xd and input acceleration xdd, computes output y, output derivative yd = J(x) * xd
        and output acceleration ydd = J(x) * xdd + H(x) * (x x^T), where J(x) is the Jacobian of the net evaluated in x, H(x) 
        is the Hessian of the net evaluated in x in tensorial form and (x x^T) a matrix. Avoids the direct computation of J and H."""
        # first jvp: compute y and yd
        y, yd = jax.jvp(self.forward, (x,), (xd,))
        if xdd is not None:
            # second jvp: compute ydd
            _, ydd = jax.jvp(
                lambda x, xd: jax.jvp(self.forward, (x,), (xd,))[1],
                (x, xd), 
                (xd, xdd)
            )
            out = (y, yd, ydd)
        else:
            out = (y, yd)
        return out
    
    @eqx.filter_jit
    def forward_with_derivatives_batch(self, x_batch: Array, xd_batch: Array, xdd_batch: Array=None) -> Array:
        """Batched version of `forward_with_derivatives` method."""
        if xdd_batch is not None:
            y_batch, yd_batch, ydd_batch = jax.vmap(self.forward_with_derivatives)(x_batch, xd_batch, xdd_batch)
            out = (y_batch, yd_batch, ydd_batch)
        else:
            y_batch, yd_batch = jax.vmap(self.forward_with_derivatives)(x_batch, xd_batch)
            out = (y_batch, yd_batch)
        return out
    
    @eqx.filter_jit
    def inverse_with_derivatives(self, y: Array, yd: Array, ydd: Array=None) -> Array:
        """Given input y, input derivative yd and input acceleration ydd, computes output x, output derivative xd = J(y) * yd
        and output acceleration xdd = J(y) * ydd + H(y) * (y y^T), where J(y) is the Jacobian of the net evaluated in y, H(y) 
        is the Hessian of the net evaluated in y in tensorial form and (y y^T) a matrix. Avoids the direct computation of J and H."""
        # first jvp: compute x and xd
        x, xd = jax.jvp(self.inverse, (y,), (yd,))
        if ydd is not None:
            # second jvp: compute xdd
            _, xdd = jax.jvp(
                lambda y, yd: jax.jvp(self.inverse, (y,), (yd,))[1],
                (y, yd), 
                (yd, ydd)
            )
            out = (x, xd, xdd)
        else:
            out = (x, xd)
        return out
    
    @eqx.filter_jit
    def inverse_with_derivatives_batch(self, y_batch: Array, yd_batch: Array, ydd_batch: Array=None) -> Array:
        """Batched version of `inverse_with_derivatives` method."""
        if ydd_batch is not None:
            x_batch, xd_batch, xdd_batch = jax.vmap(self.inverse_with_derivatives)(y_batch, yd_batch, ydd_batch)
            out = (x_batch, xd_batch, xdd_batch)
        else:
            x_batch, xd_batch = jax.vmap(self.inverse_with_derivatives)(y_batch, yd_batch)
            out = (x_batch, xd_batch)
        return out


# Utility function to create alternating masks
def create_alternating_masks(input_dim: int, num_layers: int) -> List[Array]:
    """
    Create alternating binary masks for coupling layers.
    
    Args
    ----
    input_dim : int
        Dimension of input.
    num_layers : int
        Number of coupling layers.
        
    Returns
    -------
    masks : List[Array]
        List of alternating binary masks. It's a list of num_layers elements, where each i-th element is
        a shape (input_dim,) binary mask for the i-th coupling layer.
    """
    masks = []
    for i in range(num_layers):
        mask = jnp.array([i % 2 if j % 2 == 0 else (i + 1) % 2 
                          for j in range(input_dim)])
        masks.append(mask)
    return masks