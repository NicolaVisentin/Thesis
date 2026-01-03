import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx

from typing import Sequence, Tuple, List

# Custom types definition
Params = List[Tuple[Array, Array]]  # [(W1, b1), (W2, b2), ...] for all layers

# Class
class MLP(eqx.Module):
    """
    Simple class for fully-connected MLP with tanh or relu activations.

    Attributes
    ----------
    layer_sizes : List[int]
        List with the sizes of the layers, i.e. [in_size, hid1, hid2, ..., out_size].
    params : Params
        List of tuples with the layers weights and biases, i.e. [(W1,b1), (W2,b2), ...].
    scale_init : float
        Scaling factor for the initialization of the parameters.
    activation_fn : str
        Activation function. Either 'tanh' or 'relu' (default: tanh).
    """
    layer_sizes: Sequence[int]
    params: Params
    scale_init: float
    activation_fn: str

    def __init__(self, key: jax.random.key, layer_sizes: Sequence[int], scale_init: int=1.0, activation_fn: str='tanh'):
        """
        Initializes an MLP (tanh/relu activations) with given layer sizes.

        Args
        ----
        key : jax.random.key
            Random key for parameters initialization.
        layer_sizes : list[int]
            List with the sizes of the layers, i.e. [in_size, hid1, hid2, ..., out_size].
        scale_init : float
            Optional scaling factor for the initialization of the weights (default: 1.0).
        """
        self.layer_sizes = layer_sizes
        self.scale_init = scale_init
        self.params = self._init_params(key)
        self.activation_fn = activation_fn
    
    def _init_params(self, key: jax.random.key) -> Params:
        """
        If tanh activations: Glorot/Xavier initialization of weights and null biases, with optional scaling factor.
        If relu activations: He initialization of weights and null biases, with optional scaling factor.
        """
        keys = jax.random.split(key, len(self.layer_sizes) - 1)
        params = []
        if self.activation_fn == 'tanh':
            for k, (m, n) in zip(keys, zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
                limit = jnp.sqrt(6.0 / (m + n))
                W = self.scale_init*jax.random.uniform(k, (n, m), minval=-limit, maxval=limit)
                b = jnp.zeros((n,))
                params.append((W, b))
        elif self.activation_fn == 'relu':
            for k, (m, n) in zip(keys, zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
                stddev = jnp.sqrt(2.0 / m)
                W = jax.random.normal(k, (n, m)) * stddev
                b = jnp.zeros((n,))
                params.append((W, b))
        else:
            raise ValueError('Choose a valid activation.')
        return params
    
    @eqx.filter_jit
    def update_params(self, new_params : Params) -> "MLP":
        """
        Returns a new instance of the class with updated parameters.
        """
        # copy istance
        updated_self = self
        # assign new parameters
        updated_self = eqx.tree_at(lambda m: m.params, updated_self, new_params)
        return updated_self
    
    @eqx.filter_jit
    def forward_single(self, x: Array) -> Array:
        """Forward pass for a single input sample x. Uses internal parameters."""
        # pass from input to second-last layer (use activation)
        if self.activation_fn == 'tanh':
            for W, b in self.params[:-1]:
                x = jnp.tanh(W @ x + b)
        else:
            for W, b in self.params[:-1]:
                x = jax.nn.relu(W @ x + b)
        # last layer: no activation
        W, b = self.params[-1]
        out = W @ x + b
        return out
    
    @staticmethod
    @eqx.filter_jit
    def _forward_single(params: Params, x: Array) -> Array:
        """Forward pass for a single input sample x. Uses external parameters. ! ONLY WORKS WITH TANH ACTIVATIONS !"""
        # pass from input to second-last layer (use activation)
        for W, b in params[:-1]:
            x = jnp.tanh(W @ x + b)
        # last layer: no activation
        W, b = params[-1]
        out = W @ x + b
        return out
    
    @eqx.filter_jit
    def forward_batch(self, x_batch: Array) -> Array:
        """Forward pass for a batch of input samples. Uses internal parameters."""
        batched_forward = jax.vmap(self.forward_single)
        out_batch = batched_forward(x_batch)
        return out_batch
    
    @eqx.filter_jit
    def _forward_batch(self, params: Params, x_batch: Array) -> Array:
        """Forward pass for a batch of input samples. Uses external parameters. ! ONLY WORKS WITH TANH ACTIVATIONS !"""
        batched_forward = jax.vmap(self._forward_single, in_axes=(None,0))
        out_batch = batched_forward(params, x_batch)
        return out_batch
    
    @eqx.filter_jit
    def forward_xd(self, x: Array, xd: Array) -> Tuple[Array, Array]:
        """Given input x and input derivative xd, computes output y and output derivative yd = J(x) * xd 
        where J is the Jacobian of the net evaluated in x. Avoids the direct computation of J."""
        y, yd = jax.jvp(self.forward_single, (x,), (xd,))
        return (y, yd)
    
    @eqx.filter_jit
    def forward_xd_batch(self, x_batch: Array, xd_batch: Array) -> Tuple[Array, Array]:
        """Batched version of `forward_xd` method."""
        y_batch, yd_batch = jax.vmap(self.forward_xd)(x_batch, xd_batch)
        return (y_batch, yd_batch)
    
    @eqx.filter_jit
    def forward_xdd(self, x: Array, xd: Array, xdd: Array) -> Array:
        """Given input x, input derivative xd and input acceleration xdd, computes output acceleration 
        ydd = J(x) * xdd + H(x) * (x x^T), where H(x) is the Hessian of the net evaluated in x in tensorial
        form and (x x^T) a matrix. Avoids the direct computation of H."""
        def fun(x, xd):
            return self.forward_xd(x, xd)[1]
        _, ydd = jax.jvp(fun, (x, xd), (xd, xdd))
        return ydd
    
    @eqx.filter_jit
    def forward_xdd_batch(self, x_batch: Array, xd_batch: Array, xdd_batch: Array) -> Array:
        """Batched version of `forward_xdd` method."""
        ydd_batch = jax.vmap(self.forward_xdd)(x_batch, xd_batch, xdd_batch)
        return ydd_batch

    @eqx.filter_jit
    def __call__(self, x: Array) -> Array:
        """
        __call__ that implements `forward` method, i.e. forward pass of the network for a single
        input, using internal parameters.
        """
        return self.forward_single(x)
    
    @eqx.filter_jit
    def compute_jacobian(self, x: Array) -> Array:
        """Computes the Jacobian of the network wrt the input x at a given x."""
        return jax.jacfwd(self.forward_single)(x) # shape (n_out, n_in)
    
    @eqx.filter_jit
    def compute_hessian(self, x: Array) -> Array:
        """Computes the Hessian of the network wrt the input x at a given x."""
        return jax.hessian(self.forward_single)(x) # shape (n_out, n_in, n_in)
    
    def save_params(self, path: str):
        """Saves the parameters of the network in the specified path as a .npz file."""
        flat_params = {}
        for i, (W, b) in enumerate(self.params):
            flat_params[f"W_{i}"] = np.array(W)
            flat_params[f"b_{i}"] = np.array(b)
        np.savez(path, **flat_params)

    @staticmethod
    def _save_params(params: Params, path: str):
        """Saves externally provided parameters params: Params in the specified path as a .npz file."""
        flat_params = {}
        for i, (W, b) in enumerate(params):
            flat_params[f"W_{i}"] = np.array(W)
            flat_params[f"b_{i}"] = np.array(b)
        np.savez(path, **flat_params)
    
    @staticmethod
    def load_params(path: str, load_as_batch: bool=False) -> Params:
        """Load parameters of the network from a .npz file. If load_as_batch is True, then loads
        the parameters as they were a batch of 1 element (i.e. adds a dimension to each parameter)."""
        with np.load(path) as data:
            keys = sorted(data.files)  # sorted to ensure correct order: W_0, b_0, ...
            num_layers = len(keys) // 2
            params = []
            for i in range(num_layers):
                if load_as_batch:
                    W = jnp.array(data[f"W_{i}"][None,:])
                    b = jnp.array(data[f"b_{i}"][None,:])
                else:
                    W = jnp.array(data[f"W_{i}"])
                    b = jnp.array(data[f"b_{i}"])
                params.append((W, b))
        return params
    
    def init_params_batch(self, key, batch_size: int) -> Params:
        """"
        Gives params as a list of tuples with batches of parameters, i.e.:
        params_batch = [(W1_batch, b1_batch), (W2_batch, b2_batch), ...] where
        W1.shape = (batch_size, dim(W1)), b1_shape = (batch_size, dim(b1)), etc.
        """
        keys = jax.random.split(key, batch_size)
        return jax.vmap(self._init_params)(keys)
    
    @staticmethod
    def extract_params_from_batch(params_batch: Params, idx: int, extract_as_batch: bool=False) -> Params:
        """
        If params_batch is a Params list that contains tuples of batched parameters, this
        method extracts a Params list with tuples of the parameters in position idx within
        the batches.

        Args
        ----
        params_batch : Params
            List of tuples with batches of parameters, i.e.:
            params_batch = [(W1, b1), (W2, b2), ...] where Wi.shape=(batch_size, ...) and
            bi.shape=(batch_size, ...).     
        idx : int
            Position in the batch of the parameters to extract.
        extract_as_batch: bool
            If True, extracts desired parameters as they were a batch of dimension 1 (default: False).

        Returns
        -------
        params : Params
            List of desired parameters.
        """
        if extract_as_batch:
            out = [(W[idx][None,:], b[idx][None,:]) for (W, b) in params_batch]
        else:
            out = [(W[idx], b[idx]) for (W, b) in params_batch]
        return out