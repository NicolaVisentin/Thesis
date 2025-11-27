import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx

from functools import partial
from typing import Sequence, Tuple, List

# Custom types definition
Params = List[Tuple[Array, Array]]  # [(W1, b1), (W2, b2), ...] for all layers

# Class
class MLP(eqx.Module):
    """
    Simple class for fully-connected MLP with tanh activations.

    Attributes
    ----------
    layer_sizes : List[int]
        List with the sizes of the layers, i.e. [in_size, hid1, hid2, ..., out_size].
    params : Params
        List of tuples with the layers weights and biases, i.e. [(W1,b1), (W2,b2), ...].
    """
    layer_sizes: Sequence[int]
    params: Params

    def __init__(self, key: jax.random.key, layer_sizes: Sequence[int]):
        """
        Initializes an MLP (tanh activations) with given layer sizes.

        Args
        ----
        key : jax.random.key
            Random key for parameters initialization.
        layer_sizes : list[int]
            List with the sizes of the layers, i.e. [in_size, hid1, hid2, ..., out_size].
        """
        self.layer_sizes = layer_sizes
        self.params = self._init_params(key)
    
    def _init_params(self, key: jax.random.key) -> Params:
        """Glorot/Xavier initialization of weights and null biases."""
        keys = jax.random.split(key, len(self.layer_sizes) - 1)
        params = []
        for k, (m, n) in zip(keys, zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            limit = jnp.sqrt(6.0 / (m + n))
            W = jax.random.uniform(k, (n, m), minval=-limit, maxval=limit)
            b = jnp.zeros((n,))
            params.append((W, b))
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
        for W, b in self.params[:-1]:
            x = jnp.tanh(W @ x + b)
        # last layer: no activation
        W, b = self.params[-1]
        out = W @ x + b
        return out
    
    @staticmethod
    @eqx.filter_jit
    def _forward_single(params: Params, x: Array) -> Array:
        """Forward pass for a single input sample x. Uses external parameters."""
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
        """Forward pass for a batch of input samples. Uses external parameters."""
        batched_forward = jax.vmap(self._forward_single, in_axes=(None,0))
        out_batch = batched_forward(params, x_batch)
        return out_batch

    @eqx.filter_jit
    def __call__(self, x: Array) -> Array:
        """
        __call__ that implements `forward` method, i.e. forward pass of the network for a single
        input, using internal parameters.
        """
        return self.forward_single(x)
    
    def save_params(self, path: str):
        """Saves the parameters of the network in the specified path as a .npz file."""
        flat_params = {}
        for i, (W, b) in enumerate(self.params):
            flat_params[f"W_{i}"] = np.array(W)
            flat_params[f"b_{i}"] = np.array(b)
        np.savez(path, **flat_params)
    
    @staticmethod
    def load_params(path: str) -> Params:
        """Load parameters of the network from a .npz file."""
        with np.load(path) as data:
            keys = sorted(data.files)  # sorted to ensure correct order: W_0, b_0, ...
            num_layers = len(keys) // 2
            params = []
            for i in range(num_layers):
                W = jnp.array(data[f"W_{i}"])
                b = jnp.array(data[f"b_{i}"])
                params.append((W, b))
        return params