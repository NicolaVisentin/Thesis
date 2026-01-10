# Imports
from typing import Dict, Tuple, Callable
from functools import partial
import jax
from jax import numpy as jnp
from jax import Array
import optax
from .generic import *


# Optimization train step
@partial(jax.jit, static_argnums=(0,2))
def train_step(
    loss_fn : Callable,
    optimiz_state,
    optimiz_update,
    params_optimiz : Tuple,
    train_batch : Dict,
):
    """
    Optimization step. Takes current optimization parameters and optimizer status and computes loss and gradients propagation
    in a batch of data, finally updates and returns the optimization parameters and optimizer status (and the loss/grads). 

    Args
    ----
    loss_fn
        Loss function as loss_fn(params_optimiz: Tuple, data_batch: Dict) -> Tuple[float, Dict]. Gives loss (float) and metrics
        (Dict) as outputs.
    optimiz_state
        Optimizer state. It's an optax object.
    optimiz_update
        Optimizer update. It's an optax object.
    params_optimiz : Tuple
        Collects all optimization parameters.
    train_batch : Dict
        Dictionary with datapoints and labels of a batch of the train set.

    Returns
    -------
    params_optimiz_updated
        Updated optimization parameters.
    optimiz_state_updated
        Updated optimizer state.
    loss
        Loss value for this batch.
    grads
        Gradients.
    metrics
        Metrics dictionary for this batch.
    """
    # compute loss and gradients
    loss_and_grads = jax.jit(jax.value_and_grad(loss_fn, argnums=0, has_aux=True))
    (loss, metrics), grads = loss_and_grads(params_optimiz, train_batch)
    
    # update optimization parameters
    updates, optimiz_state = optimiz_update(grads, optimiz_state, params_optimiz)
    params_optimiz = optax.apply_updates(params_optimiz, updates)

    return params_optimiz, optimiz_state, loss, grads, metrics


# Run the entire optimization with lax.scan. Supports loss in form Loss(params_optimiz, data_batch)
def train_with_scan(
        key: Array,
        optimizer,
        params_optimiz : Tuple,
        loss_fn: Callable,
        train_set: Dict,
        val_set: Dict,
        n_iter: int,
        batch_size: int,
    ) -> Dict:
    """
    Perform a full training loop (epochs + batches) using jax.lax.scan. This function performs:
    - inner loop: iterating over batches within an epoch
    - outer loop: iterating over epochs
    Both loops are JIT-compiled.

    Args
    ----
    key : Array
        Random key for shuffling training data.
    optimizer
        Optimizer instance from optax (must have .update and .init methods).
    params_optimiz : Tuple
        Initial parameters to be optimized.
    loss_fn : Callable
        Function computing loss and metrics: `loss_fn(params_optimiz, data_batch) -> (loss, metrics_dict)`.
        Note: metrics_dict must have key "MSE".
    train_set : Dict
        Training dataset.
    val_set : Dict
        Validation dataset.
    n_iter : int
        Number of epochs.
    batch_size : int
        Number of samples per batch.

    Returns
    -------
    results : Dict
        Dictionary containing:
        - "params_optimiz": optimized parameters
        - "train_loss_ts": array of training losses vs epochs
        - "train_MSE_ts": array of training MSE vs epochs
        - "val_loss_ts": array of validation losses vs epochs
        - "val_MSE_ts ": array of validation MSE vs epochs
    """
    # Inner loop function: train one epoch
    @partial(jax.jit, static_argnums=(4,5))
    def train_one_epoch(key, optimiz_state, params_optimiz, train_set, batch_size, train_size):
        # get indices to extract shuffled batches
        key, subkey = jax.random.split(key)
        batch_ids = batch_indx_generator(key=subkey, dataset_size=train_size, batch_size=batch_size)

        # step function for the batches
        def batch_step(carry, batch_i_ids):
            optimiz_state, params_optimiz = carry
            # extract a random batch
            train_batch = extract_batch(train_set, batch_i_ids)
            # update parameters
            params_optimiz, optimiz_state, loss, _, train_metrics = train_step(
                loss_fn=loss_fn,
                optimiz_state=optimiz_state,
                optimiz_update=optimizer.update,
                params_optimiz=params_optimiz,
                train_batch=train_batch,
            )
            return (optimiz_state, params_optimiz), (loss, train_metrics["MSE"])

        # run scan on the batches to complete one epoch
        (optimiz_state, params_optimiz), (loss_vec, MSE_vec) = jax.lax.scan(
            batch_step,
            (optimiz_state, params_optimiz),
            batch_ids
        )
        # compute train loss for this epoch
        train_loss_epoch = jnp.mean(loss_vec)
        train_MSE_epoch = jnp.mean(MSE_vec)
        return key, optimiz_state, params_optimiz, train_loss_epoch, train_MSE_epoch
    
    # Outer loop: iterate over epochs
    train_size = next(iter(train_set.values())).shape[0]
    def epoch_step(carry, _):
        key, optimiz_state, params_optimiz = carry
        # run inner loop (perform training for one epoch)
        key, optimiz_state, params_optimiz, train_loss_epoch, train_MSE_epoch = train_one_epoch(
            key, optimiz_state, params_optimiz,
            train_set,
            batch_size, train_size
        )
        # perform validation after training on the epoch
        val_loss_epoch, val_metrics = loss_fn(
            params_optimiz=params_optimiz,
            data_batch=val_set,
        )
        return (key, optimiz_state, params_optimiz), (train_loss_epoch, train_MSE_epoch, val_loss_epoch, val_metrics["MSE"])

    # Run scan on outer loop (epochs)
    optimiz_state = optimizer.init(params_optimiz)
    (_, optimiz_state, params_optimiz), (train_loss_ts, train_MSE_ts, val_loss_ts, val_MSE_ts) = jax.lax.scan(
        epoch_step,
        (key, optimiz_state, params_optimiz),
        xs=None,
        length=n_iter
    )
    results = {
        "params_optimiz": params_optimiz,
        "train_loss_ts": train_loss_ts,
        "train_MSE_ts": train_MSE_ts,
        "val_loss_ts": val_loss_ts,
        "val_MSE_ts": val_MSE_ts,
    }
    return results


# Run the entire optimization with lax.scan, BUT has one additional (static) argument that 
# is passed to the loss function and can be vmapped but is NOT used for gradient computation.
# Supports loss in form Loss(params_optimiz, data_batch, additional_arg).
def train_with_scan_modified(
        key: Array,
        optimizer,
        params_optimiz : Tuple,
        loss_fn: Callable,
        train_set: Dict,
        val_set: Dict,
        n_iter: int,
        batch_size: int,
        additional_loss_argument,
    ) -> Dict:
    """
    Perform a full training loop (epochs + batches) using jax.lax.scan. This function performs:
    - inner loop: iterating over batches within an epoch
    - outer loop: iterating over epochs
    Both loops are JIT-compiled.

    Args
    ----
    key : Array
        Random key for shuffling training data.
    optimizer
        Optimizer instance from optax (must have .update and .init methods).
    params_optimiz : Tuple
        Initial parameters to be optimized.
    loss_fn : Callable
        Function computing loss and metrics: `loss_fn(params_optimiz, data_batch, additional_arg) -> (loss, metrics_dict)`.
        Note: metrics_dict must have key "MSE". Note: additional_arg must have that exact signature name in the loss definition.
    train_set : Dict
        Training dataset.
    val_set : Dict
        Validation dataset.
    n_iter : int
        Number of epochs.
    batch_size : int
        Number of samples per batch.
    additional_loss_argument
        Additional argument for the loss function.

    Returns
    -------
    results : Dict
        Dictionary containing:
        - "params_optimiz": optimized parameters
        - "train_loss_ts": array of training losses vs epochs
        - "train_MSE_ts": array of training MSE vs epochs
        - "val_loss_ts": array of validation losses vs epochs
        - "val_MSE_ts ": array of validation MSE vs epochs
    """
    # Change signature of loss function to adapt to the rest of the code
    loss_fn = partial(loss_fn, additional_arg=additional_loss_argument)

    # Inner loop function: train one epoch
    @partial(jax.jit, static_argnums=(4,5))
    def train_one_epoch(key, optimiz_state, params_optimiz, train_set, batch_size, train_size):
        # get indices to extract shuffled batches
        key, subkey = jax.random.split(key)
        batch_ids = batch_indx_generator(key=subkey, dataset_size=train_size, batch_size=batch_size)

        # step function for the batches
        def batch_step(carry, batch_i_ids):
            optimiz_state, params_optimiz = carry
            # extract a random batch
            train_batch = extract_batch(train_set, batch_i_ids)
            # update parameters
            params_optimiz, optimiz_state, loss, _, train_metrics = train_step(
                loss_fn=loss_fn,
                optimiz_state=optimiz_state,
                optimiz_update=optimizer.update,
                params_optimiz=params_optimiz,
                train_batch=train_batch,
            )
            return (optimiz_state, params_optimiz), (loss, train_metrics["MSE"])

        # run scan on the batches to complete one epoch
        (optimiz_state, params_optimiz), (loss_vec, MSE_vec) = jax.lax.scan(
            batch_step,
            (optimiz_state, params_optimiz),
            batch_ids
        )
        # compute train loss for this epoch
        train_loss_epoch = jnp.mean(loss_vec)
        train_MSE_epoch = jnp.mean(MSE_vec)
        return key, optimiz_state, params_optimiz, train_loss_epoch, train_MSE_epoch
    
    # Outer loop: iterate over epochs
    train_size = next(iter(train_set.values())).shape[0]
    def epoch_step(carry, _):
        key, optimiz_state, params_optimiz = carry
        # run inner loop (perform training for one epoch)
        key, optimiz_state, params_optimiz, train_loss_epoch, train_MSE_epoch = train_one_epoch(
            key, optimiz_state, params_optimiz,
            train_set,
            batch_size, train_size
        )
        # perform validation after training on the epoch
        val_loss_epoch, val_metrics = loss_fn(
            params_optimiz=params_optimiz,
            data_batch=val_set,
        )
        return (key, optimiz_state, params_optimiz), (train_loss_epoch, train_MSE_epoch, val_loss_epoch, val_metrics["MSE"])

    # Run scan on outer loop (epochs)
    optimiz_state = optimizer.init(params_optimiz)
    (_, optimiz_state, params_optimiz), (train_loss_ts, train_MSE_ts, val_loss_ts, val_MSE_ts) = jax.lax.scan(
        epoch_step,
        (key, optimiz_state, params_optimiz),
        xs=None,
        length=n_iter
    )
    results = {
        "params_optimiz": params_optimiz,
        "train_loss_ts": train_loss_ts,
        "train_MSE_ts": train_MSE_ts,
        "val_loss_ts": val_loss_ts,
        "val_MSE_ts": val_MSE_ts,
    }
    return results