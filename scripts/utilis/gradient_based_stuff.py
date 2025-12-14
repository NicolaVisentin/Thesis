# Imports
from typing import Dict, Tuple, Callable
from functools import partial
import jax
from jax import numpy as jnp
from jax import Array
import optax


# =====================================================
# Generic
# =====================================================

# Inverse softplus function
def InverseSoftplus(x):
    """
    Inverse softplus function.
    """
    return jnp.where(x<50, jnp.log(jnp.exp(x)-1), x)

# Initialize a square matrix with singular values strictly greater than a threshold
def init_A_svd(
        key, n: int, 
        s_min: float=0.0, 
        s_max: float=1.0, 
        log_space_sampling: bool=False
) -> Array:
    """
    Initialize an n-by-n matrix A such that its singular values s_i are all in a given
    range: s_min <= s_i <= s_max for i = 1, ..., n.
    
    Args
    ----
    key : jax.random.PRNGKey
    n : int
        Matrix dimension.
    s_min : float
        Optional threshold such that s_i >= s_min, with s=diag(S), A = U*S*V.T (default: 0.0).
    s_max : float
        Optional threshold such that s_i <= s_max, with s=diag(S), A = U*S*V.T (default: 1.0).
    log_space_sampling : bool
        If True, samples [s_min, s_max] in the log space (default: False). In this case s_min
        must be > 0.
    Returns
    -------
    A : Array
        Randomly generated matrix.
    """
    if log_space_sampling and s_min == 0:
        raise ValueError("If log_space_sampling, then s_min must be > 0!")
    
    key, keyU, keyV, keyS = jax.random.split(key, 4)
    
    # Sample orthogonal matrices
    Gu = jax.random.normal(keyU, (n,n))
    Gv = jax.random.normal(keyV, (n,n))
    U, RU = jnp.linalg.qr(Gu)
    U = U * jnp.sign(jnp.diag(RU))
    V, RV = jnp.linalg.qr(Gv)
    V = V * jnp.sign(jnp.diag(RV))

    # Sample singular values > thresh. Samples from [s_min, s_max] 
    if log_space_sampling:
        s_min = jnp.log10(s_min)
        s_max = jnp.log10(s_max)
    s = jax.random.uniform(
        key=keyS, 
        shape=(n,), 
        minval=s_min, 
        maxval=s_max
    )
    if log_space_sampling:
        s = 10 ** s
        
    return U @ jnp.diag(s) @ V.T


# =====================================================
# Dataset handling
# =====================================================

# Split dataset in train/validation/test
def split_dataset(
        key, 
        dataset : Dict,
        train_ratio : float = 0.7, 
        test_ratio : float = 0.2,
    ) -> tuple[Dict, Dict, Dict]:
    """
    Split datapoints and labels into train, validation, and test sets.

    Args
    ----
    key 
        jax.random.key.
    dataset : dict
        Dataset provided as a dictionary with all elements (m datapoints and m labels). Each key has shape (m, ...),
        where m in the size of the dataset.
    train_ratio : float
        Train set ratio between 0 and 1 (default: 0.7). train_ratio + val_ratio + test_ratio = 1.
    test_ratio : float
        Test set ratio between 0 and 1 (default: 0.2). train_ratio + val_ratio + test_ratio = 1.
    
    Returns
    -------
    train_set, val_set, test_set : dict
        Dictionaries for train set, validation set and test set, with the same keys of the original dataset.
    """
    # dataset size
    first_key = next(iter(dataset.keys()))
    m = dataset[first_key].shape[0]

    # shuffle indices
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, m)

    # compute split indices
    train_end = int(train_ratio * m)
    test_end = m - int(test_ratio * m)

    # create splits for each item in dataset
    train_set = {k: v[perm[:train_end]] for k, v in dataset.items()}
    val_set   = {k: v[perm[train_end:test_end]] for k, v in dataset.items()}
    test_set  = {k: v[perm[test_end:]] for k, v in dataset.items()}

    return train_set, val_set, test_set


# Shuffled batch generator
@partial(jax.jit, static_argnums=(1,2))
def batch_indx_generator(
    key,
    dataset_size : int, 
    batch_size : int = 512,
) -> Array:
    """
    Given a certain dataset_size and a certain desired batch_size, his function
    returns the indices to generate a shuffled collection of batches out of a dataset. 

    Args
    ----
    key
        jax.random.key.
    dataset_size : int
        Dataset size.
    batch_size : int
        Batch size (default: 512).
    
    Returns
    -------
    batch_ids : Array
        Array containing indices to extract shuffled batches out of a dataset. If dataset_size is 
        not divisible by batch_size, then the last batch is filled with repeated data. Shape (n_batches, batch_size)

    Example
    -------
    dataset_size=10, batch_size=3 -> batch_ids=[[0,7,3],[1,9,4],[2,6,8],[5,0,7]]
    """
    # Shuffle dataset indices
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, dataset_size)

    # Number of batches (ceil division)
    num_batches = (dataset_size + batch_size - 1) // batch_size

    # Pad permutation to make reshape legal: extend last batch if incomplete
    pad_size = num_batches * batch_size - dataset_size
    perm = jnp.pad(perm, (0, pad_size), mode="wrap")

    # Reshape into (num_batches, batch_size)
    batch_ids = perm.reshape((num_batches, batch_size))

    return batch_ids


# Batch extractor
@jax.jit
def extract_batch(
    dataset : Dict,
    indices : Array,
) -> Dict:
    """
    Extracts a batch from a dataset according to provided indices.

    Args
    ----
    dataset : dict
        Dataset provided as a dictionary with all elements (m datapoints and m labels). Each key has shape (m, ...),
        where m in the size of the dataset.
    indices : Array
        Array containing indices of the elements to extract. Shape (batch_size,)
        
    Returns
    -------
    batch : tuple
        Batch provided as a dictionary with all elements (batch_size datapoints and batch_size labels). Each key has shape (batch_size, ...).
    """
    batch = {key: value[indices] for key, value in dataset.items()}
    return batch


# =====================================================
# Training
# =====================================================

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