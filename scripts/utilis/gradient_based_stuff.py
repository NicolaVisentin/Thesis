# Imports
from typing import Dict
from functools import partial
import jax
from jax import numpy as jnp
from jax import Array


# Inverse softplus function
def InverseSoftplus(x):
    """
    Inverse softplus function.
    """
    return jnp.log(jnp.exp(x)-1)


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