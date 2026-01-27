# Imports
from typing import Dict, Tuple
from pathlib import Path
from functools import partial
import numpy as np
import gzip
import struct
import jax
from jax import numpy as jnp
from jax import Array


# =====================================================
# Dataset handling
# =====================================================

# Loads MNIST dataset as numpy arrays
def load_mnist_data(folder: Path) -> Tuple[dict, dict]:
    """
    Loads MNIST dataset as numpy arrays.
    
    Args
    ----
    folder : Path
        Path with the dataset raw files.

    Returns
    -------
    mnist_train_set : dict
        Dictionary with keys:
        - **"images"**: raw MNIST train images as numpy arrays of shape (n_train_images, 1, 28, 28). Grayscale images in [0, 1].
        - **"labels"**: labels.
    mnist_test_set : dict
        Dictionary with keys:
        - **"images"**: raw MNIST test images as numpy arrays of shape (n_test_images, 1, 28, 28). Grayscale images in [0, 1].
        - **"labels"**: labels.
    """
    # load train images and labels
    with gzip.open(folder/"train-images-idx3-ubyte.gz", 'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images_train = np.frombuffer(f.read(), dtype=np.uint8)
        images_train = images_train.reshape(num, 1, rows, cols)

    with gzip.open(folder/"train-labels-idx1-ubyte.gz", 'rb') as f:
        _, num = struct.unpack(">II", f.read(8))
        labels_train = np.frombuffer(f.read(), dtype=np.uint8)

    # load test images and labels
    with gzip.open(folder/"t10k-images-idx3-ubyte.gz", 'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images_test = np.frombuffer(f.read(), dtype=np.uint8)
        images_test = images_test.reshape(num, 1, rows, cols)

    with gzip.open(folder/"t10k-labels-idx1-ubyte.gz", 'rb') as f:
        _, num = struct.unpack(">II", f.read(8))
        labels_test = np.frombuffer(f.read(), dtype=np.uint8)

    # convert from [0, 255] to [0, 1]
    images_train = images_train.astype(np.float32) / 255.0
    images_test  = images_test.astype(np.float32) / 255.0

    mnist_train_set = {"images": images_train, "labels": labels_train}
    mnist_test_set = {"images": images_test, "labels": labels_test}
    return (mnist_train_set, mnist_test_set)


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


# Load Mackey-Glass dataset
def load_mackey_glass_data(csvfolder: Path, lag=84, washout=200, train_portion=0.5, val_portion=0.25):
    """Get the Mackey-Glass dataset and return the train, validation and test datasets
    as numpy arrays.

    Args:
        csvfolder (Path): Path to the directory containing the mackey_glass.csv file.
        lag (int, optional): Number of time steps to look back. Defaults to 84.
        washout (int, optional): Number of time steps to discard. Defaults to 200.

    Returns:
        Tuple[Tuple[Array, Array], Tuple[Array, Array], Tuple[Array, Array]]: Train, validation and test datasets.
    """
    with open(csvfolder/'mackey_glass.csv', "r") as f:
        data_lines = f.readlines()[0]

    # 10k steps
    dataset = np.array([float(el) for el in data_lines.split(",")], dtype=np.float64)

    end_train = int(dataset.shape[0] * train_portion)
    end_val = end_train + int(dataset.shape[0] * val_portion)
    end_test = dataset.shape[0]

    train_dataset = dataset[: end_train - lag]
    train_target = dataset[washout + lag : end_train]

    val_dataset = dataset[end_train : end_val - lag]
    val_target = dataset[end_train + washout + lag : end_val]

    test_dataset = dataset[end_val : end_test - lag]
    test_target = dataset[end_val + washout + lag : end_test]

    return (
        (train_dataset, train_target),
        (val_dataset, val_target),
        (test_dataset, test_target),
    )


# =====================================================
# Other
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
