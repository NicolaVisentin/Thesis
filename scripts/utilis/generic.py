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
from scipy.integrate import odeint


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


# Build Lorenz96 dataset
def get_lorenz(
        dim : int,
        F : float,
        num_batch : int = 128,
        lag : int = 25,
        washout : int = 200,
        window_size : int = 0,
        dt : float = 0.01,
        duration : float = 20,
        seed = 0,
):
    """
    Builds a batch of temporal sequences from the Lorenz96 equations.
 
    Each trajectory is obtained by integrating the Lorenz96 ODE from a random
    initial condition drawn uniformly in [F-0.5, F+0.5]^dim. The time axis
    runs with time step dt from t=0 to t=duration [s], extended by ``lag`` and 
    ``washout`` extra steps.
 
    Args
    ----
    dim : int
        Dimension of the Lorenz96 system (number of coupled variables).
    F : float
        Constant external forcing applied to every variable.
    num_batch : int, optional
        Number of independent trajectories to simulate (default: 128).
    lag : int, optional
        Number of prediction lag steps for the Lorenz96 task (default: 25).
    washout : int, optional
        Number of washout steps for the Lorenz96 task (default: 200).
    window_size : int, optional
        Length (in time steps) of each sliding input window. When > 0 the
        function extracts all non-overlapping windows and returns
        ``(windows, targets)``; when 0 the raw trajectory array is returned
        instead (default: 0).
    dt : float, optional
        Integration step for building the sequences (default: 0.01 s).
    duration : float, optional
        Duration of the time sequence (without lag and washout) in seconds (default: 20 s).
    seed : int, optional
        Seed for sampling initial conditions (default: 0).
 
    Returns
    -------
    dataset : jnp.ndarray
        Returned when ``window_size == 0``. Batch of temporal sequences from 
        t = 0 to t = duration + lag*dt + washout*dt. Shape (num_batch, num_timesteps, dim)
    windows : jnp.ndarray
        Returned when ``window_size > 0``. Sliding input windows concatenated across all
        trajectories. Shape (B_total, window_size, dim)
    targets : jnp.ndarray
        Returned when ``window_size > 0``. Target state for each window (i.e. the state
        ``lag`` steps after the last element of the window). Shape (B_total, dim)
    """
    # https://en.wikipedia.org/wiki/Lorenz_96_model
    def L96(x, t):
        """Lorenz96 model with constant forcing"""
        d = np.zeros(dim)
        for i in range(dim):
            d[i] = (x[(i + 1) % dim] - x[i - 2]) * x[i - 1] - x[i] + F
        return d

    t = np.arange(0.0, duration + (lag * dt) + (washout * dt), dt)

    rng = np.random.default_rng(seed)
    dataset = []
    for i in range(num_batch):
        x0 = rng.random(dim) + F - 0.5  # [F-0.5, F+0.5]
        x = odeint(L96, x0, t)
        dataset.append(x)

    # (num_batch, num_timesteps, dim) — kept as JAX array
    dataset = jnp.array(np.stack(dataset, axis=0), dtype=jnp.float32)

    if window_size > 0:
        all_windows, all_targets = [], []
        for i in range(dataset.shape[0]):
            w, trg = get_fixed_length_windows(dataset[i], window_size, prediction_lag=lag)
            all_windows.append(w)
            all_targets.append(trg)

        windows = jnp.concatenate(all_windows, axis=0) # (B_total, window_size, dim)
        targets = jnp.concatenate(all_targets, axis=0) # (B_total, dim)
        return windows, targets
    else:
        return dataset


def get_fixed_length_windows(tensor, length, prediction_lag=1):
    """
    tensor : (T,) or (T, I)
    returns: windows (B, L, I), targets (B, I)
    """
    assert tensor.ndim <= 2
    if tensor.ndim == 1:
        tensor = tensor[:, None]   # (T, 1)  — JAX equivalent of unsqueeze(-1)

    T, I = tensor.shape

    # Build sliding windows via jax.lax.dynamic_slice or plain slicing.
    # Number of windows = T - prediction_lag - length + 1
    num_windows = T - prediction_lag - length + 1

    # Stack windows: shape (B, L, I)
    windows = jnp.stack(
        [tensor[i : i + length] for i in range(num_windows)],
        axis=0,
    )  # (B, L, I) — already in the right order (no permute needed)

    # Targets: one step per window, offset by prediction_lag
    targets = tensor[length + prediction_lag - 1 : length + prediction_lag - 1 + num_windows]
    # shape: (B, I)

    return windows, targets  # (B, L, I), (B, I)


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
