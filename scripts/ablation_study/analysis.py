# =====================================================
# Setup
# =====================================================

# Choose device (cpu or gpu)
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Imports
import sys
from pathlib import Path
import numpy as onp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

curr_folder = Path(__file__).parent      # current folder
sys.path.append(str(curr_folder.parent)) # scripts folder
from utilis import *

# Jax settings
jax.config.update("jax_enable_x64", True)  # double precision
jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)
seed = 123
key = jax.random.key(seed)

# Folders
main_folder = curr_folder.parent.parent                                            # main folder "codes"
plots_folder = main_folder/'plots and videos'/curr_folder.stem/Path(__file__).stem # folder for plots and videos
dataset_folder = main_folder/'datasets'                                            # folder with the dataset
data_folder = main_folder/'saved data/ablation_study'                              # folder with saved data

plots_folder.mkdir(parents=True, exist_ok=True)


# =====================================================
# 0.0 REFERENCE CASE
# =====================================================

# Choose data
test_case = '0.0_reference' # choose test case
prefix = '1_oomA'           # choose prefix for data to load

# Show comparison of all different samplings (short training)
all_loss_curves = onp.load(data_folder/test_case/f'{prefix}_all_loss_curves.npz')
all_rmse_before = onp.load(data_folder/test_case/f'{prefix}_all_rmse_before.npz')
all_rmse_after = onp.load(data_folder/test_case/f'{prefix}_all_rmse_after.npz')

all_train_mse_ts = all_loss_curves["train_MSEs_ts"]
all_rmse_before = all_rmse_before["RMSE_before"]
all_rmse_after = all_rmse_after["RMSE_after"]
n_samples = all_rmse_before.shape[0]

(plots_folder/test_case).mkdir(parents=True, exist_ok=True)
plt.figure()
plt.plot(onp.arange(n_samples)+1, all_rmse_before, 'gx', label='test RMSE before')
plt.plot(onp.arange(n_samples)+1, all_rmse_after, 'go', label='test RMSE after')
plt.plot(onp.arange(n_samples)+1, onp.sqrt(all_train_mse_ts[:,-1]), 'ro', label='final train RMSE')
plt.yscale('log')
plt.grid(True)
plt.xlabel('sample n.')
plt.ylabel('RMSE')
plt.title(f'Results for various initial guesses')
plt.legend()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(plots_folder/test_case/'samples_comparison', bbox_inches='tight')
plt.show()

# Show BEST result among all samples (long training)
