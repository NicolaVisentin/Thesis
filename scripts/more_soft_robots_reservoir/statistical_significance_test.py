import numpy as np
from scipy.stats import mannwhitneyu

"""
Mann-Whitney U significance tests for the benchmarking experiments: our approach
vs baselines ("unoptimized", "partially pretrained").
All raw per-seed values are hard-coded below (up to 4 runs per configuration).

Handling of problematic runs:
- NaN run  (None below)  -> excluded; the test simply uses the remaining runs.
- Diverged run ("div:X")  -> kept in the Mann-Whitney test (rank-based, so it
    only counts as the worst rank), but excluded from descriptive mean/std.

Metric direction:
- sMNIST      -> Accuracy, higher is better.
- Mackey-Glass / Lorenz96 -> NRMSE, lower is better.

Test: one-sided Mann-Whitney U ("our approach is better"), exact method (small samples).
"""

# Raw data (hard coded) and setup
data = {
    "sMNIST": {
        "lower_is_better": False, # classification accuracy: higher is better
        6:  {"ours": [0.6424, 0.6534, 0.6218, 0.6434],
            "unoptimized": [0.3712, 0.5376, 0.3976, 0.4248],
            "partially": [0.6082, 0.5962, 0.5900, 0.6052]},
        9:  {"ours": [0.6798, 0.6444, 0.6626, 0.6844],
            "unoptimized": [0.3666, 0.5308, 0.4522, 0.3330],
            "partially": [0.6068, 0.6314, 0.6222, 0.6140]},
        12: {"ours": [0.6304, 0.6734, 0.6572, 0.6862],
            "unoptimized": [0.4718, 0.5996, 0.5042, 0.5296],
            "partially": [0.5994, 0.6426, 0.6506, 0.6148]},
        15: {"ours": [0.7196, 0.7140, 0.7138, 0.7044],
            "unoptimized": [0.3714, 0.6318, 0.3660, 0.3940],
            "partially": [0.7284, 0.5410, 0.4690, 0.6860]},
    },
    "ADIAC": {
            "lower_is_better": False, # classification accuracy: higher is better
            6:  {"ours": [0.4595, 0.4357, 0.4405, 0.4476],
                "unoptimized": [0.3476, 0.2714, 0.3095, 0.2929],
                "partially": [0.3048, 0.4381, 0.3881, 0.4429]},
            9:  {"ours": [0.4548, 0.5357, 0.5190, 0.5310],
                "unoptimized": [0.4238, 0.3976, 0.3714, 0.3500],
                "partially": [0.1643, 0.4929, 0.3190, 0.5071]},
            12: {"ours": [0.5024, 0.5024, 0.5095, 0.4929],
                "unoptimized": [0.4905, 0.3381, 0.4357, 0.3476],
                "partially": [0.2381, 0.2190, 0.4429, 0.4167]},
            15: {"ours": [0.5024, 0.5619, 0.5333, 0.5262],
                "unoptimized": [0.4857, 0.2119, 0.6000, 0.5190],
                "partially": [0.4167, 0.2381, 0.2143, 0.4214]},
        },
    "Mackey-Glass": {
        "lower_is_better": True, # NRMSE: lower is better
        6:  {"ours": [0.529971, 0.523017, 0.523637, 0.524097],
            "unoptimized": [0.713295, 0.896077, 0.563369, 0.696531],
            "partially": [0.795480, 0.523017, 0.538880, 0.532117]},
        9:  {"ours": [0.498168, 0.500019, 0.529598, 0.492952],
            "unoptimized": [0.728679, 0.839122, 0.720081, 0.689492],
            "partially": [0.509839, 0.562238, 0.613869, 0.570043]},
        12: {"ours": [0.428501, 0.456062, 0.429164, 0.427103],
            "unoptimized": [0.712862, 0.883061, 0.556907, 0.635217],
            "partially": [None, 0.621812, 0.599603, 0.639629]},
        15: {"ours": [0.421557, 0.433578, 0.425380, 0.420137],
            "unoptimized": [0.711437, 0.812779, 0.717735, 0.678939],
            "partially": [0.497475, 0.481226, "div:301681", 0.476295]},
    },
    "Lorenz96": {
        "lower_is_better": True, # NRMSE, lower is better
        6:  {"ours": [0.569331, 0.565004, 0.566337, 0.566348],
            "unoptimized": [0.672494, 0.847201, 0.602625, 0.675680],
            "partially": [0.606512, 0.813911, 0.574410, 0.652679]},
        9:  {"ours": [0.522059, 0.521936, 0.521430, 0.522149],
            "unoptimized": [0.955513, 0.589739, 0.595031, 0.599063],
            "partially": [0.530333, 0.730225, 0.526154, 0.735342]},
        12: {"ours": [0.464620, 0.476204, 0.461021, 0.449093],
            "unoptimized": [0.539597, "div:6.303236515258028e+90", 0.546038, 0.470961],
            "partially": [0.519680, 0.500026, 0.479805, 0.524575]},
        15: {"ours": [0.451392, 0.473310, 0.455775, 0.443129],
            "unoptimized": [0.872217, 0.471598, 0.850005, 0.524543],
            "partially": [0.507516, 0.466502, 0.479136, 0.844571]},
    },
}

TASKS = ["sMNIST", "ADIAC", "Mackey-Glass", "Lorenz96"]
NYS = [6, 9, 12, 15]
BASELINES = ["unoptimized", "partially"]

def parse_runs(raw):
    """Return (valid, test):
    valid = numeric, non-diverged, non-NaN (for mean/std);
    test  = numeric incl. diverged, excl. NaN (for the U test)."""
    valid, test = [], []
    for x in raw:
        if x is None: # NaN run: dropped everywhere
            continue
        if isinstance(x, str) and x.startswith("div:"):
            test.append(float(x.split("div:")[1]))
        else:
            valid.append(float(x)); test.append(float(x))
    return valid, test

# Print means and std devs for each case
print("=" * 82)
print("DESCRIPTIVE  (mean +/- std over VALID runs; NaN & diverged excluded)")
print("=" * 82)
hdr = f"{'Task':<14}{'ny':>3}   {'ours':>17}{'unoptimized':>19}{'partially':>19}"
print(hdr); print("-" * len(hdr))
for task in TASKS:
    for ny in NYS:
        cells = []
        for m in ["ours", "unoptimized", "partially"]:
            valid, _ = parse_runs(data[task][ny][m])
            mean = np.mean(valid)
            std = np.std(valid, ddof=1) if len(valid) > 1 else 0.0
            tag = "" if len(valid) == 4 else f"[n{len(valid)}]"
            cells.append(f"{mean:.4f}+-{std:.4f}{tag}")
        print(f"{task:<14}{ny:>3}   {cells[0]:>17}{cells[1]:>19}{cells[2]:>19}")

# Mann-Whitney test
print("\n" + "=" * 82)
print("MANN-WHITNEY U   ours vs baseline   (one-sided: 'ours is better', exact)")
print("=" * 82)
results = []
for task in TASKS:
    lower = data[task]["lower_is_better"]
    for ny in NYS:
        _, ours = parse_runs(data[task][ny]["ours"])
        for b in BASELINES:
            _, base = parse_runs(data[task][ny][b])
            x = np.array(ours) * (-1 if lower else 1)
            y = np.array(base) * (-1 if lower else 1)
            U, p = mannwhitneyu(x, y, alternative="greater", method="exact")
            results.append(dict(task=task, ny=ny, base=b,
                                no=len(x), nb=len(y), U=U, p=p))
m = len(results)
for r in results:
    r["pbonf"] = min(1.0, r["p"] * m)

print(f"{'Task':<14}{'ny':>3}{'baseline':>13}{'n_o':>4}{'n_b':>4}"
    f"{'U':>6}{'p':>9}{'p_bonf':>9}{'sig':>5}")
print("-" * 74)
for r in results:
    sig = "*" if r["p"] < 0.05 else ""
    print(f"{r['task']:<14}{r['ny']:>3}{r['base']:>13}{r['no']:>4}{r['nb']:>4}"
        f"{r['U']:>6.1f}{r['p']:>9.4f}{r['pbonf']:>9.4f}{sig:>5}")

ns = sum(1 for r in results if r["p"] < 0.05)
print(f"\nSignificant (uncorrected, p<0.05): {ns}/{m}")
print(f"Significant (Bonferroni):          "
    f"{sum(1 for r in results if r['pbonf'] < 0.05)}/{m}")
print("Floor: 4v4 -> min one-sided p = 1/C(8,4) = 0.0143 ; "
    "4v3 -> 1/C(7,4) = 0.0286.")
