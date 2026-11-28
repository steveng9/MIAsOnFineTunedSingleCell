"""HFRA (human fetal retina atlas, local name `cg`): does the Geneformer picture hold outside blood?

HFRA has 22 donors, too few for the scMAMA-MIA split sizes, so each trial draws a random split:
8 member donors, 8 holdout donors, 6 auxiliary donors (validation and the unrelated-donor null).
Splits are written to data/splits/cg/ (gitignored; never committed). Configs: base, head, last2,
full, LoRA r8 at the default learning rates, early stopping, plus the null. 3 trials.
With 8 vs 8 donors the null SD of one AUC is about 0.15, so this is a coarse check.
Rows: results/metrics_hfra.csv; per-cell scores in scores/ (cg_*).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, f"{REPO}/src"]
import phase1_pilot as p1  # noqa: E402
import run_phase4 as r4  # noqa: E402
import run_sweep as rs  # noqa: E402

ND, N_HOLD, TRIALS = 8, 8, [1, 2, 3]
SPLIT_ROOT = f"{REPO}/data/splits/cg"
CSV = f"{REPO}/results/metrics_hfra.csv"
JOBS = [("base", None, "base", None), ("head", None, "head", None), ("last2", None, "last2", None),
        ("full", None, "full", None), ("lora", 8, "lora", None)]


def make_splits(donors):
    ids = sorted(set(donors))
    for trial in TRIALS:
        root = f"{SPLIT_ROOT}/{ND}d/{trial}"
        if os.path.exists(f"{root}/train.npy"):
            continue
        os.makedirs(root, exist_ok=True)
        perm = np.random.default_rng([rs.SEED, 22, trial]).permutation(ids)
        np.save(f"{root}/train.npy", perm[:ND])
        np.save(f"{root}/holdout.npy", perm[ND:ND + N_HOLD])
        np.save(f"{root}/auxiliary.npy", perm[ND + N_HOLD:])


_orig_load_split = p1.load_split


def load_split(dataset, nd, trial):
    if dataset != "cg":
        return _orig_load_split(dataset, nd, trial)
    root = f"{SPLIT_ROOT}/{nd}d/{trial}"
    s = {k: set(map(str, np.load(f"{root}/{k}.npy", allow_pickle=True).tolist()))
         for k in ("train", "holdout", "auxiliary")}
    assert not s["train"] & s["holdout"] and not s["auxiliary"] & (s["train"] | s["holdout"])
    s["validation"] = set(s["auxiliary"])
    return s


p1.load_split = load_split

if __name__ == "__main__":
    data = rs.Data("cg")
    make_splits(data.donors)
    for trial in TRIALS:
        r4.run_cohort(data, ND, trial, JOBS, CSV, False, with_null=True)
    print("HFRA COMPLETE", flush=True)
