"""Read-only follow-up audit: locate split overlaps and check .raw counts."""
import glob
import itertools
import os

import anndata as ad
import numpy as np

ROOT = os.path.expanduser("~/data/scMAMAMIA")
KEYS = ("train", "holdout", "auxiliary")

for name in ["ok", "aida", "cg"]:
    print(f"\n===== {name}: split overlaps =====")
    a = ad.read_h5ad(f"{ROOT}/{name}/full_dataset_cleaned.h5ad", backed="r")
    all_donors = set(a.obs["individual"].astype(str).unique())
    print("donors in h5ad", len(all_donors), "| cell types", a.obs["cell_type"].nunique())

    for trial_dir in sorted(glob.glob(f"{ROOT}/{name}/splits/*d/*")):
        present = [k for k in KEYS if os.path.exists(f"{trial_dir}/{k}.npy")]
        if len(present) < 3:
            print(os.path.relpath(trial_dir, f"{ROOT}/{name}/splits"), "missing", set(KEYS) - set(present))
            continue
        s = {k: set(map(str, np.load(f"{trial_dir}/{k}.npy", allow_pickle=True).tolist())) for k in KEYS}
        overlaps = {f"{x}&{y}": len(s[x] & s[y]) for x, y in itertools.combinations(KEYS, 2) if s[x] & s[y]}
        unknown = sum(len(s[k] - all_donors) for k in KEYS)
        if overlaps or unknown:
            sizes = {k: len(v) for k, v in s.items()}
            print(os.path.relpath(trial_dir, f"{ROOT}/{name}/splits"), "sizes", sizes,
                  "overlaps", overlaps, "ids_not_in_h5ad", unknown)

    print(f"--- {name}: counts check ---")
    idx = np.sort(np.random.default_rng(0).choice(a.n_obs, 200, replace=False))
    if a.raw is not None:
        x = a.raw.X[idx]  # backed .raw: index the on-disk matrix directly
        x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        nz = x[x > 0]
        print("raw shape", a.raw.shape, "| max", float(x.max()), "| all integer",
              bool(np.allclose(nz, np.round(nz))), "| median total", float(np.median(x.sum(1))),
              "| raw var_names ENSG frac",
              float(np.mean([str(v).startswith("ENSG") for v in a.raw.var_names])))
    else:
        print("no .raw; X is the count source")
    a.file.close()
