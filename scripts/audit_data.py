"""Read-only audit of the scMAMA-MIA datasets: counts, gene IDs, metadata, split disjointness."""
import glob
import os

import anndata as ad
import numpy as np

ROOT = os.path.expanduser("~/data/scMAMAMIA")

for name in ["ok", "aida", "cg"]:
    path = f"{ROOT}/{name}/full_dataset_cleaned.h5ad"
    print(f"\n===== {name} =====")
    a = ad.read_h5ad(path, backed="r")
    print("shape", a.shape, "| X type", type(a.X).__name__, "| raw present", a.raw is not None)
    print("layers", list(a.layers.keys()))
    print("obs", list(a.obs.columns))
    print("var", list(a.var.columns))
    print("var_names head", list(a.var_names[:4]))
    print("frac ENSG var_names", float(np.mean([str(v).startswith("ENSG") for v in a.var_names])))

    idx = np.sort(np.random.default_rng(0).choice(a.n_obs, 200, replace=False))
    x = a[idx].X
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    nz = x[x > 0]
    print("sample: max", float(x.max()), "| nonzero frac", float((x > 0).mean()),
          "| all integer", bool(np.allclose(nz, np.round(nz))),
          "| median total per cell", float(np.median(x.sum(1))))

    split_root = f"{ROOT}/{name}/splits"
    nds = sorted(os.path.basename(p) for p in glob.glob(f"{split_root}/*d"))
    print("split donor counts", nds)
    bad = checked = 0
    for trial_dir in glob.glob(f"{split_root}/*d/*"):
        try:
            s = [set(np.load(f"{trial_dir}/{k}.npy", allow_pickle=True).tolist())
                 for k in ("train", "holdout", "auxiliary")]
        except FileNotFoundError:
            continue
        checked += 1
        bad += bool((s[0] & s[1]) | (s[0] & s[2]) | (s[1] & s[2]))
    print(f"splits checked {checked}, with overlap {bad}")
    a.file.close()
