"""Phase 5 re-analysis from saved per-cell scores (CPU only, no training).

Problem: with k cells per type, a member's "seen" cells are balanced across cell types, so rare
types are over-represented relative to holdout donors, whose cells follow the natural mix. Rare
types are harder to classify, so members look worse to any classifier. The reference-only null
fell to 0.03-0.2 instead of 0.5.

Fix: "type_centered" aggregation. Each cell's score has the mean score of its cell type (over all
scored cells in that run) subtracted before the donor mean. Under the null, score depends on cell
type but not membership, so the composition difference cancels. The reference-only null checks it.

Cohorts and supports are rebuilt with the same seeds as run_phase5.py and checked against the
logged support sizes. Outputs (aggregate statistics only): results/phase5_typecentered.csv,
results/phase5_summary.csv.
"""
import glob
import os
import re
import sys

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, f"{REPO}/src"]
import phase1_pilot as p1  # noqa: E402

SEED, ND, N_PERM = 0, 50, 500
KS = [1, 5, 10, 50, "all"]
NAME = re.compile(r"^(ok|aida)_50d_t(\d+)_(.+)_k(\w+)$")


def rebuild(name, donors, ctypes, trial):
    split = p1.load_split(name, ND, trial)
    cells = p1.select_cells(donors, split, np.random.default_rng([SEED, ND, trial]), p1.CELL_CAP, p1.VAL_CELL_CAP)
    fit_all = p1.flat(cells["fit"])
    classes = sorted({ctypes[x] for x in fit_all})
    cid = {t: i for i, t in enumerate(classes)}
    usable = [x for x in fit_all if ctypes[x] in cid]
    rng = np.random.default_rng([SEED, trial, 5])
    by_type = {}
    for x in usable:
        by_type.setdefault(cid[ctypes[x]], []).append(x)
    supports = {}
    for k in KS:
        supports[str(k)] = set(usable) if k == "all" else \
            {int(v) for t, xs in by_type.items() for v in rng.permutation(xs)[:k]}
    return cells, cid, supports


def maps(cells, cid, ctypes, support):
    ok = lambda x: ctypes[int(x)] in cid  # noqa: E731
    donor_of = {int(x): d for d, v in cells["fit"].items() for x in v}
    members = sorted({donor_of[x] for x in support})
    seen = {d: [int(x) for x in cells["fit"][d] if int(x) in support] for d in members}
    unseen = {d: [int(x) for x in list(cells["fit"][d]) + list(cells["unseen"].get(d, []))
                  if int(x) not in support and ok(x)] for d in members}
    hold = {d: [int(x) for x in v if ok(x)] for d, v in cells["holdout"].items()}
    return seen, unseen, hold


def donor_auc(mmap, hmap, score, perm_rng):
    mmap = {d: v for d, v in mmap.items() if len(v)}
    hmap = {d: v for d, v in hmap.items() if len(v)}
    s = np.array([score[v].mean() for v in mmap.values()] + [score[v].mean() for v in hmap.values()])
    y = np.r_[np.ones(len(mmap)), np.zeros(len(hmap))]
    auc = roc_auc_score(y, s)
    perm = [roc_auc_score(perm_rng.permutation(y), s) for _ in range(N_PERM)]
    return auc, float(np.std(perm)), len(mmap)


def main():
    logged = pd.read_csv(f"{REPO}/results/metrics_phase5.csv")
    rows = []
    for name in ("ok", "aida"):
        obs = ad.read_h5ad(os.path.expanduser(f"~/data/scMAMAMIA/{name}/full_dataset_cleaned.h5ad"), backed="r").obs
        donors, ctypes = obs["individual"].astype(str).to_numpy(), obs["cell_type"].astype(str).to_numpy()
        for trial in (1, 2, 3):
            cells, cid, supports = rebuild(name, donors, ctypes, trial)
            for k, sup in supports.items():
                want = logged[(logged.dataset == name) & (logged.trial == trial) & (logged.k.astype(str) == k)]
                assert (want.n_support == len(sup)).all(), f"support rebuild mismatch {name} t{trial} k{k}"
            for f in sorted(glob.glob(f"{REPO}/scores/phase5/{name}_50d_t{trial}_*.npz")):
                _, _, method, k = NAME.match(os.path.basename(f)[:-4]).groups()
                z = np.load(f)
                n = int(z["cell"].max()) + 1
                ct = pd.Series(ctypes[z["cell"]])
                seen, unseen, hold = maps(cells, cid, ctypes, supports[k])
                for attack, raw in (("raw_loss", -z["loss"]), ("calibrated_loss", z["ref"] - z["loss"]),
                                    ("reference_null", -z["ref"])):
                    for how in ("mean", "type_centered"):
                        v = raw - ct.groupby(ct).transform(lambda s: raw[s.index].mean()).to_numpy() \
                            if how == "type_centered" else raw
                        score = np.full(n, np.nan)
                        score[z["cell"]] = v
                        for threat, mmap in (("seen_cells", seen), ("same_donor_unseen_cells", unseen)):
                            if sum(len(x) > 0 for x in mmap.values()) < 2:
                                continue
                            auc, sd, nm = donor_auc(mmap, hold, score, np.random.default_rng([SEED, trial, 7]))
                            rows.append({"dataset": name, "trial": trial, "method": method, "k": k,
                                         "threat_model": threat, "attack": attack, "aggregation": how,
                                         "auc": auc, "perm_sd": sd, "n_member_donors": nm})
            print(f"{name} t{trial} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"{REPO}/results/phase5_typecentered.csv", index=False)
    util = logged[logged.aggregation == "mean"].groupby(["dataset", "method", "k"]).agg(
        macro_f1=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"), accuracy=("accuracy", "mean"),
        n_trainable=("n_trainable", "first"), n_support=("n_support", "mean")).reset_index()
    util["k"] = util.k.astype(str)
    summ = out.groupby(["dataset", "method", "k", "threat_model", "attack", "aggregation"]).agg(
        auc_mean=("auc", "mean"), auc_sd=("auc", "std"), perm_sd_mean=("perm_sd", "mean"),
        n_member_donors=("n_member_donors", "mean"), n_trials=("auc", "size")).reset_index()
    summ = summ.merge(util, on=["dataset", "method", "k"], how="left")
    summ.to_csv(f"{REPO}/results/phase5_summary.csv", index=False)
    print("PHASE 5 ANALYSIS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
