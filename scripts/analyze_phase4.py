"""Phase 4 analysis from saved per-cell scores (CPU only, no training).

1. Batch-stratified AUC for every saved run. AIDA strata are donor country. OneK1K strata are the
   barcode suffix, which looks like a sequencing pool (inferred, not documented in the data).
   Only member-holdout donor pairs within the same stratum are compared. The null spread comes
   from permuting membership within each stratum.
2. Summary over trials, pooled and stratified.
3. Learning-rate check: best auxiliary validation loss per (config, lr). Never chosen on holdout.

Outputs (aggregate statistics only): results/phase4_stratified.csv, results/phase4_summary.csv,
results/phase4_lr_check.csv.
"""
import glob
import json
import os
import re
import sys

import anndata as ad
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, f"{REPO}/src"]
import phase1_pilot as p1  # noqa: E402

SEED, N_PERM = 0, 500
THREATS = {"seen_cells": "fit", "same_donor_unseen_cells": "unseen"}
NAME = re.compile(r"^(ok|aida)_(\d+)d_t(\d+)_(.+?)(?:_r(\d+))?$")


def pair_counts(pos, neg):
    neg = np.sort(neg)
    lo, hi = np.searchsorted(neg, pos, "left"), np.searchsorted(neg, pos, "right")
    return float((lo + 0.5 * (hi - lo)).sum()), len(pos) * len(neg)


def stratified_auc(score, member, strata):
    wins = total = 0.0
    for s in np.unique(strata):
        w, t = pair_counts(score[(strata == s) & member], score[(strata == s) & ~member])
        wins, total = wins + w, total + t
    return wins / total if total else np.nan, total


def permuted_sd(score, member, strata, rng):
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    out = []
    for _ in range(N_PERM):
        shuffled = member.copy()
        for g in groups:
            shuffled[g] = rng.permutation(member[g])
        out.append(stratified_auc(score, shuffled, strata)[0])
    return float(np.mean(out)), float(np.std(out))


def donor_strata(name):
    a = ad.read_h5ad(os.path.expanduser(f"~/data/scMAMAMIA/{name}/full_dataset_cleaned.h5ad"), backed="r")
    obs = a.obs
    donors = obs["individual"].astype(str).to_numpy()
    if name == "aida":
        label = obs["Country"].astype(str).to_numpy()
    else:
        label = obs["barcode_col"].astype(str).str.rsplit("-", n=1).str[1].to_numpy()
    strata = pd.DataFrame({"d": donors, "s": label}).groupby("d")["s"].agg(lambda v: v.value_counts().index[0])
    return donors, strata


def stratified_rows():
    rows, cache = [], {}
    for name in ("ok", "aida"):
        files = sorted(glob.glob(f"{REPO}/scores/{name}_*.npz"))
        if not files:
            continue
        donors, strata = donor_strata(name)
        for f in files:
            m = NAME.match(os.path.basename(f)[:-4])
            nd, trial, config, rank = int(m[2]), int(m[3]), m[4], m[5]
            if config == "base":
                continue
            key = (name, nd, trial)
            if key not in cache:
                split = p1.load_split(name, nd, trial)
                cache[key] = p1.select_cells(donors, split, np.random.default_rng([SEED, nd, trial]),
                                             p1.CELL_CAP, p1.VAL_CELL_CAP)
            cells = cache[key]
            z = np.load(f)
            score = dict(zip(z["cell"].tolist(), (z["base"] - z["loss"]).tolist()))
            hold = {d: np.mean([score[int(c)] for c in ids]) for d, ids in cells["holdout"].items()}
            for threat, k in THREATS.items():
                mem = {d: np.mean([score[int(c)] for c in ids]) for d, ids in cells[k].items()}
                names = list(mem) + list(hold)
                s = np.array(list(mem.values()) + list(hold.values()))
                member = np.r_[np.ones(len(mem), bool), np.zeros(len(hold), bool)]
                st = strata.loc[names].to_numpy()
                pooled = stratified_auc(s, member, np.zeros(len(s), int))[0]
                strat, pairs = stratified_auc(s, member, st)
                pm, psd = permuted_sd(s, member, st, np.random.default_rng([SEED, nd, trial, len(rows)]))
                rows.append({"dataset": name, "nd": nd, "trial": trial, "config": config,
                             "rank": float(rank) if rank else np.nan, "threat_model": threat,
                             "pooled_auc": pooled, "stratified_auc": strat, "n_pairs": int(pairs),
                             "n_strata": len(np.unique(st)), "perm_mean": pm, "perm_sd": psd,
                             "z": (strat - pm) / psd if psd > 0 else np.nan})
            print(f"{os.path.basename(f)}: " + ", ".join(f"{r['threat_model']} {r['pooled_auc']:.3f}->{r['stratified_auc']:.3f}"
                                                         for r in rows[-2:]), flush=True)
    return pd.DataFrame(rows)


def lr_rows():
    metrics = pd.read_csv(f"{REPO}/results/metrics.csv")
    sel = metrics[(metrics.nd == 50) & metrics.trial.isin([1, 2]) & (metrics.objective == "O1")
                  & (metrics.attack == "calibrated_loss") & (metrics.aggregation == "mean")]
    rows = []
    for (name, trial, config, rank), g in sel.groupby(["dataset", "trial", "config", "rank"], dropna=False):
        family = config.split("_lr")[0]
        if family not in ("head", "last2", "full", "lora"):
            continue
        lr = float(config.split("_lr")[1]) if "_lr" in config else (1e-3 if family == "lora" else 1e-4)
        tag = f"{name}_50d_t{trial}_{config}" + (f"_r{int(rank)}" if not np.isnan(rank) else "")
        path = f"{REPO}/logs/history_{tag}.json"
        hist = json.load(open(path)) if os.path.exists(path) else []
        auc = g.set_index("threat_model")["auc"]
        rows.append({"dataset": name, "trial": trial, "family": family, "rank": rank, "lr": lr,
                     "best_val_loss": min((h["val_loss"] for h in hist), default=np.nan),
                     "epochs_run": len(hist), "holdout_ce": g["utility"].iloc[0],
                     "seen_auc": auc.get("seen_cells"), "unseen_auc": auc.get("same_donor_unseen_cells")})
    return pd.DataFrame(rows)


def main():
    lr = lr_rows()
    if len(lr):
        lr.to_csv(f"{REPO}/results/phase4_lr_check.csv", index=False)
        print(lr.groupby(["dataset", "family", "rank", "lr"], dropna=False)[
            ["best_val_loss", "holdout_ce", "seen_auc", "unseen_auc"]].mean().round(4).to_string())

    t = stratified_rows()
    t.to_csv(f"{REPO}/results/phase4_stratified.csv", index=False)
    summary = t.groupby(["dataset", "nd", "config", "rank", "threat_model"], dropna=False).agg(
        n_trials=("trial", "nunique"), pooled_mean=("pooled_auc", "mean"), pooled_sd=("pooled_auc", "std"),
        stratified_mean=("stratified_auc", "mean"), stratified_sd=("stratified_auc", "std"),
        perm_sd_mean=("perm_sd", "mean"), pairs_mean=("n_pairs", "mean")).reset_index()
    summary.to_csv(f"{REPO}/results/phase4_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
    print("PHASE 4 ANALYSIS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
