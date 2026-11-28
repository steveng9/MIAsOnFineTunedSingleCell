"""Summaries for the 2026-11-28 follow-up experiments (CPU). Each part runs only if its inputs exist.

1. paired_null      Phase 4: config AUC minus the same trial's null AUC (pooled and within-stratum),
                    mean and standard error over trials. Nulls are overdispersed, so this pairing is
                    a fairer test than comparing with 0.5.
2. label_only       Phase 5 rerun: label-only attack (was the prediction correct?), raw and
                    calibrated (target correct - reference correct), type-centered donor means.
3. lira             exp_lira.py: mean over trials.
4. fixed_epochs     exp_fixed_epochs.py: both models, 30 epochs, vs their early-stopped runs.
5. scgpt_v2         exp_scgpt_v2.py: pretraining-matched binning vs Phase 6.
6. hfra             exp_hfra.py.
7. lr_matched       head/LoRA at 3e-4 across donor counts vs the grid's default rates.
Outputs: results/followup_*.csv.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, f"{REPO}/src"]
R = f"{REPO}/results"


def mean_sd(df, keys, col="auc"):
    return df.groupby(keys)[col].agg(auc_mean="mean", auc_sd="std", n_trials="size").reset_index()


def metric_rows(path, attack="calibrated_loss", agg="mean"):
    d = pd.read_csv(path)
    return d[(d.attack == attack) & (d.aggregation == agg)]


def paired_null():
    s = pd.read_csv(f"{R}/phase4_stratified.csv")
    null = s[s.config == "null_unrelated_cohort"].set_index(["dataset", "nd", "trial", "threat_model"])
    rows = []
    for _, r in s[s.config != "null_unrelated_cohort"].iterrows():
        k = (r.dataset, r.nd, r.trial, r.threat_model)
        if k not in null.index:
            continue
        n = null.loc[k]
        rows.append({**r[["dataset", "nd", "trial", "config", "rank", "threat_model"]].to_dict(),
                     "pooled_minus_null": r.pooled_auc - n.pooled_auc,
                     "stratified_minus_null": r.stratified_auc - n.stratified_auc})
    d = pd.DataFrame(rows)
    out = d.groupby(["dataset", "nd", "config", "rank", "threat_model"], dropna=False).agg(
        pooled_diff=("pooled_minus_null", "mean"), pooled_se=("pooled_minus_null", "sem"),
        stratified_diff=("stratified_minus_null", "mean"), stratified_se=("stratified_minus_null", "sem"),
        n_trials=("pooled_minus_null", "size")).reset_index()
    out.to_csv(f"{R}/followup_paired_null.csv", index=False)
    return out


def label_only():
    files = sorted(glob.glob(f"{REPO}/scores/phase5b/*.npz"))
    if not files:
        return None
    import anndata as ad
    import analyze_phase5 as a5
    rows = []
    for name in ("ok", "aida"):
        obs = ad.read_h5ad(os.path.expanduser(f"~/data/scMAMAMIA/{name}/full_dataset_cleaned.h5ad"), backed="r").obs
        donors, ctypes = obs["individual"].astype(str).to_numpy(), obs["cell_type"].astype(str).to_numpy()
        for trial in (1, 2, 3):
            cells, cid, supports = a5.rebuild(name, donors, ctypes, trial)
            for f in sorted(glob.glob(f"{REPO}/scores/phase5b/{name}_50d_t{trial}_*.npz")):
                _, _, method, k = a5.NAME.match(os.path.basename(f)[:-4]).groups()
                z = np.load(f)
                if "correct" not in z:
                    continue
                n = int(z["cell"].max()) + 1
                ct = pd.Series(ctypes[z["cell"]])
                seen, unseen, hold = a5.maps(cells, cid, ctypes, supports[k])
                for attack, raw in (("label_only", z["correct"].astype(float)),
                                    ("label_only_calibrated", z["correct"].astype(float) - z["ref_correct"]),
                                    ("label_only_reference_null", z["ref_correct"].astype(float)),
                                    ("loss_calibrated", z["ref"] - z["loss"])):
                    v = raw - ct.groupby(ct).transform(lambda s_: raw[s_.index].mean()).to_numpy()
                    score = np.full(n, np.nan)
                    score[z["cell"]] = v
                    for threat, mmap in (("seen_cells", seen), ("same_donor_unseen_cells", unseen)):
                        if sum(len(x) > 0 for x in mmap.values()) < 2:
                            continue
                        auc, sd, nm = a5.donor_auc(mmap, hold, score, np.random.default_rng([0, trial, 7]))
                        rows.append({"dataset": name, "trial": trial, "method": method, "k": k,
                                     "threat_model": threat, "attack": attack, "auc": auc, "perm_sd": sd,
                                     "n_member_donors": nm})
    d = pd.DataFrame(rows)
    d.to_csv(f"{R}/followup_label_only_runs.csv", index=False)
    out = mean_sd(d, ["dataset", "method", "k", "threat_model", "attack"])
    out.to_csv(f"{R}/followup_label_only.csv", index=False)
    return out


def lira():
    if not os.path.exists(f"{R}/lira.csv"):
        return None
    d = pd.read_csv(f"{R}/lira.csv")
    out = d.groupby(["dataset", "target", "threat_model", "attack"]).agg(
        pooled_mean=("pooled_auc", "mean"), pooled_sd=("pooled_auc", "std"),
        stratified_mean=("stratified_auc", "mean"), stratified_sd=("stratified_auc", "std"),
        pairs=("n_pairs", "mean"), n_trials=("trial", "size")).reset_index()
    out.to_csv(f"{R}/followup_lira.csv", index=False)
    return out


def fixed_epochs():
    parts = []
    for model, new, old in (("geneformer", "metrics_fixed_epochs_geneformer.csv", "metrics.csv"),
                            ("scgpt", "metrics_fixed_epochs_scgpt.csv", "metrics_scgpt.csv")):
        if not os.path.exists(f"{R}/{new}"):
            continue
        a = metric_rows(f"{R}/{new}")
        a = a.assign(config=a.config.str.replace("_fixed30", ""), regime="fixed_30_epochs")
        b = metric_rows(f"{R}/{old}")
        b = b[(b.nd == 50) & b.trial.isin([1, 2, 3]) & b.config.isin(["full", "last2", "lora", "null_unrelated_cohort"])]
        b = b[(b.config != "lora") | (b["rank"] == 8)].assign(regime="early_stopping")
        parts.append(pd.concat([a, b]).assign(model=model))
    if not parts:
        return None
    d = pd.concat(parts)
    out = d.groupby(["model", "dataset", "config", "regime", "threat_model"]).agg(
        auc_mean=("auc", "mean"), auc_sd=("auc", "std"), utility=("utility", "mean"),
        n_trials=("trial", "nunique")).reset_index()
    out.to_csv(f"{R}/followup_fixed_epochs.csv", index=False)
    return out


def scgpt_v2():
    if not os.path.exists(f"{R}/metrics_scgpt_v2.csv"):
        return None
    parts = []
    for tag, f in (("v2_ties_left", "metrics_scgpt_v2.csv"), ("v1_ties_random", "metrics_scgpt.csv")):
        d = metric_rows(f"{R}/{f}")
        base = pd.read_csv(f"{R}/{f}")
        base = base[(base.config == "base") & (base.attack == "raw_loss") & (base.aggregation == "mean")]
        d = pd.concat([d, base]).assign(pipeline=tag)
        d = d[d.config.isin(["base", "head", "last2", "full", "lora", "null_unrelated_cohort", "full_positive"])]
        d = d[(d.config != "lora") | d["rank"].isin([8, 64])]
        parts.append(d)
    d = pd.concat(parts)
    d["config"] = d.config + d["rank"].fillna(0).astype(int).astype(str).replace("0", "")
    out = d.groupby(["pipeline", "dataset", "nd", "config", "threat_model"]).agg(
        auc_mean=("auc", "mean"), auc_sd=("auc", "std"), utility=("utility", "mean"),
        n_trials=("trial", "nunique")).reset_index()
    out.to_csv(f"{R}/followup_scgpt_v2.csv", index=False)
    return out


def hfra():
    if not os.path.exists(f"{R}/metrics_hfra.csv"):
        return None
    d = pd.read_csv(f"{R}/metrics_hfra.csv")
    d = d[(d.aggregation == "mean")]
    d["config"] = d.config + d["rank"].fillna(0).astype(int).astype(str).replace("0", "")
    out = d.groupby(["config", "threat_model", "attack"]).agg(
        auc_mean=("auc", "mean"), auc_sd=("auc", "std"), utility=("utility", "mean"),
        n_trials=("trial", "nunique")).reset_index()
    out.to_csv(f"{R}/followup_hfra.csv", index=False)
    return out


def lr_matched():
    d = metric_rows(f"{R}/metrics.csv")
    d = d[d.trial.isin([1, 2, 3]) & d.config.isin(["head", "lora", "head_lr0.0003", "lora_lr0.0003", "last2", "full"])]
    d = d[(~d.config.str.startswith("lora")) | d["rank"].isin([8, 64])]
    if not (d.config == "head_lr0.0003").any():
        return None
    d["config"] = d.config + d["rank"].fillna(0).astype(int).astype(str).replace("0", "")
    out = d.groupby(["dataset", "nd", "config", "threat_model"]).agg(
        auc_mean=("auc", "mean"), auc_sd=("auc", "std"), utility=("utility", "mean"),
        n_trials=("trial", "nunique"), n_trainable=("n_trainable", "first")).reset_index()
    out.to_csv(f"{R}/followup_lr_matched.csv", index=False)
    return out


if __name__ == "__main__":
    for f in (paired_null, label_only, lira, fixed_epochs, scgpt_v2, hfra, lr_matched):
        out = f()
        print(f"{f.__name__}: {'skipped (no inputs yet)' if out is None else f'{len(out)} rows'}", flush=True)
