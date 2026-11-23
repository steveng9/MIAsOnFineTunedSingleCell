"""Debug the failed calibrated null (AIDA, 50 donors, trial 4).

A. Are all recorded 50-donor nulls jointly consistent with the exact chance distribution?
B. Does the deviation survive retraining with other seeds (split property vs training noise)?
C. Is the donor-level null score tied to donor covariates, and are they imbalanced in this split?

Only aggregate statistics are written. No donor identifiers leave this process.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, f"{REPO}/src"]
import phase1_pilot as p1  # noqa: E402
import run_sweep as rs  # noqa: E402

DATASET, ND, TRIAL = "aida", 50, 4
SEEDS = [0, 1, 2]
THREATS = {"seen_cells": "fit", "same_donor_unseen_cells": "unseen"}
out = {}

# A. Pooled calibration of the null.
d = pd.read_csv(f"{REPO}/results/metrics.csv")
nulls = d[(d.config == "null_unrelated_cohort") & (d.nd == ND) & (d.attack == "calibrated_loss")
          & (d.aggregation == "mean")]
sd = p1.null_sd(ND, ND)
out["pooled"] = {"null_sd": sd}
for threat in THREATS:
    a = nulls[nulls.threat_model == threat].sort_values(["dataset", "trial"])
    z = (a.auc.to_numpy() - 0.5) / sd
    chi2 = float((z ** 2).sum())
    out["pooled"][threat] = {
        "n_cohorts": len(z), "aucs": [round(float(x), 4) for x in a.auc], "mean_auc": float(a.auc.mean()),
        "empirical_sd": float(a.auc.std(ddof=1)), "max_abs_z": float(np.abs(z).max()),
        "chi2": chi2, "chi2_p": float(stats.chi2.sf(chi2, len(z))),
        "mean_test_p": float(2 * stats.norm.sf(abs(z.mean()) * np.sqrt(len(z)))),
        "p_any_beyond_2sd_by_chance": float(1 - (1 - 2 * stats.norm.sf(2)) ** len(z))}
print(json.dumps(out["pooled"], indent=2), flush=True)

# B. Retrain the same null with different seeds.
data = rs.Data(DATASET)
c = rs.Cohort(data, ND, TRIAL, False)
base_loss = p1.cell_losses(rs.fresh(), c.seqs, c.eval_ids, data.tok)
donor_tables, out["seeds"] = {}, {}
for seed in SEEDS:
    model = rs.fresh()
    rs.apply_config(model, "full")
    p1.train(model, c.seqs, c.null_fit, p1.flat(c.cells["validation"]), data.tok, lr=1e-4,
             epochs=rs.NULL_EPOCHS, patience=None, seed=seed)
    loss = p1.cell_losses(model, c.seqs, c.eval_ids, data.tok)
    score = {x: base_loss[x] - loss[x] for x in loss}
    row = {}
    for threat, key in THREATS.items():
        auc, _, _, _ = p1.evaluate(c.cells[key], c.cells["holdout"], score, c.ctype, "mean")
        row[threat] = float(auc)
    out["seeds"][str(seed)] = row
    groups = {"member": p1.donor_scores(c.cells["unseen"], score, c.ctype, "mean"),
              "holdout": p1.donor_scores(c.cells["holdout"], score, c.ctype, "mean")}
    donor_tables[seed] = pd.DataFrame(
        [{"donor": k, "member": g == "member", "score": v} for g, m in groups.items() for k, v in m.items()]
    ).set_index("donor")
    print(f"seed {seed}: {row}", flush=True)
    del model
    torch.cuda.empty_cache()

corr = pd.concat({s: t.score for s, t in donor_tables.items()}, axis=1).corr(method="spearman")
out["donor_score_spearman_across_seeds"] = {
    f"{a}_{b}": float(corr.loc[a, b]) for i, a in enumerate(SEEDS) for b in SEEDS[i + 1:]}
print("across-seed donor score Spearman", out["donor_score_spearman_across_seeds"], flush=True)

# C. Donor covariates: association with the score and imbalance between groups.
t = donor_tables[0].copy()
t["score"] = pd.concat({s: x.score for s, x in donor_tables.items()}, axis=1).mean(axis=1)
obs = data.adata.obs
info = obs[obs["individual"].astype(str).isin(t.index)].copy()
info["individual"] = info["individual"].astype(str)
for col in ("Country", "sex", "ethnicity"):
    t[col] = info.groupby("individual", observed=True)[col].agg(lambda v: str(v.iloc[0]))
t["mean_umis"] = info.groupby("individual", observed=True)["nCount_RNA"].mean()
target_ids = p1.flat(c.cells["unseen"]) + p1.flat(c.cells["holdout"])
t["mean_tokens"] = pd.Series({
    k: float(np.mean([len(c.seqs[int(x)]) for x in v]))
    for m in (c.cells["unseen"], c.cells["holdout"]) for k, v in m.items()})

out["covariates"] = {}
for col in ("Country", "sex", "ethnicity"):
    groups = [g.score.to_numpy() for _, g in t.groupby(col) if len(g) > 1]
    table = pd.crosstab(t[col], t.member)
    out["covariates"][col] = {
        "score_kruskal_p": float(stats.kruskal(*groups).pvalue) if len(groups) > 1 else None,
        "score_r2": float(1 - ((t.score - t.groupby(col).score.transform("mean")) ** 2).sum()
                          / ((t.score - t.score.mean()) ** 2).sum()),
        "membership_imbalance_p": float(stats.chi2_contingency(table)[1]),
        "group_counts": {str(k): [int(x) for x in v] for k, v in zip(table.index, table.to_numpy())}}
for col in ("mean_umis", "mean_tokens"):
    rho, p = stats.spearmanr(t[col], t.score)
    out["covariates"][col] = {"score_spearman": float(rho), "score_p": float(p),
                              "membership_auc": float(roc_auc_score(t.member, t[col]))}
print(json.dumps(out["covariates"], indent=2), flush=True)

json.dump(out, open(f"{REPO}/results/null_diagnostic_aida_50d_t4.json", "w"), indent=2)
print("DIAGNOSTIC COMPLETE", flush=True)
