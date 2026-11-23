"""Does comparing donors within country remove the calibrated-null deviation in AIDA?

The stratified AUC counts only member-vs-holdout donor pairs from the same country.
Its null spread comes from permuting membership labels within each country.
Only aggregate statistics are written.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, f"{REPO}/src"]
import phase1_pilot as p1  # noqa: E402
import run_sweep as rs  # noqa: E402

DATASET, ND, TRIALS, SEED, N_PERM = "aida", 50, [1, 2, 3, 4], 0, 2000
THREATS = {"seen_cells": "fit", "same_donor_unseen_cells": "unseen"}


def pair_counts(pos, neg):
    """Concordant pairs (ties count one half) and total pairs."""
    neg = np.sort(neg)
    wins = np.searchsorted(neg, pos, "left") + 0.5 * (np.searchsorted(neg, pos, "right") - np.searchsorted(neg, pos, "left"))
    return float(wins.sum()), len(pos) * len(neg)


def stratified_auc(score, member, strata):
    wins = total = 0.0
    for s in np.unique(strata):
        where = strata == s
        w, t = pair_counts(score[where & member], score[where & ~member])
        wins, total = wins + w, total + t
    return wins / total


def permuted(score, member, strata, rng):
    shuffled = member.copy()
    for s in np.unique(strata):
        where = np.flatnonzero(strata == s)
        shuffled[where] = rng.permutation(member[where])
    return stratified_auc(score, shuffled, strata)


data = rs.Data(DATASET)
obs = data.adata.obs
country = obs.assign(individual=obs["individual"].astype(str)).groupby("individual", observed=True)["Country"].agg(
    lambda v: str(v.iloc[0]))

rows = []
for trial in TRIALS:
    c = rs.Cohort(data, ND, trial, False)
    base_loss = p1.cell_losses(rs.fresh(), c.seqs, c.eval_ids, data.tok)
    model = rs.fresh()
    rs.apply_config(model, "full")
    p1.train(model, c.seqs, c.null_fit, p1.flat(c.cells["validation"]), data.tok, lr=1e-4,
             epochs=rs.NULL_EPOCHS, patience=None, seed=SEED)
    loss = p1.cell_losses(model, c.seqs, c.eval_ids, data.tok)
    cell_score = {x: base_loss[x] - loss[x] for x in loss}
    for threat, key in THREATS.items():
        m = p1.donor_scores(c.cells[key], cell_score, c.ctype, "mean")
        h = p1.donor_scores(c.cells["holdout"], cell_score, c.ctype, "mean")
        names = list(m) + list(h)
        score = np.array(list(m.values()) + list(h.values()))
        member = np.r_[np.ones(len(m), bool), np.zeros(len(h), bool)]
        strata = country.loc[names].to_numpy()
        pooled = stratified_auc(score, member, np.zeros(len(score), int))
        strat = stratified_auc(score, member, strata)
        rng = np.random.default_rng([SEED, trial])
        perms = np.array([permuted(score, member, strata, rng) for _ in range(N_PERM)])
        rows.append({"trial": trial, "threat_model": threat, "pooled_auc": pooled, "within_country_auc": strat,
                     "perm_mean": float(perms.mean()), "perm_sd": float(perms.std()),
                     "z": float((strat - perms.mean()) / perms.std()),
                     "perm_p": float((1 + (np.abs(perms - perms.mean()) >= abs(strat - perms.mean())).sum()) / (N_PERM + 1))})
        print(rows[-1], flush=True)
    del model, c
    torch.cuda.empty_cache()

table = pd.DataFrame(rows)
summary = table.groupby("threat_model").agg(
    pooled_mean=("pooled_auc", "mean"), pooled_sd=("pooled_auc", "std"),
    within_mean=("within_country_auc", "mean"), within_sd=("within_country_auc", "std"),
    max_abs_z=("z", lambda z: float(np.abs(z).max())))
print(table.round(4).to_string(index=False))
print(summary.round(4).to_string())
json.dump({"rows": rows, "summary": summary.reset_index().to_dict("records")},
          open(f"{REPO}/results/null_diagnostic_within_country.json", "w"), indent=2)
print("STRATIFIED DIAGNOSTIC COMPLETE", flush=True)
