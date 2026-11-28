"""Offline LiRA-style shadow-model attack at 20 donors (Geneformer, objective O1).

The calibrated attack compares a target model's loss with the base model's loss. LiRA instead
compares it with models trained the same way on other donors ("OUT" shadows): for each cell, the
shadows give the loss it gets when its donor was not trained on. Score per cell:
  shadow_mean  mean_OUT_loss - target_loss
  lira         (mean_OUT_loss - target_loss) / sd, with per-cell sd shrunk toward the global sd
               (4 shadows are too few for per-cell variances alone)
Scores are averaged per donor and compared with the calibrated attack, pooled and within strata.

Shadows: 4 per (dataset, trial, config), each fine-tuned on 20 donors drawn from donors outside the
cohort (not train, holdout, or auxiliary for this split), up to 200 cells each, with the target's
config, learning rate, and early stopping. Targets are the existing Phase 4 runs at 20 donors
(per-cell losses in scores/), configs head, last2, full; the unrelated-donor null model is attacked
with full-config shadows as a null.
Outputs: scores/lira/*.npz (shadow losses, local), results/lira.csv.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, f"{REPO}/src"]
import analyze_phase4 as a4  # noqa: E402
import phase1_pilot as p1  # noqa: E402
import run_sweep as rs  # noqa: E402
from lora import apply_config  # noqa: E402

ND, TRIALS, N_SHADOW, SHRINK = 20, [1, 2, 3], 4, 4
CONFIGS = ["head", "last2", "full"]
OUT_DIR = f"{REPO}/scores/lira"
CSV = f"{REPO}/results/lira.csv"


def shadow_pool(data, split, cohort):
    used = split["train"] | split["holdout"] | split["auxiliary"]
    counts = pd.Series(data.donors).value_counts()
    return sorted(d for d in counts.index if d not in used and counts[d] >= 50)


def train_shadows(data, c, trial, config):
    path = f"{OUT_DIR}/{data.name}_{ND}d_t{trial}_{config}.npz"
    if os.path.exists(path):
        return dict(np.load(path))
    pool = shadow_pool(data, c.split, c)
    eval_ids = c.eval_ids
    losses = []
    for s in range(N_SHADOW):
        rng = np.random.default_rng([rs.SEED, ND, trial, 500 + s])
        donors = rng.choice(pool, ND, replace=False)
        fit = []
        for d in donors:
            where = np.flatnonzero(data.donors == d)
            fit.extend(rng.permutation(where)[:p1.CELL_CAP].tolist())
        new = [x for x in fit if x not in c.seqs]
        c.seqs.update(data.tokenize(np.array(new)))
        model = rs.fresh()
        apply_config(model, config, None)
        lr = rs.LR["default"]
        print(f"shadow {data.name} t{trial} {config} #{s}: {len(fit)} cells", flush=True)
        p1.train(model, c.seqs, fit, p1.flat(c.cells["validation"]), data.tok, lr=lr,
                 epochs=rs.MAX_EPOCHS, patience=rs.PATIENCE, seed=rs.SEED + 1 + s)
        l = p1.cell_losses(model, c.seqs, eval_ids, data.tok)
        losses.append([l[x] for x in eval_ids])
        del model
        torch.cuda.empty_cache()
    out = {"cell": np.array(eval_ids, dtype=np.int64), "loss": np.array(losses, dtype=np.float32)}
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def donor_means(cells, score_by_cell):
    return {d: float(np.mean([score_by_cell[int(x)] for x in v])) for d, v in cells.items()}


def evaluate(name, trial, c, strata_of, target_label, shadows):
    t = np.load(f"{REPO}/scores/{name}_{ND}d_t{trial}_{target_label}.npz")
    tgt = dict(zip(t["cell"].tolist(), t["loss"].tolist()))
    base = dict(zip(t["cell"].tolist(), t["base"].tolist()))
    sl = dict(zip(shadows["cell"].tolist(), shadows["loss"].T))
    mu = {x: float(v.mean()) for x, v in sl.items()}
    sd = {x: float(v.std(ddof=1)) for x, v in sl.items()}
    g = float(np.sqrt(np.mean([v ** 2 for v in sd.values()])))
    k = N_SHADOW - 1
    sd_s = {x: np.sqrt((k * sd[x] ** 2 + SHRINK * g ** 2) / (k + SHRINK)) for x in sd}
    scores = {"calibrated_loss": {x: base[x] - tgt[x] for x in tgt},
              "shadow_mean": {x: mu[x] - tgt[x] for x in tgt},
              "lira": {x: (mu[x] - tgt[x]) / sd_s[x] for x in tgt}}
    rows = []
    for threat, key in (("seen_cells", "fit"), ("same_donor_unseen_cells", "unseen")):
        mem = {d: v for d, v in c.cells[key].items() if len(v)}
        hold = c.cells["holdout"]
        for attack, s in scores.items():
            dm, dh = donor_means(mem, s), donor_means(hold, s)
            vals = np.array(list(dm.values()) + list(dh.values()))
            member = np.r_[np.ones(len(dm), bool), np.zeros(len(dh), bool)]
            strata = np.array([strata_of.get(d, "NA") for d in list(dm) + list(dh)])
            pooled = a4.stratified_auc(vals, member, np.zeros(len(vals)))[0]
            strat, pairs = a4.stratified_auc(vals, member, strata)
            rows.append({"dataset": name, "nd": ND, "trial": trial, "target": target_label,
                         "threat_model": threat, "attack": attack, "pooled_auc": pooled,
                         "stratified_auc": strat, "n_pairs": pairs})
    return rows


def main():
    rows = []
    for name in ("ok", "aida"):
        data = rs.Data(name)
        strata_of = a4.donor_strata(name)[1].to_dict()
        for trial in TRIALS:
            c = rs.Cohort(data, ND, trial, False)
            for config in CONFIGS:
                sh = train_shadows(data, c, trial, config)
                rows += evaluate(name, trial, c, strata_of, config, sh)
                if config == "full":
                    rows += evaluate(name, trial, c, strata_of, "null_unrelated_cohort", sh)
                pd.DataFrame(rows).to_csv(CSV, index=False)
            del c
    print(pd.DataFrame(rows).groupby(["dataset", "target", "threat_model", "attack"])[
        ["pooled_auc", "stratified_auc"]].mean().round(3).to_string(), flush=True)
    print("LIRA COMPLETE", flush=True)


if __name__ == "__main__":
    main()
