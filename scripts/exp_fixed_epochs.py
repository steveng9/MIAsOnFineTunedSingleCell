"""Fixed-epoch comparison of Geneformer and scGPT (no early stopping).

Question: does scGPT leak less than Geneformer because it resists memorization, or because early
stopping ended its runs after 4-17 epochs while Geneformer often trained 30?
Both models train full, last2, and LoRA r8 for exactly 30 epochs at their default learning rates
(1e-4; LoRA 1e-3), on the same 50-donor cohorts (trials 1-3, OneK1K and AIDA). Nulls and base
losses are those of the original runs (same cohorts and seeds; the base is recomputed here).
Rows: results/metrics_fixed_epochs_geneformer.csv and results/metrics_fixed_epochs_scgpt.csv.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, f"{REPO}/src"]
import phase1_pilot as p1  # noqa: E402
import run_phase4 as r4  # noqa: E402  (enables per-cell score saving for Geneformer)
import run_scgpt as r6  # noqa: E402
import run_sweep as rs  # noqa: E402

EPOCHS, ND, TRIALS = 30, 50, [1, 2, 3]
CONFIGS = [("full", None), ("last2", None), ("lora", 8)]
CSV_GF = f"{REPO}/results/metrics_fixed_epochs_geneformer.csv"
CSV_SC = f"{REPO}/results/metrics_fixed_epochs_scgpt.csv"
LABEL = "{}_fixed30"


def geneformer(name, trial, data):
    todo = [(c, r) for c, r in CONFIGS if not rs.done(CSV_GF, name, ND, trial, LABEL.format(c), r)]
    if not todo:
        return
    c = rs.Cohort(data, ND, trial, False)
    base = p1.cell_losses(rs.fresh(), c.seqs, c.eval_ids, data.tok)
    for config, rank in todo:
        rs.run_config(data, c, base, config, rank, CSV_GF, False, label=LABEL.format(config),
                      epochs=EPOCHS, patience=None)
    del c
    torch.cuda.empty_cache()


def scgpt(name, trial, data):
    todo = [(c, r) for c, r in CONFIGS if not r6.done(CSV_SC, name, ND, trial, LABEL.format(c), r)]
    if not todo:
        return
    c, base = r6.cohort_base(data, ND, trial, False)
    for config, rank in todo:
        r6.run_config(data, c, base, config, rank, CSV_SC, False, label=LABEL.format(config),
                      epochs=EPOCHS, patience=None)
    del c
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for model in ("scgpt", "geneformer"):
        for name in ("ok", "aida"):
            data = r6.ScData(name) if model == "scgpt" else rs.Data(name)
            for trial in TRIALS:
                (scgpt if model == "scgpt" else geneformer)(name, trial, data)
            del data
    print("FIXED EPOCHS COMPLETE", flush=True)
