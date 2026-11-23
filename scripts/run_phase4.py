"""Phase 4: learning-rate check, then the donor-count grid for Geneformer V1 (objective O1).

Order:
  1. LR check at 50 donors, trials 1-2, so LoRA and freezing share learning rates
  2. Grid over donor counts and trials, one full trial at a time, so a partial run stays usable

The per-run null is expected to wander across this many runs, so it only stops the queue
beyond 4 null SDs. Smaller deviations are analyzed afterward from the recorded rows.
Per-cell losses are saved locally to `scores/` for per-cell-type and subgroup analysis.
Finished runs are skipped, so the script can be restarted safely.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, f"{REPO}/src"]
import phase1_pilot as p1  # noqa: E402
import run_sweep as rs  # noqa: E402
from lora import LoRALinear  # noqa: E402

DATASETS = ["ok", "aida"]
DONOR_COUNTS = [10, 20, 100, 200, 50]
TRIALS = [1, 2, 3, 4, 5]
FREEZE = ["head", "last2", "full"]
RANKS = [8, 64]
LR_TRIALS = [1, 2]
LR_ND = 50
LR_GRID = {"freeze": [3e-4, 1e-3], "lora": [1e-4, 3e-4]}
NULL_STOP_SDS = 4
SMOKE = False
_attack_rows = p1.attack_rows


def capture_scores(meta, cells, base_loss, loss, ctype, attacks):
    """Save per-cell losses, then compute attack rows as usual."""
    if not SMOKE:
        os.makedirs(f"{REPO}/scores", exist_ok=True)
        rank = "" if np.isnan(meta["rank"]) else f"_r{int(meta['rank'])}"
        ids = np.array(sorted(loss), dtype=np.int64)
        np.savez_compressed(
            f"{REPO}/scores/{meta['dataset']}_{meta['nd']}d_t{meta['trial']}_{meta['config']}{rank}.npz",
            cell=ids, base=np.array([base_loss[int(c)] for c in ids], dtype=np.float32),
            loss=np.array([loss[int(c)] for c in ids], dtype=np.float32))
    return _attack_rows(meta, cells, base_loss, loss, ctype, attacks)


def save_adapters_only(model, name, config):
    """Keep LoRA adapters. Full checkpoints are skipped to protect the disk."""
    if config == "lora":
        os.makedirs(f"{REPO}/checkpoints", exist_ok=True)
        adapter = {n: {"A": m.lora_A.detach().cpu(), "B": m.lora_B.detach().cpu()}
                   for n, m in model.named_modules() if isinstance(m, LoRALinear)}
        torch.save(adapter, f"{REPO}/checkpoints/{name}.pt")


p1.attack_rows = capture_scores
rs.save_artifacts = save_adapters_only


def lr_label(config, lr):
    return f"{config}_lr{lr:g}"


def lr_jobs():
    jobs = [(c, None, lr_label(c, lr), lr) for c in FREEZE for lr in LR_GRID["freeze"]]
    jobs += [("lora", r, lr_label("lora", lr), lr) for r in RANKS for lr in LR_GRID["lora"]]
    return jobs


def grid_jobs():
    jobs = [("base", None, "base", None)]
    jobs += [(c, None, c, None) for c in FREEZE]
    jobs += [("lora", r, "lora", None) for r in RANKS]
    return jobs


def run_cohort(data, nd, trial, jobs, csv, smoke, with_null):
    pending = [j for j in jobs if not rs.done(csv, data.name, nd, trial, j[2], j[1])]
    null_pending = with_null and not rs.done(csv, data.name, nd, trial, "null_unrelated_cohort", None)
    positive_pending = nd == 10 and not rs.done(csv, data.name, 10, trial, "full_positive", None)
    if not (pending or null_pending or positive_pending):
        print(f"skip cohort {data.name} {nd}d trial {trial}", flush=True)
        return

    c = rs.Cohort(data, nd, trial, smoke)
    base_loss = p1.cell_losses(rs.fresh(), c.seqs, c.eval_ids, data.tok)

    if positive_pending:
        rows = rs.run_config(data, c, base_loss, "full", None, csv, smoke, label="full_positive", lr=5e-4,
                             epochs=30, patience=None)
        auc = next(r["auc"] for r in rows if r["threat_model"] == "seen_cells"
                   and r["attack"] == "calibrated_loss" and r["aggregation"] == "mean")
        if auc < 0.9 and not smoke:
            sys.exit(f"POSITIVE CONTROL FAILED: {data.name} trial {trial}, AUC {auc:.3f}")

    if null_pending:
        rows = rs.run_config(data, c, base_loss, "full", None, csv, smoke, fit_ids=c.null_fit,
                             label="null_unrelated_cohort", epochs=rs.NULL_EPOCHS, patience=None)
        sd = p1.null_sd(len(c.cells["fit"]), len(c.cells["holdout"]))
        aucs = {r["threat_model"]: r["auc"] for r in rows
                if r["attack"] == "calibrated_loss" and r["aggregation"] == "mean"}
        print(f"CALIBRATED NULL {data.name} {nd}d t{trial}: {aucs} | null SD {sd:.3f}", flush=True)
        if not smoke and any(abs(a - 0.5) > NULL_STOP_SDS * sd for a in aucs.values()):
            sys.exit(f"CALIBRATED NULL BEYOND {NULL_STOP_SDS} SD: {data.name} {nd}d t{trial}: {aucs}")

    for config, rank, label, lr in pending:
        rs.run_config(data, c, base_loss, config, rank, csv, smoke, label=label, lr=lr)
    del c
    torch.cuda.empty_cache()


def plan():
    total = 0
    for name in DATASETS:
        for nd in DONOR_COUNTS:
            sizes = []
            for trial in TRIALS:
                s = p1.load_split(name, nd, trial)
                assert not (s["validation"] & (s["train"] | s["holdout"]))
                sizes.append((len(s["train"]), len(s["holdout"]), len(s["validation"])))
                total += len(grid_jobs()) + 1
            print(f"{name} {nd}d (train, holdout, clean validation) by trial: {sizes}")
    print(f"grid runs before skipping: {total} | LR-check runs: {len(DATASETS) * len(LR_TRIALS) * len(lr_jobs())}")


def main():
    global SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.plan:
        plan()
        return

    SMOKE = args.smoke
    csv = f"{REPO}/logs/smoke_phase4.csv" if SMOKE else f"{REPO}/results/metrics.csv"
    if SMOKE and os.path.exists(csv):
        os.remove(csv)
    torch.manual_seed(rs.SEED)
    if not SMOKE:
        json.dump({"datasets": DATASETS, "donor_counts": DONOR_COUNTS, "trials": TRIALS, "freeze": FREEZE,
                   "ranks": RANKS, "default_lr": rs.LR, "lr_check": {"nd": LR_ND, "trials": LR_TRIALS, "grid": LR_GRID},
                   "max_epochs": rs.MAX_EPOCHS, "patience": rs.PATIENCE, "null_stop_sds": NULL_STOP_SDS,
                   "seed": rs.SEED, "model_revision": rs.MODEL_REVISION},
                  open(f"{REPO}/results/phase4_meta.json", "w"), indent=2)

    loaded = {name: rs.Data(name) for name in (["aida"] if SMOKE else DATASETS)}
    if SMOKE:
        data = loaded["aida"]
        run_cohort(data, 50, 1, [lr_jobs()[0], lr_jobs()[-1]], csv, True, with_null=False)
        run_cohort(data, 10, 1, grid_jobs()[:2], csv, True, with_null=True)
        print("PHASE 4 SMOKE PASSED", flush=True)
        return

    print("=== LR check ===", flush=True)
    for trial in LR_TRIALS:
        for name, data in loaded.items():
            run_cohort(data, LR_ND, trial, lr_jobs(), csv, False, with_null=False)

    print("=== Donor-count grid ===", flush=True)
    for trial in TRIALS:
        for nd in DONOR_COUNTS:
            for name, data in loaded.items():
                run_cohort(data, nd, trial, grid_jobs(), csv, False, with_null=True)
        print(f"TRIAL {trial} COMPLETE", flush=True)
    print("PHASE 4 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
