"""Phase 6: repeat the core of Phases 1-3 with scGPT whole-human (objective O1, generative bins).

Same donor splits, cell selection, attacks, and controls as Geneformer (`run_sweep.Cohort`,
`phase1_pilot.attack_rows`). Only the model, tokenizer, and loss change:
  - input: nonzero genes binned into 51 levels, <cls> first, at most 1,200 tokens
  - loss: mean squared error on a random 25% of bins that are hidden and generated
  - eval: 3 fixed hidden-gene masks per cell, seeded by (MASK_SEED, cell ID, mask index)

Order: positive controls (10 donors) for both datasets, then 50 donors trial 1 with the
unrelated-donor null and every config, then trials 2-3 with five configs, each trial across both
datasets. A failed positive control drops that dataset. The null drops a dataset only beyond 4 SDs;
batch-stratified scoring happens afterward.
Finished runs are skipped, so the script can be restarted safely.
"""
import argparse
import json
import os
import platform
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import torch
import transformers

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, f"{REPO}/src"]
import phase1_pilot as p1  # noqa: E402
import run_sweep as rs  # noqa: E402
import scgpt_model as sm  # noqa: E402
from lora import LoRALinear  # noqa: E402

DEVICE = "cuda"
SEED, MASK_SEED, N_MASKS, GEN_FRAC = 0, 1234, 3, 0.25
BATCH, EVAL_BATCH = 8, 16
MAX_EPOCHS, PATIENCE, NULL_EPOCHS = 30, 3, 10
LR = {"lora": 1e-3, "default": 1e-4}
DATASETS = ["aida", "ok"]
ALL_CONFIGS = [("head", None), ("last1", None), ("last2", None), ("last4", None), ("full", None)] + \
              [("lora", r) for r in (1, 2, 4, 8, 16, 32, 64)]
CORE_CONFIGS = [("head", None), ("last2", None), ("full", None), ("lora", 8), ("lora", 64)]
TRIAL_PLAN = [(1, ALL_CONFIGS), (2, CORE_CONFIGS), (3, CORE_CONFIGS)]
NULL_STOP_SDS = 4
# 5e-4 diverged on OneK1K at epoch 20 (loss 121 -> 203, constant predictions); rows archived in
# results/metrics_scgpt_positive_lr5e-4.csv. 1e-4 matches the main full fine-tune.
POSITIVE_LR = 1e-4
CSV = f"{REPO}/results/metrics_scgpt.csv"


class ScData:
    """Same interface as run_sweep.Data, with the scGPT tokenizer."""

    def __init__(self, name):
        self.name = name
        self.adata = ad.read_h5ad(os.path.expanduser(f"~/data/scMAMAMIA/{name}/full_dataset_cleaned.h5ad"), backed="r")
        raw = self.adata.raw is not None
        self.matrix = self.adata.raw.X if raw else self.adata.X
        var = self.adata.raw.var if raw else self.adata.var
        names = var["feature_name"].to_numpy() if "feature_name" in var.columns else var.index.to_numpy()
        self.tok = sm.ScGPTTokenizer(names, seed=SEED)
        self.donors = self.adata.obs["individual"].astype(str).to_numpy()
        self.ctypes = self.adata.obs["cell_type"].astype(str).to_numpy()
        print(f"{name}: {len(self.tok.keep):,} genes in the scGPT vocabulary", flush=True)

    def tokenize(self, indices, chunk=256):
        indices = np.sort(np.unique(indices))
        seqs = {}
        for start in range(0, len(indices), chunk):
            part = indices[start:start + chunk]
            rows = self.matrix[part].toarray()
            for i, cell in enumerate(part):
                seqs[int(cell)] = self.tok.encode(rows[i], int(cell))
        assert all(len(g) > 1 for g, _ in seqs.values()), "cell with no usable genes"
        return seqs


def collate(items, pad_id):
    n = max(len(g) for g, _ in items)
    genes = torch.full((len(items), n), pad_id, dtype=torch.long)
    vals = torch.full((len(items), n), sm.PAD_VALUE)
    for i, (g, v) in enumerate(items):
        genes[i, :len(g)] = torch.as_tensor(g, dtype=torch.long)
        vals[i, :len(v)] = torch.as_tensor(v, dtype=torch.float32)
    return genes, vals, genes.eq(pad_id)


def fixed_masks(cells, ids, shape, m):
    gen = torch.zeros(shape, dtype=torch.bool)
    for i, c in enumerate(ids):
        length = len(cells[c][0])
        n = max(1, round(GEN_FRAC * (length - 1)))
        pos = 1 + np.random.default_rng([MASK_SEED, c, m]).choice(length - 1, n, replace=False)
        gen[i, torch.as_tensor(pos)] = True
    return gen


@torch.no_grad()
def cell_losses(model, seqs, cell_ids, pad_id):
    """Mean squared error on hidden bins per cell, averaged over fixed masks."""
    model.eval()
    order = sorted(cell_ids, key=lambda c: len(seqs[c][0]))
    totals = {c: 0.0 for c in order}
    for m in range(N_MASKS):
        for start in range(0, len(order), EVAL_BATCH):
            ids = order[start:start + EVAL_BATCH]
            genes, vals, pad = collate([seqs[c] for c in ids], pad_id)
            gen = fixed_masks(seqs, ids, genes.shape, m)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(genes.to(DEVICE), vals.to(DEVICE), pad.to(DEVICE), gen.to(DEVICE))
            err = ((pred.float().cpu() - vals) ** 2) * gen
            per_cell = (err.sum(1) / gen.sum(1)).tolist()
            for c, value in zip(ids, per_cell):
                totals[c] += value / N_MASKS
    return totals


def train(model, seqs, fit_ids, val_ids, pad_id, lr, epochs, patience, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    best, best_val, stale, history = None, np.inf, 0, []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(fit_ids)
        running, n_batches = 0.0, 0
        for start in range(0, len(order), BATCH):
            genes, vals, pad = collate([seqs[c] for c in order[start:start + BATCH].tolist()], pad_id)
            gen = (torch.rand(genes.shape) < GEN_FRAC) & ~pad
            gen[:, 0] = False
            if not gen.any():
                continue
            genes, vals, pad, gen = genes.to(DEVICE), vals.to(DEVICE), pad.to(DEVICE), gen.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(genes, vals, pad, gen)
            loss = ((pred.float() - vals) ** 2)[gen].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            running += loss.item()
            n_batches += 1
        val = float(np.mean(list(cell_losses(model, seqs, val_ids, pad_id).values())))
        history.append({"epoch": epoch, "train_loss": running / max(n_batches, 1), "val_loss": val})
        print(f"  epoch {epoch}: train {history[-1]['train_loss']:.3f} | val {val:.3f}", flush=True)
        if val < best_val:
            best_val, stale = val, 0
            if patience is not None:
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if patience is not None and stale >= patience:
                break
    if best is not None:
        model.load_state_dict(best)
    return history


def done(csv, dataset, nd, trial, label, rank):
    if not os.path.exists(csv):
        return False
    d = pd.read_csv(csv)
    hit = d[(d.dataset == dataset) & (d.nd == nd) & (d.trial == trial) & (d.config == label)]
    hit = hit[hit["rank"].isna()] if rank is None else hit[hit["rank"] == rank]
    return len(hit) > 0


def run_config(data, c, base_loss, config, rank, csv, smoke, fit_ids=None, label=None, lr=None,
               epochs=None, patience=PATIENCE):
    label = label or config
    if done(csv, data.name, c.nd, c.trial, label, rank):
        print(f"skip {label} rank={rank}", flush=True)
        return None
    t0 = time.time()
    model = sm.load_scgpt(DEVICE)
    n = sm.apply_config(model, config, rank)
    history = []
    if config != "base":
        lr = lr or LR.get(config, LR["default"])
        epochs = 1 if smoke else (epochs or MAX_EPOCHS)
        print(f"--- scgpt {label} rank={rank} | trainable {n:,} | lr {lr}", flush=True)
        history = train(model, c.seqs, fit_ids or p1.flat(c.cells["fit"]), p1.flat(c.cells["validation"]),
                        data.tok.pad_id, lr, epochs, patience, SEED)
    loss = cell_losses(model, c.seqs, c.eval_ids, data.tok.pad_id) if config != "base" else base_loss
    utility = float(np.mean([loss[x] for x in p1.flat(c.cells["holdout"])]))
    meta = {"model": "scgpt", "version": "whole-human", "dataset": data.name, "nd": c.nd, "trial": c.trial,
            "objective": "O1", "config": label, "rank": rank if rank is not None else np.nan, "n_trainable": n,
            "utility": utility, "utility_metric": "holdout_generated_bin_mse", "seed": SEED,
            "runtime": time.time() - t0}
    attacks = ["raw_loss"] if config == "base" else ["raw_loss", "calibrated_loss"]
    rows, _ = p1.attack_rows(meta, c.cells, base_loss, loss, c.ctype, attacks)
    rs.append(csv, rows)
    tag = f"{data.name}_{c.nd}d_t{c.trial}_{label}" + (f"_r{rank}" if rank else "")
    json.dump(history, open(f"{REPO}/logs/history_scgpt_{'smoke_' if smoke else ''}{tag}.json", "w"))
    if not smoke:
        os.makedirs(f"{REPO}/scores/scgpt", exist_ok=True)
        ids = np.array(sorted(loss), dtype=np.int64)
        np.savez_compressed(f"{REPO}/scores/scgpt/{tag}.npz", cell=ids,
                            base=np.array([base_loss[int(x)] for x in ids], dtype=np.float32),
                            loss=np.array([loss[int(x)] for x in ids], dtype=np.float32))
        if config == "lora":
            os.makedirs(f"{REPO}/checkpoints/scgpt", exist_ok=True)
            torch.save({k: {"A": m.lora_A.detach().cpu(), "B": m.lora_B.detach().cpu()}
                        for k, m in model.named_modules() if isinstance(m, LoRALinear)},
                       f"{REPO}/checkpoints/scgpt/{tag}.pt")
    summary = {(r["threat_model"], r["attack"]): round(r["auc"], 4) for r in rows if r["aggregation"] == "mean"}
    print(f"scgpt {label} rank={rank}: holdout MSE {utility:.3f} | epochs {len(history)} | "
          f"{time.time() - t0:.0f}s | {summary}", flush=True)
    del model
    torch.cuda.empty_cache()
    return rows


def cohort_base(data, nd, trial, smoke):
    c = rs.Cohort(data, nd, trial, smoke)
    model = sm.load_scgpt(DEVICE)
    base = cell_losses(model, c.seqs, c.eval_ids, data.tok.pad_id)
    del model
    return c, base


def positive_control(data, csv):
    """Returns True if the dataset passes. A failed control drops the dataset from the queue."""
    name = data.name
    if not done(csv, name, 10, 1, "full_positive", None):
        c, base = cohort_base(data, 10, 1, False)
        run_config(data, c, base, "full", None, csv, False, label="full_positive", lr=POSITIVE_LR,
                   epochs=MAX_EPOCHS, patience=None)
        del c
    d = pd.read_csv(csv)
    auc = d[(d.dataset == name) & (d.config == "full_positive") & (d.threat_model == "seen_cells")
            & (d.attack == "calibrated_loss") & (d.aggregation == "mean")]["auc"].iloc[0]
    print(f"POSITIVE CONTROL scgpt {name}: seen AUC {auc:.3f}", flush=True)
    if auc < 0.9:
        print(f"POSITIVE CONTROL FAILED for scgpt {name}; dropping this dataset", flush=True)
    return auc >= 0.9


def run_cohort(data, trial, configs, csv):
    """Returns False if the null fails, which drops the dataset from later trials."""
    name = data.name
    pending = [(cf, r) for cf, r in configs if not done(csv, name, 50, trial, cf, r)]
    if not pending and done(csv, name, 50, trial, "null_unrelated_cohort", None):
        return True
    c, base = cohort_base(data, 50, trial, False)
    run_config(data, c, base, "base", None, csv, False)
    rows = run_config(data, c, base, "full", None, csv, False, fit_ids=c.null_fit,
                      label="null_unrelated_cohort", epochs=NULL_EPOCHS, patience=None)
    if rows:
        sd = p1.null_sd(len(c.cells["fit"]), len(c.cells["holdout"]))
        aucs = {r["threat_model"]: r["auc"] for r in rows
                if r["attack"] == "calibrated_loss" and r["aggregation"] == "mean"}
        print(f"CALIBRATED NULL scgpt {name} 50d t{trial}: {aucs} | null SD {sd:.3f}", flush=True)
        if any(abs(a - 0.5) > NULL_STOP_SDS * sd for a in aucs.values()):
            print(f"NULL BEYOND {NULL_STOP_SDS} SD; dropping scgpt {name} from later trials", flush=True)
            return False
    for config, rank in pending:
        run_config(data, c, base, config, rank, csv, False)
    del c
    torch.cuda.empty_cache()
    return True


def smoke_run(csv):
    data = ScData("aida")
    c, base = cohort_base(data, 10, 1, True)
    run_config(data, c, base, "base", None, csv, True)
    run_config(data, c, base, "lora", 2, csv, True)
    run_config(data, c, base, "full", None, csv, True, fit_ids=c.null_fit, label="null_unrelated_cohort")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    csv = f"{REPO}/logs/smoke_scgpt.csv" if args.smoke else CSV
    if args.smoke and os.path.exists(csv):
        os.remove(csv)
    if not args.smoke:
        json.dump({"model": "scGPT whole-human", "source": sm.SCGPT_SOURCE, "sha256": sm.SCGPT_SHA256,
                   "objective": "generative bin prediction (pretraining objective of this checkpoint)",
                   "gen_frac": GEN_FRAC, "n_masks": N_MASKS, "mask_seed": MASK_SEED, "seed": SEED,
                   "batch": BATCH, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "lr": LR,
                   "trial_plan": [[t, [list(x) for x in cf]] for t, cf in TRIAL_PLAN],
                   "torch": torch.__version__, "transformers": transformers.__version__,
                   "python": platform.python_version()},
                  open(f"{REPO}/results/scgpt_meta.json", "w"), indent=2)
    if args.smoke:
        smoke_run(csv)
        print("SCGPT SMOKE PASSED", flush=True)
        return
    # Trial by trial across datasets, so a partial run stays balanced.
    active = [d for d in (ScData(n) for n in DATASETS) if positive_control(d, csv)]
    for trial, configs in TRIAL_PLAN:
        active = [d for d in active if run_cohort(d, trial, configs, csv)]
        print(f"SCGPT TRIAL {trial} COMPLETE", flush=True)
    print("PHASE 6 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
