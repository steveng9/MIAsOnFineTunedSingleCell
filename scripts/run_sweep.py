"""Phases 2 and 3: freezing and LoRA sweeps for Geneformer V1 (objective O1).

Each dataset runs its gates first and stops if one fails:
  positive control  few donors, all weights, many epochs
  calibrated null   fine-tune on unrelated auxiliary donors, then attack members vs holdout;
                    neither group was trained on, so AUC should be near 0.5
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
import peft
import torch
import transformers
from transformers import BertForMaskedLM

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, f"{REPO}/src"]
import phase1_pilot as p1  # noqa: E402
from gf_tokenizer import MODEL_DIR, MODEL_REVISION, GeneformerV1Tokenizer, load_dict  # noqa: E402
from lora import LoRALinear, apply_config  # noqa: E402

p1.EVAL_BATCH = 16
SEED = p1.SEED
FREEZE_CONFIGS = ["head", "last1", "last2", "last4", "full"]
RANKS = [1, 2, 4, 8, 16, 32, 64]
LR = {"lora": 1e-3, "default": 1e-4}
MAX_EPOCHS, PATIENCE = 30, 3
NULL_EPOCHS = 10
COLUMNS = ["model", "version", "dataset", "nd", "trial", "objective", "config", "rank", "n_trainable",
           "threat_model", "attack", "aggregation", "auc", "tpr_at_fpr", "utility", "utility_metric",
           "seed", "runtime"]


class Data:
    def __init__(self, name):
        self.name = name
        self.adata = ad.read_h5ad(os.path.expanduser(f"~/data/scMAMAMIA/{name}/full_dataset_cleaned.h5ad"), backed="r")
        self.matrix = self.adata.raw.X if self.adata.raw is not None else self.adata.X
        genes = np.asarray(self.adata.raw.var_names if self.adata.raw is not None else self.adata.var_names).astype(str)
        if not genes[0].startswith("ENSG"):
            lookup = load_dict("gene_name_id_dict_gc30M.pkl")
            genes = np.array([lookup.get(g, "") for g in genes])
            usable = [g for g in genes if g]
            assert len(usable) == len(set(usable)), "symbol mapping produced duplicate Ensembl IDs"
        self.tok = GeneformerV1Tokenizer(genes)
        self.donors = self.adata.obs["individual"].astype(str).to_numpy()
        self.ctypes = self.adata.obs["cell_type"].astype(str).to_numpy()

    def tokenize(self, indices, chunk=256):
        indices = np.sort(np.unique(indices))
        seqs = {}
        for start in range(0, len(indices), chunk):
            part = indices[start:start + chunk]
            rows = self.matrix[part].toarray()
            for i, cell in enumerate(part):
                seqs[int(cell)] = self.tok.encode(rows[i])
        assert all(len(s) > 0 for s in seqs.values()), "empty tokenized cell"
        return seqs


class Cohort:
    def __init__(self, data, nd, trial, smoke):
        self.nd, self.trial = nd, trial
        split = p1.load_split(data.name, nd, trial)
        rng = np.random.default_rng([SEED, nd, trial])
        cap, val_cap = (10, 5) if smoke else (p1.CELL_CAP, p1.VAL_CELL_CAP)
        if smoke:
            for key, n in (("train", 3), ("holdout", 3), ("validation", 2)):
                split[key] = set(sorted(split[key])[:n])
        self.split = split
        self.cells = p1.select_cells(data.donors, split, rng, cap, val_cap)
        counts = p1.check_leakage(self.cells, split)

        # Separate, larger sample of the validation donors for the unrelated-cohort null.
        null_rng = np.random.default_rng([SEED, nd, trial, 99])
        self.null_fit = []
        for d in self.cells["validation"]:
            where = np.flatnonzero(data.donors == d)
            self.null_fit.extend(null_rng.permutation(where)[:cap].tolist())
        target_ids = set(p1.flat(self.cells["fit"]) + p1.flat(self.cells["unseen"]) + p1.flat(self.cells["holdout"]))
        assert not target_ids & set(self.null_fit), "null-model training cells overlap attack targets"

        all_ids = np.array(sorted(target_ids | set(p1.flat(self.cells["validation"])) | set(self.null_fit)))
        self.seqs = data.tokenize(all_ids)
        self.ctype = {int(c): data.ctypes[int(c)] for c in all_ids}
        self.eval_ids = p1.flat(self.cells["fit"]) + p1.flat(self.cells["unseen"]) + p1.flat(self.cells["holdout"])
        print(f"{data.name} {nd}d trial {trial}: {counts} | null-fit cells {len(self.null_fit)}", flush=True)


def fresh():
    return BertForMaskedLM.from_pretrained(MODEL_DIR).to(p1.DEVICE)


def done(csv, dataset, nd, trial, config, rank):
    if not os.path.exists(csv):
        return False
    d = pd.read_csv(csv)
    hit = d[(d.dataset == dataset) & (d.nd == nd) & (d.trial == trial) & (d.objective == "O1") & (d.config == config)]
    hit = hit[hit["rank"].isna()] if rank is None else hit[hit["rank"] == rank]
    return len(hit) > 0


def append(csv, rows):
    new = pd.DataFrame(rows).reindex(columns=COLUMNS)
    if os.path.exists(csv):
        old = pd.read_csv(csv).reindex(columns=COLUMNS)
        new = pd.concat([old, new], ignore_index=True)
    tmp = f"{csv}.tmp"
    new.to_csv(tmp, index=False)
    os.replace(tmp, csv)


def save_artifacts(model, name, config):
    os.makedirs(f"{REPO}/checkpoints", exist_ok=True)
    if config == "lora":
        adapter = {n: {"A": m.lora_A.detach().cpu(), "B": m.lora_B.detach().cpu()}
                   for n, m in model.named_modules() if isinstance(m, LoRALinear)}
        torch.save(adapter, f"{REPO}/checkpoints/{name}.pt")
    elif config == "full":
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f"{REPO}/checkpoints/{name}.pt")


def run_config(data, c, base_loss, config, rank, csv, smoke, fit_ids=None, label=None, lr=None,
               epochs=None, patience=PATIENCE):
    label = label or config
    if done(csv, data.name, c.nd, c.trial, label, rank):
        print(f"skip {label} rank={rank}", flush=True)
        return None
    t0 = time.time()
    model = fresh()
    n = apply_config(model, config, rank)
    history = []
    if config != "base":
        lr = lr or LR.get(config, LR["default"])
        epochs = 1 if smoke else (epochs or MAX_EPOCHS)
        print(f"--- {label} rank={rank} | trainable {n:,} | lr {lr}", flush=True)
        history = p1.train(model, c.seqs, fit_ids or p1.flat(c.cells["fit"]),
                           p1.flat(c.cells["validation"]), data.tok, lr=lr, epochs=epochs,
                           patience=patience, seed=SEED)
    loss = p1.cell_losses(model, c.seqs, c.eval_ids, data.tok) if config != "base" else base_loss
    utility = float(np.mean([loss[x] for x in p1.flat(c.cells["holdout"])]))
    meta = {"model": "geneformer", "version": "V1-10M", "dataset": data.name, "nd": c.nd, "trial": c.trial,
            "objective": "O1", "config": label, "rank": rank if rank is not None else np.nan, "n_trainable": n,
            "utility": utility, "utility_metric": "holdout_masked_ce", "seed": SEED, "runtime": time.time() - t0}
    attacks = ["raw_loss"] if config == "base" else ["raw_loss", "calibrated_loss"]
    rows, _ = p1.attack_rows(meta, c.cells, base_loss, loss, c.ctype, attacks)
    append(csv, rows)
    tag = f"{data.name}_{c.nd}d_t{c.trial}_{label}" + (f"_r{rank}" if rank else "")
    json.dump(history, open(f"{REPO}/logs/history_{'smoke_' if smoke else ''}{tag}.json", "w"))
    if not smoke and label in ("full", "lora"):
        save_artifacts(model, tag, config)
    summary = {(r["threat_model"], r["attack"]): round(r["auc"], 4) for r in rows if r["aggregation"] == "mean"}
    print(f"{label} rank={rank}: holdout CE {utility:.4f} | best epochs {len(history)} | "
          f"{time.time() - t0:.0f}s | {summary}", flush=True)
    del model
    torch.cuda.empty_cache()
    return rows


def sweep(name, nd, trial, csv, smoke, freeze_configs, ranks):
    data = Data(name)

    pos = Cohort(data, 10, trial, smoke)
    pos_base = p1.cell_losses(fresh(), pos.seqs, pos.eval_ids, data.tok)
    if name == "aida" and not smoke and done(csv, name, 10, trial, "full_positive", None):
        print("positive control already recorded for aida", flush=True)
    else:
        rows = run_config(data, pos, pos_base, "full", None, csv, smoke, label="full_positive", lr=5e-4,
                          epochs=30, patience=None)
        if rows is not None:
            auc = next(r["auc"] for r in rows if r["threat_model"] == "seen_cells"
                       and r["attack"] == "calibrated_loss" and r["aggregation"] == "mean")
            if auc < 0.9 and not smoke:
                sys.exit(f"POSITIVE CONTROL FAILED for {name}: AUC {auc:.3f}")
    del pos

    c = Cohort(data, nd, trial, smoke)
    base_loss = p1.cell_losses(fresh(), c.seqs, c.eval_ids, data.tok)
    run_config(data, c, base_loss, "base", None, csv, smoke)

    rows = run_config(data, c, base_loss, "full", None, csv, smoke, fit_ids=c.null_fit,
                      label="null_unrelated_cohort", epochs=NULL_EPOCHS, patience=None)
    if rows is not None:
        sd = p1.null_sd(len(c.cells["fit"]), len(c.cells["holdout"]))
        aucs = {r["threat_model"]: r["auc"] for r in rows
                if r["attack"] == "calibrated_loss" and r["aggregation"] == "mean"}
        print(f"CALIBRATED NULL {name}: {aucs} | null SD {sd:.3f}", flush=True)
        if not smoke and any(abs(a - 0.5) > 2 * sd for a in aucs.values()):
            sys.exit(f"CALIBRATED NULL FAILED for {name}: {aucs}")

    for config in freeze_configs:
        run_config(data, c, base_loss, config, None, csv, smoke)
    for r in ([2] if smoke else ranks):
        run_config(data, c, base_loss, "lora", r, csv, smoke)
    data.adata.file.close()
    print(f"SWEEP COMPLETE {name} {nd}d trial {trial}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ok", "aida"])
    ap.add_argument("--nd", type=int, default=50)
    ap.add_argument("--trial", type=int, default=1)
    ap.add_argument("--freeze", nargs="*", default=FREEZE_CONFIGS)
    ap.add_argument("--ranks", nargs="*", type=int, default=RANKS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    csv = f"{REPO}/logs/smoke_sweep.csv" if args.smoke else f"{REPO}/results/metrics.csv"
    if args.smoke and os.path.exists(csv):
        os.remove(csv)
    torch.manual_seed(SEED)
    if not args.smoke:
        json.dump({"python": platform.python_version(), "torch": torch.__version__,
                   "transformers": transformers.__version__, "peft": peft.__version__,
                   "anndata": ad.__version__, "numpy": np.__version__, "model_revision": MODEL_REVISION,
                   "seed": SEED, "mask_seed": p1.MASK_SEED, "mask_frac": p1.MASK_FRAC, "n_masks": p1.N_MASKS,
                   "cell_cap": p1.CELL_CAP, "lr": LR, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                   "null_epochs": NULL_EPOCHS, "ranks": args.ranks, "freeze_configs": args.freeze,
                   "datasets": args.datasets, "nd": args.nd, "trial": args.trial},
                  open(f"{REPO}/results/sweep_meta_{args.nd}d_t{args.trial}.json", "w"), indent=2)
    for name in args.datasets:
        sweep(name, args.nd, args.trial, csv, args.smoke, args.freeze, args.ranks)
    print("ALL SWEEPS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
