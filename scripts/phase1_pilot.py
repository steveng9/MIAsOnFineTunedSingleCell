"""Phase 1: Geneformer V1 pilot with required controls (objective O1, masked-gene loss).

Order: leakage check -> null -> positive -> label shuffle -> pilot fine-tune.
The script exits before the pilot if any control fails.
"""
import argparse
import copy
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
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import BertForMaskedLM

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{REPO}/src")
from gf_tokenizer import MODEL_DIR, MODEL_REVISION, GeneformerV1Tokenizer  # noqa: E402

SEED, MASK_SEED = 0, 1234
MASK_FRAC, N_MASKS = 0.15, 3
CELL_CAP, VAL_CELL_CAP, VAL_DONOR_CAP = 200, 50, 20
BATCH, EVAL_BATCH = 12, 24
FPR_TARGET = 0.10
DEVICE = "cuda"


def load_split(dataset, nd, trial):
    root = os.path.expanduser(f"~/data/scMAMAMIA/{dataset}/splits/{nd}d/{trial}")
    s = {k: set(map(str, np.load(f"{root}/{k}.npy", allow_pickle=True).tolist()))
         for k in ("train", "holdout", "auxiliary")}
    assert not s["train"] & s["holdout"], "train and holdout overlap"
    s["validation"] = s["auxiliary"] - s["train"] - s["holdout"]
    assert s["validation"], "no auxiliary donors remain after removing train and holdout"
    return s


def select_cells(donors_arr, split, rng, cap, val_cap):
    by_donor = pd.Series(np.arange(len(donors_arr))).groupby(donors_arr).apply(lambda v: v.to_numpy())
    out = {k: {} for k in ("fit", "unseen", "holdout", "validation")}
    for d in sorted(split["train"]):
        p = rng.permutation(by_donor[d])
        n_fit = min(cap, int(np.ceil(len(p) / 2)))
        out["fit"][d] = p[:n_fit]
        if len(p) > n_fit:
            out["unseen"][d] = p[n_fit:n_fit + cap]
    for d in sorted(split["holdout"]):
        out["holdout"][d] = rng.permutation(by_donor[d])[:cap]
    for d in sorted(split["validation"])[:VAL_DONOR_CAP]:
        out["validation"][d] = rng.permutation(by_donor[d])[:val_cap]
    return out


def check_leakage(cells, split):
    ids = {k: np.concatenate(list(v.values())) for k, v in cells.items()}
    assert not set(cells["fit"]) & split["holdout"], "holdout donor in fit data"
    assert not set(cells["validation"]) & split["holdout"], "holdout donor in validation data"
    assert not set(cells["validation"]) & split["train"], "member donor in validation data"
    for a_key, b_key in (("fit", "holdout"), ("validation", "holdout"), ("fit", "unseen"), ("fit", "validation")):
        assert not np.intersect1d(ids[a_key], ids[b_key]).size, f"cell overlap: {a_key}/{b_key}"
    return {k: len(v) for k, v in ids.items()}


def tokenize(adata, tok, indices, chunk=256):
    indices = np.sort(np.unique(indices))
    seqs = {}
    for start in range(0, len(indices), chunk):
        part = indices[start:start + chunk]
        rows = adata.raw.X[part].toarray()
        for i, cell in enumerate(part):
            seqs[int(cell)] = tok.encode(rows[i])
    return seqs


def collate(batch, pad_id):
    max_len = max(len(s) for s in batch)
    x = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, s in enumerate(batch):
        x[i, :len(s)] = torch.as_tensor(s)
        attn[i, :len(s)] = 1
    return x, attn


def masked_logits(model, x, attn, mask):
    hidden = model.bert(input_ids=x, attention_mask=attn).last_hidden_state
    return model.cls(hidden[mask])


@torch.no_grad()
def cell_losses(model, seqs, cell_ids, tok):
    """Mean masked-gene loss per cell, using fixed masks seeded by cell ID."""
    model.eval()
    order = sorted(cell_ids, key=lambda c: len(seqs[c]))
    totals = {c: 0.0 for c in order}
    for m in range(N_MASKS):
        for start in range(0, len(order), EVAL_BATCH):
            ids = order[start:start + EVAL_BATCH]
            x, attn = collate([seqs[c] for c in ids], tok.pad_id)
            mask = torch.zeros_like(x, dtype=torch.bool)
            for i, c in enumerate(ids):
                length = len(seqs[c])
                n = max(1, round(MASK_FRAC * length))
                pos = np.random.default_rng([MASK_SEED, c, m]).choice(length, n, replace=False)
                mask[i, torch.as_tensor(pos)] = True
            labels = x[mask].to(DEVICE)
            inp = x.masked_fill(mask, tok.mask_id).to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = masked_logits(model, inp, attn.to(DEVICE), mask.to(DEVICE))
            loss = torch.nn.functional.cross_entropy(logits.float(), labels, reduction="none").cpu()
            row = mask.nonzero()[:, 0]
            summed = torch.zeros(len(ids)).index_add_(0, row, loss)
            counts = mask.sum(1).float()
            for c, value in zip(ids, (summed / counts).tolist()):
                totals[c] += value / N_MASKS
    return totals


def train(model, seqs, fit_ids, val_ids, tok, lr, epochs, patience, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    best, best_val, stale, history = None, np.inf, 0, []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(fit_ids)
        running, n_batches = 0.0, 0
        for start in range(0, len(order), BATCH):
            ids = order[start:start + BATCH].tolist()
            x, attn = collate([seqs[c] for c in ids], tok.pad_id)
            prob = torch.full(x.shape, MASK_FRAC) * attn
            mask = torch.bernoulli(prob).bool()
            if not mask.any():
                continue
            labels = x[mask].to(DEVICE)
            inp = x.masked_fill(mask, tok.mask_id).to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = masked_logits(model, inp, attn.to(DEVICE), mask.to(DEVICE))
                loss = torch.nn.functional.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            n_batches += 1
        val = float(np.mean(list(cell_losses(model, seqs, val_ids, tok).values())))
        history.append({"epoch": epoch, "train_loss": running / max(n_batches, 1), "val_loss": val})
        print(f"  epoch {epoch}: train {history[-1]['train_loss']:.4f} | val {val:.4f}", flush=True)
        if val < best_val:
            best_val, stale = val, 0
            if patience is not None:
                best = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if patience is not None and stale >= patience:
                break
    if best is not None:
        model.load_state_dict(best)
    return history


def aggregate(values, cell_types, how):
    v = pd.Series(values)
    if how == "mean":
        return float(v.mean())
    if how == "median":
        return float(v.median())
    return float(v.groupby(np.asarray(cell_types)).mean().mean())


def donor_scores(cell_map, score, ctype, how):
    return {d: aggregate([score[int(c)] for c in ids], [ctype[int(c)] for c in ids], how)
            for d, ids in cell_map.items()}


def evaluate(member_map, holdout_map, score, ctype, how):
    m, h = donor_scores(member_map, score, ctype, how), donor_scores(holdout_map, score, ctype, how)
    y = np.r_[np.ones(len(m)), np.zeros(len(h))]
    s = np.r_[list(m.values()), list(h.values())]
    fpr, tpr, _ = roc_curve(y, s)
    return float(roc_auc_score(y, s)), float(tpr[fpr <= FPR_TARGET].max()), y, s


def null_sd(n, m):
    return float(np.sqrt((n + m + 1) / (12 * n * m)))


def attack_rows(meta, cells, base_loss, loss, ctype, attacks):
    rows, scored = [], {}
    for threat, key in (("seen_cells", "fit"), ("same_donor_unseen_cells", "unseen")):
        for attack in attacks:
            if attack == "raw_loss":
                score = {c: -loss[c] for c in loss}
            else:
                score = {c: base_loss[c] - loss[c] for c in loss}
            for how in ("mean", "median", "celltype_equal"):
                auc, tpr, y, s = evaluate(cells[key], cells["holdout"], score, ctype, how)
                rows.append({**meta, "threat_model": threat, "attack": attack, "aggregation": how,
                             "auc": auc, "tpr_at_fpr": tpr})
                scored[(threat, attack, how)] = (y, s)
    return rows, scored


def prepare(adata, tok, donors, ctypes, dataset, nd, trial, smoke):
    split = load_split(dataset, nd, trial)
    rng = np.random.default_rng([SEED, nd, trial])
    cap, val_cap = (10, 5) if smoke else (CELL_CAP, VAL_CELL_CAP)
    if smoke:
        for key, n in (("train", 3), ("holdout", 3), ("validation", 2)):
            split[key] = set(sorted(split[key])[:n])
    cells = select_cells(donors, split, rng, cap, val_cap)
    counts = check_leakage(cells, split)
    print(f"{dataset} {nd}d trial {trial}: donors train {len(split['train'])}, holdout {len(split['holdout'])}, "
          f"validation {len(cells['validation'])} | cells {counts}", flush=True)
    all_ids = np.concatenate([v for group in cells.values() for v in group.values()])
    seqs = tokenize(adata, tok, all_ids)
    assert all(len(s) > 0 for s in seqs.values()), "empty tokenized cell"
    ctype = {int(c): ctypes[int(c)] for c in all_ids}
    return split, cells, seqs, ctype


def flat(group):
    return [int(c) for v in group.values() for c in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out_csv = f"{REPO}/logs/smoke_metrics.csv" if args.smoke else f"{REPO}/results/metrics.csv"
    if os.path.exists(out_csv):
        os.remove(out_csv) if args.smoke else sys.exit(f"{out_csv} exists; refusing to overwrite")

    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    dataset, version = "aida", "V1-10M"
    adata = ad.read_h5ad(os.path.expanduser(f"~/data/scMAMAMIA/{dataset}/full_dataset_cleaned.h5ad"), backed="r")
    donors = adata.obs["individual"].astype(str).to_numpy()
    ctypes = adata.obs["cell_type"].astype(str).to_numpy()
    tok = GeneformerV1Tokenizer(adata.raw.var_names)
    base_model = BertForMaskedLM.from_pretrained(MODEL_DIR).to(DEVICE)
    base_state = copy.deepcopy({k: v.cpu() for k, v in base_model.state_dict().items()})
    n_full = sum(p.numel() for p in base_model.parameters())

    json.dump({"python": platform.python_version(), "torch": torch.__version__,
               "transformers": transformers.__version__, "peft": peft.__version__,
               "anndata": ad.__version__, "numpy": np.__version__, "model_revision": MODEL_REVISION,
               "model_version": version, "seed": SEED, "mask_seed": MASK_SEED, "mask_frac": MASK_FRAC,
               "n_masks": N_MASKS, "cell_cap": CELL_CAP, "dataset": dataset},
              open(f"{REPO}/{'logs/smoke_meta' if args.smoke else 'results/phase1_meta'}.json", "w"), indent=2)

    rows, gates = [], {}
    common = {"model": "geneformer", "version": version, "dataset": dataset, "objective": "O1",
              "rank": np.nan, "seed": SEED, "utility_metric": "holdout_masked_ce"}

    def fresh():
        base_model.load_state_dict(base_state)
        for p in base_model.parameters():
            p.requires_grad = True
        return base_model

    # NULL: pilot cohort on the unmodified base model.
    nd, trial = 50, 1
    split, cells, seqs, ctype = prepare(adata, tok, donors, ctypes, dataset, nd, trial, args.smoke)
    eval_ids = flat(cells["fit"]) + flat(cells["unseen"]) + flat(cells["holdout"])
    t0 = time.time()
    base_loss = cell_losses(fresh(), seqs, eval_ids, tok)
    util = float(np.mean([base_loss[c] for c in flat(cells["holdout"])]))
    meta = {**common, "nd": nd, "trial": trial, "config": "base_null", "n_trainable": 0,
            "utility": util, "runtime": time.time() - t0}
    null_rows, _ = attack_rows(meta, cells, base_loss, base_loss, ctype, ["raw_loss"])
    rows += null_rows
    sd = null_sd(len(cells["fit"]), len(cells["holdout"]))
    null_aucs = [r["auc"] for r in null_rows if r["aggregation"] == "mean"]
    gates["null"] = {"aucs_mean_agg": null_aucs, "null_sd": sd,
                     "passed": all(abs(a - 0.5) <= 2 * sd for a in null_aucs)}
    print("NULL", gates["null"], flush=True)

    # POSITIVE: few donors, many epochs, all weights.
    p_nd = 10
    p_split, p_cells, p_seqs, p_ctype = prepare(adata, tok, donors, ctypes, dataset, p_nd, trial, args.smoke)
    p_eval = flat(p_cells["fit"]) + flat(p_cells["unseen"]) + flat(p_cells["holdout"])
    p_base = cell_losses(fresh(), p_seqs, p_eval, tok)
    t0 = time.time()
    model = fresh()
    history = train(model, p_seqs, flat(p_cells["fit"]), flat(p_cells["validation"]), tok,
                    lr=5e-4, epochs=2 if args.smoke else 30, patience=None, seed=SEED)
    p_loss = cell_losses(model, p_seqs, p_eval, tok)
    util = float(np.mean([p_loss[c] for c in flat(p_cells["holdout"])]))
    meta = {**common, "nd": p_nd, "trial": trial, "config": "full_positive", "n_trainable": n_full,
            "utility": util, "runtime": time.time() - t0}
    p_rows, p_scored = attack_rows(meta, p_cells, p_base, p_loss, p_ctype, ["raw_loss", "calibrated_loss"])
    rows += p_rows
    json.dump(history, open(f"{REPO}/logs/{'smoke_' if args.smoke else ''}positive_history.json", "w"))
    pos_auc = next(r["auc"] for r in p_rows if r["threat_model"] == "seen_cells"
                   and r["attack"] == "calibrated_loss" and r["aggregation"] == "mean")
    gates["positive"] = {"auc_seen_calibrated_mean": pos_auc, "passed": pos_auc >= 0.9}
    print("POSITIVE", gates["positive"], flush=True)

    # LABEL SHUFFLE: permute membership labels on the positive-control scores.
    y, s = p_scored[("seen_cells", "calibrated_loss", "mean")]
    perm_rng = np.random.default_rng(SEED)
    perm = [roc_auc_score(perm_rng.permutation(y), s) for _ in range(1000)]
    rows.append({**meta, "config": "full_positive_label_shuffle", "threat_model": "seen_cells",
                 "attack": "calibrated_loss", "aggregation": "mean", "auc": float(np.mean(perm)),
                 "tpr_at_fpr": np.nan})
    gates["shuffle"] = {"mean_auc": float(np.mean(perm)), "sd": float(np.std(perm)),
                        "passed": abs(float(np.mean(perm)) - 0.5) <= 0.02}
    print("SHUFFLE", gates["shuffle"], flush=True)
    gates["leakage"] = {"passed": True}

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    failed = [k for k, v in gates.items() if not v["passed"]]
    json.dump(gates, open(f"{REPO}/{'logs/smoke_gates' if args.smoke else 'results/phase1_gates'}.json", "w"), indent=2)
    if failed and not args.smoke:
        sys.exit(f"CONTROL FAILED: {failed}. Stopping before the pilot.")

    # PILOT: full fine-tune on 50 donors with early stopping on auxiliary donors.
    t0 = time.time()
    model = fresh()
    history = train(model, seqs, flat(cells["fit"]), flat(cells["validation"]), tok,
                    lr=1e-4, epochs=1 if args.smoke else 10, patience=2, seed=SEED)
    loss = cell_losses(model, seqs, eval_ids, tok)
    util = float(np.mean([loss[c] for c in flat(cells["holdout"])]))
    meta = {**common, "nd": nd, "trial": trial, "config": "full_pilot", "n_trainable": n_full,
            "utility": util, "runtime": time.time() - t0}
    pilot_rows, _ = attack_rows(meta, cells, base_loss, loss, ctype, ["raw_loss", "calibrated_loss"])
    rows += pilot_rows
    json.dump(history, open(f"{REPO}/logs/{'smoke_' if args.smoke else ''}pilot_history.json", "w"))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("PILOT")
    print(pd.DataFrame(pilot_rows).query("aggregation == 'mean'")[["threat_model", "attack", "auc", "tpr_at_fpr"]])
    print("PHASE 1 COMPLETE", flush=True)
    adata.file.close()


if __name__ == "__main__":
    main()
