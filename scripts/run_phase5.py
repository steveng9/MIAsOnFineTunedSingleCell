"""Phase 5: fine-tuning (objective O2, cell-type head) vs in-context learning (ICL-A), Geneformer V1.

Both sides use the same 50-donor cohorts, member fit cells as data, and holdout donors for utility.

ICL-A (no weight updates): the frozen base model embeds cells (mean of the second-to-last layer,
as Geneformer recommends for embeddings). A query is classified against a labeled support set of
member cells, by k nearest neighbors (cosine, 10 neighbors) or by cell-type prototypes (softmax
over cosine to class means, temperature 0.1).

O2 fine-tuning: a linear head on the mean-pooled final layer, trained with cross-entropy.
Configs: head only (encoder frozen), last 2 layers, full, LoRA r=8. Early stopping on auxiliary
validation donors.

Support size: k cells per cell type, drawn from member fit cells, k in {1, 5, 10, 50}, plus "all"
(every fit cell, the same data O1 fine-tuning used). k = 0 has no ICL-A analogue and is skipped.

Attack (query access returning class probabilities): per-cell loss = cross-entropy of the true
type. Calibrated score = L_reference - L_target, where the reference is the same method built on
unrelated auxiliary donors (null-fit cells). Members are the donors that contributed support or
training cells; "seen" scores those exact cells and "unseen" scores the same donors' other cells.
Reference-only scores (neither group used) give a null AUC.

ICL-B (LLM few-shot prompting on cell sentences) is not run; see RESULTS.md for why.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import BertForMaskedLM

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, f"{REPO}/src"]
import phase1_pilot as p1  # noqa: E402
import run_sweep as rs  # noqa: E402
from gf_tokenizer import MODEL_DIR  # noqa: E402
from lora import LoRALinear, apply_config  # noqa: E402

DEVICE = "cuda"
SEED = 0
DATASETS, ND, TRIALS = ["ok", "aida"], 50, [1, 2, 3]
KS = [1, 5, 10, 50, "all"]
FT_ALL = [("head", None), ("last2", None), ("full", None), ("lora", 8)]
FT_FEW = [("head", None), ("full", None)]
LR = {"head": 1e-3, "lora": 1e-3, "default": 1e-4}
BATCH, EVAL_BATCH, PATIENCE = 12, 32, 2
KNN, TAU = 10, 0.1
CSV = f"{REPO}/results/metrics_phase5.csv"


def fresh():
    return BertForMaskedLM.from_pretrained(MODEL_DIR).to(DEVICE)


def pooled(model, x, attn, layer):
    out = model.bert(input_ids=x, attention_mask=attn, output_hidden_states=layer != -1)
    h = out.last_hidden_state if layer == -1 else out.hidden_states[layer]
    m = attn.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1)


@torch.no_grad()
def embed(model, seqs, ids, pad_id, layer):
    model.eval()
    order = sorted(ids, key=lambda c: len(seqs[c]))
    out = {}
    for s in range(0, len(order), EVAL_BATCH):
        b = order[s:s + EVAL_BATCH]
        x, attn = p1.collate([seqs[c] for c in b], pad_id)
        with torch.autocast("cuda", dtype=torch.float16):
            e = pooled(model, x.to(DEVICE), attn.to(DEVICE), layer).float()
        out.update(zip(b, e.cpu()))
    return out


# ---------- in-context learning (ICL-A) ----------

def icl_probs(method, support_x, support_y, query_x, n_cls):
    """Class probabilities for each query, given a labeled support set. No weights change."""
    s = torch.nn.functional.normalize(support_x.to(DEVICE), dim=1)
    q = torch.nn.functional.normalize(query_x.to(DEVICE), dim=1)
    y = support_y.to(DEVICE)
    out = []
    if method == "knn":
        k = min(KNN, len(s))
        for start in range(0, len(q), 4096):
            sim = q[start:start + 4096] @ s.T
            idx = sim.topk(k, dim=1).indices
            votes = torch.zeros(len(idx), n_cls, device=DEVICE).scatter_add_(1, y[idx], torch.ones_like(idx, dtype=torch.float))
            out.append(((votes + 0.1) / (k + 0.1 * n_cls)).cpu())
    else:
        proto = torch.zeros(n_cls, s.shape[1], device=DEVICE).index_add_(0, y, s)
        present = torch.bincount(y, minlength=n_cls) > 0
        proto = torch.nn.functional.normalize(proto, dim=1)
        logits = (q @ proto.T) / TAU
        logits[:, ~present] = -1e4
        out.append(torch.softmax(logits, 1).cpu())
    return torch.cat(out)


# ---------- O2 fine-tuning ----------

class Classifier(torch.nn.Module):
    def __init__(self, n_cls, config, rank):
        super().__init__()
        self.lm = fresh()
        self.n_trainable_encoder = apply_config(self.lm, "base" if config == "head" else config, rank)
        if config != "lora":  # MLM decoder is unused here, so don't count or train it
            for p in self.lm.cls.parameters():
                p.requires_grad = False
        self.head = torch.nn.Linear(self.lm.config.hidden_size, n_cls).to(DEVICE)
        self.frozen = config == "head"

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, attn):
        return self.head(pooled(self.lm, x, attn, -1).float())


def ft_probs(model, seqs, ids, pad_id):
    model.eval()
    out = {}
    with torch.no_grad():
        order = sorted(ids, key=lambda c: len(seqs[c]))
        for s in range(0, len(order), EVAL_BATCH):
            b = order[s:s + EVAL_BATCH]
            x, attn = p1.collate([seqs[c] for c in b], pad_id)
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.softmax(model(x.to(DEVICE), attn.to(DEVICE)).float(), 1)
            out.update(zip(b, p.cpu()))
    return out


def train_classifier(model, seqs, fit, val, label, pad_id, lr, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    scaler = torch.amp.GradScaler("cuda")
    epochs = 10 if len(fit) > 1000 else 40
    best, best_val, stale, hist = None, np.inf, 0, []
    for epoch in range(epochs):
        model.train()
        if model.frozen:
            model.lm.eval()
        order = rng.permutation(fit)
        tot = 0.0
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH].tolist()
            x, attn = p1.collate([seqs[c] for c in b], pad_id)
            y = torch.as_tensor([label[c] for c in b], device=DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(x.to(DEVICE), attn.to(DEVICE)), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(b)
        probs = ft_probs(model, seqs, val, pad_id)
        vloss = float(np.mean([-np.log(probs[c][label[c]].item() + 1e-12) for c in val]))
        hist.append({"epoch": epoch, "train_loss": tot / len(fit), "val_loss": vloss})
        if vloss < best_val:
            best_val, stale = vloss, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best)
    return hist


# ---------- shared evaluation ----------

def ce(probs, label, ids):
    return {c: float(-np.log(probs[c][label[c]].item() + 1e-12)) for c in ids}


def utility(probs, label, holdout):
    y = np.array([label[c] for c in holdout])
    p = np.array([int(probs[c].argmax()) for c in holdout])
    return f1_score(y, p, average="macro"), float((y == p).mean())


def attack(meta, members_seen, members_unseen, holdout_map, loss, ref_loss, ctype):
    rows = []
    for threat, mmap in (("seen_cells", members_seen), ("same_donor_unseen_cells", members_unseen)):
        mmap = {d: v for d, v in mmap.items() if len(v)}
        if len(mmap) < 2:
            continue
        for name, score in (("raw_loss", {c: -v for c, v in loss.items()}),
                            ("calibrated_loss", {c: ref_loss[c] - loss[c] for c in loss}),
                            ("reference_null", {c: -ref_loss[c] for c in ref_loss})):
            for how in ("mean", "median", "celltype_equal"):
                auc, tpr, _, _ = p1.evaluate(mmap, holdout_map, score, ctype, how)
                rows.append({**meta, "threat_model": threat, "attack": name, "aggregation": how, "auc": auc,
                             "tpr_at_fpr": tpr, "n_member_donors": len(mmap)})
    return rows


def done(dataset, trial, method, k):
    if not os.path.exists(CSV):
        return False
    d = pd.read_csv(CSV)
    return len(d[(d.dataset == dataset) & (d.trial == trial) & (d.method == method) & (d.k.astype(str) == str(k))]) > 0


def run_cohort(data, trial, smoke):
    c = rs.Cohort(data, ND, trial, smoke)
    fit_all = p1.flat(c.cells["fit"])
    classes = sorted({c.ctype[x] for x in fit_all})
    cid = {t: i for i, t in enumerate(classes)}
    usable = lambda ids: [x for x in ids if c.ctype[x] in cid]  # noqa: E731
    label = {x: cid[c.ctype[x]] for x in c.seqs if c.ctype[x] in cid}
    hold_ids = usable(p1.flat(c.cells["holdout"]))
    val_ids = usable(p1.flat(c.cells["validation"]))
    ref_ids = usable(c.null_fit)
    holdout_map = {d: np.array(usable(v.tolist())) for d, v in c.cells["holdout"].items()}
    donor_of = {int(x): d for key in ("fit", "unseen") for d, v in c.cells[key].items() for x in v}
    unseen_ids = usable(p1.flat(c.cells["unseen"]))
    query = sorted(set(usable(fit_all)) | set(unseen_ids) | set(hold_ids))
    print(f"phase5 {data.name} t{trial}: {len(classes)} classes | dropped holdout cells of unseen types: "
          f"{len(p1.flat(c.cells['holdout'])) - len(hold_ids)}", flush=True)

    rng = np.random.default_rng([SEED, trial, 5])
    by_type = {}
    for x in usable(fit_all):
        by_type.setdefault(label[x], []).append(x)
    supports = {}
    for k in (KS if not smoke else [1, "all"]):
        if k == "all":
            supports[k] = usable(fit_all)
        else:
            supports[k] = sorted(int(v) for t, xs in by_type.items() for v in rng.permutation(xs)[:k])

    def member_maps(support):
        sup = set(support)
        donors = sorted({donor_of[x] for x in support})
        seen = {d: np.array([x for x in c.cells["fit"][d] if int(x) in sup]) for d in donors}
        unseen = {d: np.array([x for x in list(c.cells["fit"][d]) + list(c.cells["unseen"].get(d, []))
                               if int(x) not in sup and int(x) in label]) for d in donors}
        return seen, unseen, len(donors)

    base_meta = {"model": "geneformer", "version": "V1-10M", "dataset": data.name, "nd": ND, "trial": trial,
                 "n_classes": len(classes), "seed": SEED}

    # ICL-A
    t0 = time.time()
    enc = fresh()
    emb = embed(enc, c.seqs, sorted(set(query) | set(ref_ids)), data.tok.pad_id, -2)
    del enc
    embed_sec_per_cell = (time.time() - t0) / len(emb)
    X = lambda ids: torch.stack([emb[x] for x in ids])  # noqa: E731
    Y = lambda ids: torch.as_tensor([label[x] for x in ids])  # noqa: E731
    for method in ("knn", "proto"):
        ref = dict(zip(query, icl_probs(method, X(ref_ids), Y(ref_ids), X(query), len(classes))))
        ref_loss = ce(ref, label, query)
        for k, support in supports.items():
            name = f"icl_{method}"
            if not smoke and done(data.name, trial, name, k):
                continue
            t1 = time.time()
            probs = dict(zip(query, icl_probs(method, X(support), Y(support), X(query), len(classes))))
            infer = time.time() - t1
            f1, acc = utility(probs, label, hold_ids)
            seen, unseen, n_donors = member_maps(support)
            meta = {**base_meta, "method": name, "k": k, "n_support": len(support), "n_support_donors": n_donors,
                    "n_trainable": 0, "macro_f1": f1, "accuracy": acc, "train_seconds": 0.0,
                    "infer_ms_per_cell": 1000 * (embed_sec_per_cell + infer / len(query)),
                    "retained": f"{len(support)} support embeddings x {emb[query[0]].numel()} floats"}
            loss = ce(probs, label, query)
            rows = attack(meta, seen, unseen, holdout_map, loss, ref_loss, c.ctype)
            rs_append(rows or [meta])
            save_scores(data.name, trial, name, k, loss, ref_loss, smoke)
            summ = {(r["threat_model"], r["attack"]): round(r["auc"], 3) for r in rows if r["aggregation"] == "mean"}
            print(f"  {name} k={k}: F1 {f1:.3f} acc {acc:.3f} | {n_donors} donors | {summ}", flush=True)

    # O2 fine-tuning: the reference model is the full config trained on auxiliary donors
    ref_model = Classifier(len(classes), "full", None)
    train_classifier(ref_model, c.seqs, ref_ids, val_ids, label, data.tok.pad_id, LR["default"], SEED)
    ref_loss = ce(ft_probs(ref_model, c.seqs, query, data.tok.pad_id), label, query)
    del ref_model
    torch.cuda.empty_cache()
    jobs = [(cfg, r, "all") for cfg, r in FT_ALL] + [(cfg, r, k) for k in KS if k != "all" for cfg, r in FT_FEW]
    if smoke:
        jobs = [("head", None, 1), ("lora", 8, "all")]
    for cfg, rank, k in jobs:
        name = f"ft_{cfg}" + (f"_r{rank}" if rank else "")
        if not smoke and done(data.name, trial, name, k):
            continue
        support = supports[k]
        t1 = time.time()
        model = Classifier(len(classes), cfg, rank)
        hist = train_classifier(model, c.seqs, support, val_ids, label, data.tok.pad_id,
                                LR.get(cfg, LR["default"]), SEED)
        train_sec = time.time() - t1
        t1 = time.time()
        probs = ft_probs(model, c.seqs, query, data.tok.pad_id)
        infer = (time.time() - t1) / len(query)
        f1, acc = utility(probs, label, hold_ids)
        seen, unseen, n_donors = member_maps(support)
        n = model.n_trainable()
        meta = {**base_meta, "method": name, "k": k, "n_support": len(support), "n_support_donors": n_donors,
                "n_trainable": n, "macro_f1": f1, "accuracy": acc, "train_seconds": train_sec,
                "infer_ms_per_cell": 1000 * infer, "epochs": len(hist),
                "retained": f"{n} trained parameters"}
        loss = ce(probs, label, query)
        rows = attack(meta, seen, unseen, holdout_map, loss, ref_loss, c.ctype)
        rs_append(rows or [meta])
        save_scores(data.name, trial, name, k, loss, ref_loss, smoke)
        if not smoke and cfg == "lora":
            os.makedirs(f"{REPO}/checkpoints/phase5", exist_ok=True)
            torch.save({n_: {"A": m.lora_A.detach().cpu(), "B": m.lora_B.detach().cpu()}
                        for n_, m in model.named_modules() if isinstance(m, LoRALinear)}
                       | {"head": model.head.state_dict()},
                       f"{REPO}/checkpoints/phase5/{data.name}_t{trial}_{name}_k{k}.pt")
        summ = {(r["threat_model"], r["attack"]): round(r["auc"], 3) for r in rows if r["aggregation"] == "mean"}
        print(f"  {name} k={k}: F1 {f1:.3f} acc {acc:.3f} | {n:,} params | {train_sec:.0f}s | {summ}", flush=True)
        del model
        torch.cuda.empty_cache()


CSV_TARGET = [CSV]


def rs_append(rows):
    df = pd.DataFrame(rows)
    path = CSV_TARGET[0]
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(f"{path}.tmp", index=False)
    os.replace(f"{path}.tmp", path)


def save_scores(name, trial, method, k, loss, ref_loss, smoke):
    if smoke:
        return
    os.makedirs(f"{REPO}/scores/phase5", exist_ok=True)
    ids = np.array(sorted(loss), dtype=np.int64)
    np.savez_compressed(f"{REPO}/scores/phase5/{name}_{ND}d_t{trial}_{method}_k{k}.npz", cell=ids,
                        loss=np.array([loss[int(x)] for x in ids], dtype=np.float32),
                        ref=np.array([ref_loss[int(x)] for x in ids], dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    if args.smoke:
        CSV_TARGET[0] = f"{REPO}/logs/smoke_phase5.csv"
        if os.path.exists(CSV_TARGET[0]):
            os.remove(CSV_TARGET[0])
        run_cohort(rs.Data("aida"), 1, True)
        print("PHASE 5 SMOKE PASSED", flush=True)
        return
    json.dump({"nd": ND, "trials": TRIALS, "ks": KS, "ft_all": FT_ALL, "ft_few": FT_FEW, "lr": LR, "knn": KNN,
               "tau": TAU, "batch": BATCH, "patience": PATIENCE, "seed": SEED, "embedding_layer": -2,
               "model_revision": rs.MODEL_REVISION}, open(f"{REPO}/results/phase5_meta.json", "w"), indent=2)
    loaded = {n: rs.Data(n) for n in DATASETS}
    for trial in TRIALS:
        for name, data in loaded.items():
            run_cohort(data, trial, False)
        print(f"PHASE 5 TRIAL {trial} COMPLETE", flush=True)
    print("PHASE 5 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
