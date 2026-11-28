"""Pretraining membership on base models, using AIDA Data Freeze v2 (no fine-tuning).

scGPT whole-human was pretrained on the CELLxGENE Census of May 2023, which held AIDA Data
Freeze v1. v2 adds donors released later. Among Singapore donors (same country, assay, and study):
  members      donors in v1 (scored only on their v1 cells, matched by barcode and library)
  non-members  donors added in v2
Geneformer V1 saw neither group (Genecorpus-30M, 2021), so its AUC measures everything that is
not pretraining membership (sequencing batch, time, composition). The quantity of interest is the
scGPT AUC minus the Geneformer AUC on the same cells (difference-in-differences), with a donor
bootstrap. A within-batch AUC uses only member/non-member pairs from the same sequencing batch.

Score per cell = -(base-model loss), centered within cell type, averaged per donor (up to 200 cells).
Outputs: results/pretrain_membership.csv (aggregates), scores/pretrain/*.npz (local only).
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, f"{REPO}/src"]
import phase1_pilot as p1  # noqa: E402
import run_scgpt as r6  # noqa: E402
import run_sweep as rs  # noqa: E402
import scgpt_model as sm  # noqa: E402

V1 = "~/data/scMAMAMIA/aida/full_dataset_cleaned.h5ad"
V2 = "~/data/scMAMAMIA/aida/full_dataset_v2.h5ad"
CAP, SEED, N_BOOT, N_PERM = 200, 0, 2000, 1000
OUT = f"{REPO}/results/pretrain_membership.csv"
TIES = "left" if "--ties-left" in sys.argv else "random"
if TIES == "left":
    OUT = f"{REPO}/results/pretrain_membership_ties_left.csv"


def v1_to_v2(name):
    bc, lib = name.split("-", 1)
    m = re.match(r"SG_B(\d+)_L(\d+)$", lib)
    return f"{bc}-SG_HEL_B{int(m[1]):03d}_L{int(m[2]):03d}" if m else None


def design(gf):
    """Pick cells: v2 Singapore donors; members restricted to their v1 cells."""
    import anndata as ad
    v1 = ad.read_h5ad(os.path.expanduser(V1), backed="r").obs
    v1_cells = {v1_to_v2(x) for x in v1.index[v1.index.str.contains("-SG_")]} - {None}
    v1_donors = set(v1["individual"].astype(str))
    obs = gf.adata.obs
    names = obs.index.to_numpy()
    batch = obs.index.str.extract(r"-(SG_HEL_B\d+)_L")[0].to_numpy()
    sg = np.flatnonzero(obs["Country"].astype(str).to_numpy() == "SG")
    rng = np.random.default_rng([SEED, 7])
    cells, member, donor_batch = {}, {}, {}
    for d in sorted(set(gf.donors[sg])):
        idx = sg[gf.donors[sg] == d]
        is_member = d in v1_donors
        if is_member:
            idx = idx[np.fromiter((names[i] in v1_cells for i in idx), bool, len(idx))]
        if len(idx) < 20:
            continue
        cells[d] = rng.permutation(idx)[:CAP]
        member[d] = is_member
        donor_batch[d] = pd.Series(batch[cells[d]]).mode()[0]
    print(f"members {sum(member.values())} | non-members {sum(not v for v in member.values())} | "
          f"cells {sum(len(v) for v in cells.values())}", flush=True)
    return cells, member, donor_batch


def donor_auc(score, cells, member, ctype):
    ids = np.concatenate(list(cells.values()))
    s = pd.Series(score[ids], index=ids)
    s = s - s.groupby(ctype[ids]).transform("mean")
    don = {d: s.loc[v].mean() for d, v in cells.items()}
    return don


def auc_of(don, member, keep=None):
    ds = [d for d in don if keep is None or d in keep]
    y = np.array([member[d] for d in ds], dtype=int)
    return roc_auc_score(y, [don[d] for d in ds])


def within_batch_auc(don, member, batch):
    wins = total = 0.0
    for b in set(batch.values()):
        pos = [don[d] for d in don if batch[d] == b and member[d]]
        neg = [don[d] for d in don if batch[d] == b and not member[d]]
        for p in pos:
            for n in neg:
                wins += 1.0 if p > n else 0.5 if p == n else 0.0
                total += 1
    return wins / total if total else np.nan, int(total)


def main():
    gf = rs.Data("aida_v2", path=V2, donor_col="donor_id")
    cells, member, batch = design(gf)
    ids = np.sort(np.concatenate(list(cells.values())))
    ctype = gf.ctypes
    os.makedirs(f"{REPO}/scores/pretrain", exist_ok=True)

    losses = {}
    saved = f"{REPO}/scores/pretrain/aida_v2_sg_geneformer.npz"
    if TIES == "left" and os.path.exists(saved):
        z = np.load(saved)
        assert np.array_equal(z["cell"], ids), "cell selection changed"
        losses["geneformer"] = dict(zip(z["cell"].tolist(), z["loss"].tolist()))
    else:
        seqs = gf.tokenize(ids)
        model = rs.fresh()
        losses["geneformer"] = p1.cell_losses(model, seqs, ids.tolist(), gf.tok)
        del model, seqs
        torch.cuda.empty_cache()

    r6.TIES = TIES
    sc = r6.ScData("aida_v2", path=V2, donor_col="donor_id")
    seqs = sc.tokenize(ids)
    model = sm.load_scgpt(r6.DEVICE)
    losses["scgpt"] = r6.cell_losses(model, seqs, ids.tolist(), sc.tok.pad_id)
    del model, seqs
    torch.cuda.empty_cache()

    n = int(ids.max()) + 1
    don = {}
    for m, l in losses.items():
        score = np.full(n, np.nan)
        score[ids] = [-l[int(c)] for c in ids]
        tag = m if (m == "geneformer" or TIES == "random") else f"{m}_ties_left"
        np.savez_compressed(f"{REPO}/scores/pretrain/aida_v2_sg_{tag}.npz", cell=ids,
                            loss=np.array([l[int(c)] for c in ids], dtype=np.float32))
        don[m] = donor_auc(score, cells, member, ctype)

    rows = []
    rng = np.random.default_rng([SEED, 11])
    mem = [d for d in member if member[d]]
    non = [d for d in member if not member[d]]
    for m in don:
        perm = []
        y = np.array([member[d] for d in don[m]], dtype=int)
        vals = np.array(list(don[m].values()))
        for _ in range(N_PERM):
            perm.append(roc_auc_score(rng.permutation(y), vals))
        wb, pairs = within_batch_auc(don[m], member, batch)
        rows.append({"model": m, "auc": auc_of(don[m], member), "perm_sd": float(np.std(perm)),
                     "within_batch_auc": wb, "within_batch_pairs": pairs,
                     "n_members": len(mem), "n_nonmembers": len(non)})
    boot = []
    for _ in range(N_BOOT):
        keep = list(rng.choice(mem, len(mem))) + list(rng.choice(non, len(non)))
        y = np.array([member[d] for d in keep], dtype=int)
        a = [roc_auc_score(y, [don[m][d] for d in keep]) for m in ("scgpt", "geneformer")]
        boot.append(a[0] - a[1])
    diff = rows[1]["auc"] - rows[0]["auc"] if rows[0]["model"] == "geneformer" else rows[0]["auc"] - rows[1]["auc"]
    rows.append({"model": "scgpt_minus_geneformer", "auc": diff, "ci_low": float(np.percentile(boot, 2.5)),
                 "ci_high": float(np.percentile(boot, 97.5)), "n_members": len(mem), "n_nonmembers": len(non)})
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).to_string(), flush=True)
    print("PRETRAIN MEMBERSHIP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
