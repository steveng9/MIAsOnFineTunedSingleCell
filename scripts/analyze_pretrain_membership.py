"""Covariate-adjusted re-analysis of exp_pretrain_membership.py (CPU, saved per-cell losses).

The raw comparison is dominated by technical differences between the v1 and v2-only sequencing
batches (Geneformer, which saw neither group, separates them at AUC 0.86). Here each model's
per-cell loss is regressed, within cell type, on log total counts, log genes detected, and
mitochondrial fraction (fit on all scored cells). Residuals are averaged per donor and compared.
If residual AUCs for Geneformer move to about 0.5 while scGPT's stay away from it, that points to
pretraining membership; if both move to 0.5, the raw differences were technical.
Output: results/pretrain_membership_adjusted.csv.
"""
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, f"{REPO}/src"]
import exp_pretrain_membership as e  # noqa: E402
import run_sweep as rs  # noqa: E402

gf = rs.Data("aida_v2", path=e.V2, donor_col="donor_id")
cells, member, batch = e.design(gf)
obs = gf.adata.obs
cov = pd.DataFrame({"lc": np.log(obs["nCount_RNA"].astype(float).to_numpy()),
                    "lf": np.log(obs["nFeature_RNA"].astype(float).to_numpy()),
                    "mt": obs["pMito"].astype(float).to_numpy()})
rows = []
for tag in ("geneformer", "scgpt", "scgpt_ties_left"):
    f = f"{REPO}/scores/pretrain/aida_v2_sg_{tag}.npz"
    if not os.path.exists(f):
        continue
    z = np.load(f)
    ids = z["cell"]
    df = cov.iloc[ids].assign(loss=z["loss"], ct=gf.ctypes[ids]).set_index(ids)
    for how in ("raw_type_centered", "covariate_adjusted"):
        if how == "raw_type_centered":
            res = df.loss - df.groupby("ct").loss.transform("mean")
        else:
            res = pd.Series(np.nan, index=df.index)
            for _, g in df.groupby("ct"):
                X = np.c_[np.ones(len(g)), g[["lc", "lf", "mt"]].to_numpy()]
                beta, *_ = np.linalg.lstsq(X, g.loss.to_numpy(), rcond=None)
                res.loc[g.index] = g.loss.to_numpy() - X @ beta
        don = {d: -res.loc[v].mean() for d, v in cells.items()}
        y = np.array([member[d] for d in don], int)
        s = np.array(list(don.values()))
        wb, pairs = e.within_batch_auc(don, member, batch)
        rows.append({"model": tag, "score": how, "auc": roc_auc_score(y, s), "within_batch_auc": wb,
                     "within_batch_pairs": pairs, "n_members": int(y.sum()), "n_nonmembers": int((1 - y).sum())})
out = pd.DataFrame(rows)
out.to_csv(f"{REPO}/results/pretrain_membership_adjusted.csv", index=False)
print(out.round(3).to_string())
tech = pd.DataFrame({"member": [member[d] for d in cells],
                     "log_counts": [cov.lc.iloc[v].mean() for v in cells.values()],
                     "log_genes": [cov.lf.iloc[v].mean() for v in cells.values()],
                     "pct_mito": [cov.mt.iloc[v].mean() for v in cells.values()]})
print(tech.groupby("member").mean().round(3).to_string())
