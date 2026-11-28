"""Does matching scGPT's pretraining preprocessing make the base model better? (CPU check)

Two differences from the pretraining pipeline (dev-temp data_collator.py, preprocess.binning,
data/cellxgene/process_allcounts.py):
  ties  pretraining binning uses _digitize(side="one"): tied counts all get the upper (left-digitize)
        bin, deterministically. Our tokenizer spread ties at random between edges.
  cls   the <cls> token's expression value is pad_value (-2) in pretraining; we used 0.
Reports hidden-bin MSE, Pearson, and prediction SD for each combination on AIDA cells.
"""
import os
import sys

import anndata as ad
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [f"{REPO}/src"]
import scgpt_model as sm  # noqa: E402

torch.set_num_threads(8)
a = ad.read_h5ad(os.path.expanduser("~/data/scMAMAMIA/aida/full_dataset_cleaned.h5ad"), backed="r")
tok = sm.ScGPTTokenizer(a.raw.var["feature_name"].to_numpy())
rows = a.raw.X[np.arange(0, 1000000, 10000)].toarray()


def encode(counts, cell, ties, cls_value):
    counts = counts[tok.keep]
    nz = np.flatnonzero(counts > 0)
    x = counts[nz].astype(np.float64)
    rng = np.random.default_rng([0, cell])
    bins = np.quantile(x, np.linspace(0, 1, tok.n_bins - 1))
    left = np.digitize(x, bins)
    if ties == "random":
        right = np.digitize(x, bins, right=True)
        digits = np.ceil(rng.random(len(x)) * (right - left) + left)
    else:
        digits = left
    genes = tok.ids[tok.keep[nz]]
    if len(genes) > tok.max_len - 1:
        pick = np.sort(rng.choice(len(genes), tok.max_len - 1, replace=False))
        genes, digits = genes[pick], digits[pick]
    return np.r_[tok.cls_id, genes], np.r_[cls_value, digits].astype(np.float32)


model = sm.load_scgpt("cpu").eval()
for ties in ("random", "left"):
    for cls_value in (0.0, -2.0):
        cells = [encode(r, i, ties, cls_value) for i, r in enumerate(rows)]
        n = max(len(g) for g, _ in cells)
        genes = torch.full((len(cells), n), tok.pad_id, dtype=torch.long)
        vals = torch.full((len(cells), n), sm.PAD_VALUE)
        for i, (g, v) in enumerate(cells):
            genes[i, :len(g)], vals[i, :len(g)] = torch.as_tensor(g), torch.as_tensor(v)
        pad = genes.eq(tok.pad_id)
        mask = (torch.rand(genes.shape, generator=torch.Generator().manual_seed(0)) < 0.25) & ~pad
        mask[:, 0] = False
        preds = []
        with torch.no_grad():
            for s in range(0, len(cells), 10):
                preds.append(model(genes[s:s + 10], vals[s:s + 10], pad[s:s + 10], gen=mask[s:s + 10]))
        pred = torch.cat(preds)
        cell_mean = torch.stack([vals[i][~pad[i]][1:].mean() for i in range(len(cells))])[:, None].expand_as(vals)
        t, p = vals[mask].numpy(), pred[mask].numpy()
        print(f"ties={ties:6s} cls={cls_value:+.0f} | truth SD {t.std():.2f} | cell-mean MSE "
              f"{((cell_mean - vals) ** 2)[mask].mean():.2f} | model MSE {((pred - vals) ** 2)[mask].mean():.2f} | "
              f"Pearson {np.corrcoef(p, t)[0, 1]:.3f} | pred SD {p.std():.2f}", flush=True)
