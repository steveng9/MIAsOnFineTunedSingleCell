"""Read-only audit of the OneK1K symbol-to-Ensembl mapping for the Geneformer V1 vocabulary."""
import collections
import os
import sys

import anndata as ad
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from gf_tokenizer import load_dict  # noqa: E402

tokens = load_dict("token_dictionary_gc30M.pkl")
medians = load_dict("gene_median_dictionary_gc30M.pkl")
name_to_id = load_dict("gene_name_id_dict_gc30M.pkl")

a = ad.read_h5ad(os.path.expanduser("~/data/scMAMAMIA/ok/full_dataset_cleaned.h5ad"), backed="r")
symbols = np.asarray(a.var_names).astype(str)
print("symbols", len(symbols), "| unique", len(set(symbols)))
print("suffixed symbols like NAME.1:", sum(s.rsplit(".", 1)[-1].isdigit() and not s.startswith(("RP", "AC", "AL", "AP", "CT")) for s in symbols))

ids = np.array([name_to_id.get(s, "") for s in symbols])
usable = np.array([g in tokens and g in medians for g in ids])
print("mapped to a usable vocabulary gene", int(usable.sum()))

by_id = collections.defaultdict(list)
for i in np.flatnonzero(usable):
    by_id[ids[i]].append(i)
dups = {g: v for g, v in by_id.items() if len(v) > 1}
print("unique Ensembl IDs", len(by_id), "| IDs hit by more than one symbol", len(dups))
print("examples", [(g, symbols[v].tolist()) for g, v in list(dups.items())[:5]])

idx = np.sort(np.random.default_rng(0).choice(a.n_obs, 5000, replace=False))
per_gene = np.asarray(a.X[idx].sum(0)).ravel()
total = per_gene.sum()
dup_columns = [i for v in dups.values() for i in v]
print(f"share of counts: usable {per_gene[usable].sum() / total:.2%} | in duplicated IDs {per_gene[dup_columns].sum() / total:.2%}")

lost = np.argsort(-(per_gene * ~usable))[:25]
print("highest-count unmapped symbols:")
print([(symbols[i], round(float(per_gene[i] / total * 100), 3)) for i in lost])
mt = np.array([s.startswith("MT-") for s in symbols])
rp = np.array([s.startswith(("RPL", "RPS")) for s in symbols])
print(f"share of counts: MT- {per_gene[mt].sum() / total:.2%} (usable {per_gene[mt & usable].sum() / total:.2%}) | "
      f"RPL/RPS {per_gene[rp].sum() / total:.2%} (usable {per_gene[rp & usable].sum() / total:.2%})")
a.file.close()
