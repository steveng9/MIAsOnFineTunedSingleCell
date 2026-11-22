"""Measure how well each dataset's genes map onto the Geneformer V1 (gc30M) vocabulary.

Dictionaries are loaded with an unpickler that refuses all imports, so a pickle
from the download cannot execute code. Source datasets are opened read-only.
"""
import io
import json
import os
import pickle

import anndata as ad
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GF = f"{REPO}/external/geneformer"
DICTS = f"{GF}/geneformer/gene_dictionaries_30m"
DATA = os.path.expanduser("~/data/scMAMAMIA")


class NoImports(pickle.Unpickler):
    """Allow only the two numpy constructors needed for scalar dictionary values."""

    def find_class(self, module, name):
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "scalar":
            return np._core.multiarray.scalar
        if module == "numpy" and name == "dtype":
            return np.dtype
        raise pickle.UnpicklingError(f"blocked import: {module}.{name}")


def load(path):
    with open(path, "rb") as f:
        return NoImports(io.BytesIO(f.read())).load()


tokens = load(f"{DICTS}/token_dictionary_gc30M.pkl")
name_to_id = load(f"{DICTS}/gene_name_id_dict_gc30M.pkl")
print("vocab size", len(tokens), "| special", [k for k in tokens if str(k).startswith("<")])
print("V1 config", json.load(open(f"{GF}/Geneformer-V1-10M/config.json")))

for name in ["ok", "aida", "cg"]:
    a = ad.read_h5ad(f"{DATA}/{name}/full_dataset_cleaned.h5ad", backed="r")
    genes = a.raw.var_names if a.raw is not None else a.var_names
    genes = np.asarray(genes).astype(str)
    ids = genes if genes[0].startswith("ENSG") else np.array([name_to_id.get(g, "") for g in genes])
    in_vocab = np.array([g in tokens for g in ids])

    # Share of a sample of counts that lands on genes the model can represent.
    idx = np.sort(np.random.default_rng(0).choice(a.n_obs, 2000, replace=False))
    x = (a.raw.X if a.raw is not None else a.X)[idx]
    per_gene = np.asarray(x.sum(0)).ravel()
    print(f"{name}: genes {len(genes)} | mapped to vocab {in_vocab.sum()} ({in_vocab.mean():.1%})"
          f" | counts on mapped genes {per_gene[in_vocab].sum() / per_gene.sum():.1%}")
    a.file.close()
