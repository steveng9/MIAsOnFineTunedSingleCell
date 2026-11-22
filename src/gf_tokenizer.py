"""Geneformer V1 (gc30M) rank-value tokenization, written to be read.

A cell becomes a gene sequence in four steps:
1. scale raw counts to 10,000 total per cell
2. divide each gene by its corpus-wide nonzero median
3. drop zero and out-of-vocabulary genes
4. sort genes by normalized value (descending) and keep the top `max_len`
"""
import io
import os
import pickle

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GF_DIR = f"{REPO}/external/geneformer"
DICT_DIR = f"{GF_DIR}/geneformer/gene_dictionaries_30m"
MODEL_DIR = f"{GF_DIR}/Geneformer-V1-10M"
MODEL_REVISION = "1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5"
TARGET_SUM = 10_000
MAX_LEN = 2048


class _SafeUnpickler(pickle.Unpickler):
    """Allow only the numpy constructors needed for scalar dictionary values."""

    def find_class(self, module, name):
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "scalar":
            return np._core.multiarray.scalar
        if module == "numpy" and name == "dtype":
            return np.dtype
        raise pickle.UnpicklingError(f"blocked import: {module}.{name}")


def load_dict(filename):
    with open(f"{DICT_DIR}/{filename}", "rb") as f:
        return _SafeUnpickler(io.BytesIO(f.read())).load()


class GeneformerV1Tokenizer:
    def __init__(self, ensembl_ids, max_len=MAX_LEN):
        token_dict = load_dict("token_dictionary_gc30M.pkl")
        medians = load_dict("gene_median_dictionary_gc30M.pkl")
        self.pad_id = int(token_dict["<pad>"])
        self.mask_id = int(token_dict["<mask>"])
        self.vocab_size = len(token_dict)
        self.max_len = max_len

        ids = np.asarray(ensembl_ids).astype(str)
        keep = np.array([g in token_dict and g in medians for g in ids])
        self.keep_idx = np.flatnonzero(keep)
        self.genes = ids[self.keep_idx]
        self.token_ids = np.array([int(token_dict[g]) for g in self.genes], dtype=np.int64)
        self.medians = np.array([float(medians[g]) for g in self.genes], dtype=np.float64)

    def encode(self, counts):
        """Tokenize one cell from a dense 1D vector of raw counts over all dataset genes."""
        counts = np.asarray(counts, dtype=np.float64).ravel()
        total = counts.sum()
        if total <= 0:
            return np.empty(0, dtype=np.int64)
        values = counts[self.keep_idx] / total * TARGET_SUM / self.medians
        nz = np.flatnonzero(values > 0)
        # Stable sort, so tied genes keep a deterministic order.
        order = nz[np.argsort(-values[nz], kind="stable")][: self.max_len]
        return self.token_ids[order]
