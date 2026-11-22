"""Phase 0 gate: one real AIDA cell through the tokenizer and Geneformer V1."""
import os
import sys

import anndata as ad
import numpy as np
import torch
from transformers import BertForMaskedLM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from gf_tokenizer import MODEL_DIR, MODEL_REVISION, GeneformerV1Tokenizer  # noqa: E402

SEED = 0
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

a = ad.read_h5ad(os.path.expanduser("~/data/scMAMAMIA/aida/full_dataset_cleaned.h5ad"), backed="r")
tok = GeneformerV1Tokenizer(a.raw.var_names)
cell = int(rng.integers(a.n_obs))
counts = a.raw.X[[cell]].toarray().ravel()
ids = tok.encode(counts)
symbols = a.raw.var["feature_name"].astype(str).to_dict() if "feature_name" in a.raw.var else {}
id_to_gene = dict(zip(tok.token_ids.tolist(), tok.genes.tolist()))
print(f"cell index {cell} | type {a.obs['cell_type'].iloc[cell]}")
print(f"nonzero genes {int((counts > 0).sum())} | total counts {counts.sum():.0f} | tokens {len(ids)}")
print("first 10 token ids", ids[:10].tolist())
print("first 10 genes", [symbols.get(id_to_gene[t], id_to_gene[t]) for t in ids[:10].tolist()])
assert len(ids) > 0 and len(ids) <= tok.max_len and ids.min() >= 2 and ids.max() < tok.vocab_size

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BertForMaskedLM.from_pretrained(MODEL_DIR).to(device).eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"revision {MODEL_REVISION} | parameters {n_params:,} | device {device}")

x = torch.tensor(ids, device=device).unsqueeze(0)
mask_pos = len(ids) // 2
masked = x.clone()
masked[0, mask_pos] = tok.mask_id
with torch.no_grad():
    out = model(input_ids=masked, attention_mask=torch.ones_like(x), output_hidden_states=True)
print("input", tuple(x.shape), "| logits", tuple(out.logits.shape),
      "| hidden states", len(out.hidden_states), "x", tuple(out.hidden_states[-1].shape))
top = out.logits[0, mask_pos].topk(5).indices.tolist()
true_id = int(x[0, mask_pos])
rank = int((out.logits[0, mask_pos] > out.logits[0, mask_pos, true_id]).sum()) + 1
print(f"masked position {mask_pos}: true token {true_id}, model rank of true token {rank} of {tok.vocab_size}, top5 {top}")
assert out.logits.shape == (1, len(ids), tok.vocab_size)
print("PHASE 0 GATE PASSED")
a.file.close()
