"""Gate for Phase 6: the scGPT re-implementation must match the official model.

1. Load the official scgpt TransformerModel (PyTorch attention path) straight from the cloned
   source, without installing the package, and compare decoder outputs on real AIDA cells.
2. LoRA on every q, k, v, out, and feed-forward linear: outputs unchanged at B = 0, counts = r(m+n).
3. Sanity: masked-bin MSE of the base model vs predicting each cell's mean bin.
"""
import json
import os
import sys
import types

import anndata as ad
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [f"{REPO}/src"]
import scgpt_model as sm  # noqa: E402

SRC = f"{REPO}/external/scGPT/scgpt"
for name, path in (("scgpt", SRC), ("scgpt.model", f"{SRC}/model")):
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    sys.modules[name] = mod
from scgpt.model.model import TransformerModel  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
args = json.load(open(f"{sm.SCGPT_DIR}/args.json"))
vocab = sm.load_vocab()

ref = TransformerModel(len(vocab), args["embsize"], args["nheads"], args["d_hid"], args["nlayers"],
                       vocab=vocab, dropout=args["dropout"], pad_token="<pad>", pad_value=args["pad_value"],
                       do_mvc=True, input_emb_style="continuous", use_fast_transformer=False)
sd = torch.load(f"{sm.SCGPT_DIR}/best_model.pt", map_location="cpu", weights_only=True)
sd = {k.replace("Wqkv.weight", "in_proj_weight").replace("Wqkv.bias", "in_proj_bias"): v for k, v in sd.items()}
missing, unexpected = ref.load_state_dict(sd, strict=False)
print("official model: missing", missing, "| unexpected", unexpected)
ref = ref.to(dev).eval()
mine = sm.load_scgpt(dev).eval()

a = ad.read_h5ad(os.path.expanduser("~/data/scMAMAMIA/aida/full_dataset_cleaned.h5ad"), backed="r")
tok = sm.ScGPTTokenizer(a.raw.var["feature_name"].to_numpy())
print(f"usable genes {len(tok.keep):,}")
rows = a.raw.X[np.arange(0, 400000, 25000)].toarray()
cells = [tok.encode(r, i) for i, r in enumerate(rows)]
print("example cell: tokens", len(cells[0][0]), "| first genes", cells[0][0][:5].tolist(),
      "| bins", cells[0][1][:5].tolist(), "| max bin", max(int(c[1].max()) for c in cells))
assert tok.encode(rows[0], 0)[1].tolist() == cells[0][1].tolist(), "tokenizer is not deterministic"


def batch(cells, mask_frac=0.25, seed=0):
    n = max(len(g) for g, _ in cells)
    genes = torch.full((len(cells), n), tok.pad_id, dtype=torch.long)
    vals = torch.full((len(cells), n), sm.PAD_VALUE)
    for i, (g, v) in enumerate(cells):
        genes[i, :len(g)], vals[i, :len(g)] = torch.as_tensor(g, dtype=torch.long), torch.as_tensor(v, dtype=torch.float)
    pad = genes.eq(tok.pad_id)
    gen = torch.Generator().manual_seed(seed)
    mask = (torch.rand(genes.shape, generator=gen) < mask_frac) & ~pad
    mask[:, 0] = False
    return genes.to(dev), vals.to(dev), vals.masked_fill(mask, sm.MASK_VALUE).to(dev), pad.to(dev), mask.to(dev)


genes, vals, masked, pad, mask = batch(cells)
with torch.no_grad():
    r = ref(genes, masked, pad)["mlm_output"]
    m = mine(genes, masked, pad)
diff = (r - m)[~pad].abs().max().item()
print(f"max |official - mine| on real tokens: {diff:.2e}")
assert diff < 1e-3, "re-implementation does not match the official model"

for rank in (1, 8):
    model = sm.load_scgpt(dev).eval()
    n = sm.apply_config(model, "lora", rank)
    with torch.no_grad():
        same = (model(genes, masked, pad) - m)[~pad].abs().max().item()
    print(f"LoRA r={rank}: trainable {n:,} (= {n // rank:,} r) | max change at B=0 {same:.1e}")
    assert same == 0.0
for config in ("head", "last1", "last2", "last4", "full"):
    print(config, f"{sm.apply_config(sm.load_scgpt(dev), config):,}")

with torch.no_grad():
    cell_mean = torch.stack([vals[i][~pad[i]][1:].mean() for i in range(len(cells))])[:, None].expand_as(vals)
    base_mse = ((cell_mean - vals) ** 2)[mask].mean().item()
    mlm_mse = ((m - vals) ** 2)[mask].mean().item()
    gen_pred = mine(genes, vals, pad, gen=mask)
    gen_mse = ((gen_pred - vals) ** 2)[mask].mean().item()
    r_gen = np.corrcoef(gen_pred[mask].cpu().numpy(), vals[mask].cpu().numpy())[0, 1]
print(f"hidden-bin MSE: cell-mean baseline {base_mse:.2f} | masked-value input {mlm_mse:.2f} | "
      f"generative input {gen_mse:.2f} (Pearson {r_gen:.3f}, prediction SD {gen_pred[mask].std().item():.2f})")
# The base model predicts nearly constant bins (MSE is no better than a per-cell mean), so the gate
# is that predictions carry signal at all. Fine-tuning sensitivity is tested by the positive control.
assert r_gen > 0.2, "scGPT predictions are uncorrelated with the truth; check tokenization"
print("SCGPT GATE PASSED")
