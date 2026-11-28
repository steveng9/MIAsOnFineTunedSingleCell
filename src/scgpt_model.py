"""scGPT whole-human, re-implemented so LoRA can attach to every projection.

The checkpoint packs query, key, and value into one `Wqkv` matrix (flash-attn layout, rows
ordered q, k, v). Here they are split into three `nn.Linear` layers, so `lora.LoRALinear` can wrap
them like any other linear layer. `scripts/test_scgpt.py` checks the outputs against the official
`scgpt` TransformerModel.

Objective O1 for scGPT is the objective this checkpoint was pretrained with (args.json:
USE_GENERATIVE_TRAINING, from the `dev-temp` branch's FlashscGPTLayer). A random subset of genes is
"generated": those positions get the gene embedding plus a flag embedding instead of a value, and
the decoder predicts their bins. Known genes attend only to known genes, and each generated gene
attends to all known genes plus itself. The loss is mean squared error on generated bins. The
checkpoint never saw masked values (-1) as inputs, which is why plain masked-value prediction
scored worse than a per-cell mean baseline.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCGPT_DIR = f"{REPO}/models/scgpt_whole_human"
SCGPT_SOURCE = "Google Drive folder 1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y (bowang-lab/scGPT README, whole-human)"
SCGPT_SHA256 = "6cb5d451ab5c4b33eb673adbe4fddc61d2389df1b89b7651a9fe2e557572b922"  # best_model.pt
MASK_VALUE, PAD_VALUE = -1.0, -2.0


class Attention(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.query, self.key, self.value = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x, allowed):
        """allowed: bool (batch, 1, query, key), True where attention is permitted."""
        b, n, d = x.shape
        split = lambda t: t.view(b, n, self.heads, d // self.heads).transpose(1, 2)  # noqa: E731
        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.out_proj(out.transpose(1, 2).reshape(b, n, d))


class Layer(nn.Module):
    """Post-norm encoder layer, as in scGPT's FlashTransformerEncoderLayer."""

    def __init__(self, d, heads, d_hid, dropout):
        super().__init__()
        self.self_attn = Attention(d, heads, dropout)
        self.linear1, self.linear2 = nn.Linear(d, d_hid), nn.Linear(d_hid, d)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, allowed):
        x = self.norm1(x + self.drop(self.self_attn(x, allowed)))
        return self.norm2(x + self.drop(self.linear2(self.drop(F.relu(self.linear1(x))))))


class ValueEncoder(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.linear1, self.linear2 = nn.Linear(1, d), nn.Linear(d, d)
        self.norm, self.drop = nn.LayerNorm(d), nn.Dropout(dropout)

    def forward(self, v):
        v = torch.clamp(v.unsqueeze(-1), max=512)
        return self.drop(self.norm(self.linear2(F.relu(self.linear1(v)))))


class ScGPT(nn.Module):
    def __init__(self, ntoken, d=512, heads=8, d_hid=512, layers=12, dropout=0.2, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(ntoken, d, padding_idx=pad_id)
        self.enc_norm = nn.LayerNorm(d)
        self.value_encoder = ValueEncoder(d, dropout)
        self.layers = nn.ModuleList(Layer(d, heads, d_hid, dropout) for _ in range(layers))
        self.flag_encoder = nn.Embedding(2, d)
        self.decoder = nn.Sequential(nn.Linear(d, d), nn.LeakyReLU(), nn.Linear(d, d), nn.LeakyReLU(),
                                     nn.Linear(d, 1))

    def encode(self, genes, values, pad, gen=None, upto=None):
        """gen: bool (batch, seq), True for genes whose value is hidden and predicted."""
        x = self.enc_norm(self.embedding(genes))
        allowed = (~pad)[:, None, None, :]
        if gen is None:
            x = x + self.value_encoder(values)
        else:
            x = x + torch.where(gen[..., None], self.flag_encoder.weight[1], self.value_encoder(values))
            eye = torch.eye(genes.shape[1], dtype=torch.bool, device=genes.device)
            allowed = allowed & ((~gen)[:, None, None, :] | eye)
        for layer in self.layers[:upto]:
            x = layer(x, allowed)
        return x

    def forward(self, genes, values, pad, gen=None):
        return self.decoder(self.encode(genes, values, pad, gen)).squeeze(-1)


def load_vocab():
    return json.load(open(f"{SCGPT_DIR}/vocab.json"))


def load_scgpt(device="cuda"):
    args = json.load(open(f"{SCGPT_DIR}/args.json"))
    vocab = load_vocab()
    model = ScGPT(len(vocab), args["embsize"], args["nheads"], args["d_hid"], args["nlayers"], args["dropout"],
                  vocab[args["pad_token"]])
    sd = torch.load(f"{SCGPT_DIR}/best_model.pt", map_location="cpu", weights_only=True)
    out = {"embedding.weight": sd["encoder.embedding.weight"],
           "enc_norm.weight": sd["encoder.enc_norm.weight"], "enc_norm.bias": sd["encoder.enc_norm.bias"]}
    for k, v in sd.items():
        if k.startswith("value_encoder.") or k == "flag_encoder.weight":
            out[k] = v
        elif k.startswith("decoder.fc."):
            out["decoder." + k[len("decoder.fc."):]] = v
        elif k.startswith("transformer_encoder.layers."):
            k = k[len("transformer_encoder."):]
            if ".self_attn.Wqkv." in k:
                kind = k.rsplit(".", 1)[1]
                for name, part in zip(("query", "key", "value"), v.chunk(3, dim=0)):
                    out[k.replace("Wqkv." + kind, f"{name}.{kind}")] = part.clone()
            else:
                out[k] = v
    model.load_state_dict(out, strict=True)
    return model.to(device)


def encoder_linear_names(model):
    """Every linear layer inside the transformer blocks (q, k, v, out, two feed-forward)."""
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.startswith("layers.")]


class ScGPTTokenizer:
    """Nonzero in-vocabulary genes, binned into 51 levels by within-cell quantiles.

    Binning is invariant to per-cell scaling and log1p, so raw counts give the same bins as
    scGPT's normalize-log-bin pipeline. Tied counts: scGPT pretraining puts them all in the upper
    bin (ties="left"); ties="random" spreads them between edges, which is what Phase 6 used (an
    error found 2026-11-28). Random draws are seeded by cell ID, so a cell always gets the same
    tokens. Cells with more than 1,199 genes are
    subsampled (seeded by cell ID) to fit 1,200 tokens with the <cls> token first.
    """

    def __init__(self, gene_names, n_bins=51, max_len=1200, seed=0, ties="random"):
        vocab = load_vocab()
        names = np.asarray(gene_names).astype(str)
        dup = set(names[pd_duplicated(names)])
        self.ids = np.array([vocab.get(g, -1) if g not in dup else -1 for g in names], dtype=np.int64)
        self.keep = np.flatnonzero(self.ids >= 0)
        self.n_bins, self.max_len, self.seed = n_bins, max_len, seed
        # "left" matches scGPT pretraining (preprocess._digitize side="one": ties share the upper bin).
        # "random" spreads ties between edges; Phases 6 and the fixed-epoch run used it.
        assert ties in ("random", "left")
        self.ties = ties
        self.cls_id, self.pad_id = vocab["<cls>"], vocab["<pad>"]

    def encode(self, counts, cell_id):
        counts = counts[self.keep]
        nz = np.flatnonzero(counts > 0)
        x = counts[nz].astype(np.float64)
        rng = np.random.default_rng([self.seed, int(cell_id)])
        bins = np.quantile(x, np.linspace(0, 1, self.n_bins - 1))
        left, right = np.digitize(x, bins), np.digitize(x, bins, right=True)
        draws = rng.random(len(x))
        digits = (left if self.ties == "left" else np.ceil(draws * (right - left) + left)).astype(np.int8)
        genes = self.ids[self.keep[nz]]
        if len(genes) > self.max_len - 1:
            pick = np.sort(rng.choice(len(genes), self.max_len - 1, replace=False))
            genes, digits = genes[pick], digits[pick]
        return np.r_[self.cls_id, genes].astype(np.int32), np.r_[0, digits].astype(np.int8)


def pd_duplicated(names):
    _, inv, counts = np.unique(names, return_inverse=True, return_counts=True)
    return counts[inv] > 1


def apply_config(model, config, rank=None):
    """scGPT version of lora.apply_config. Head = the expression decoder (3 linear layers)."""
    from lora import LoRALinear, n_trainable
    for p in model.parameters():
        p.requires_grad = False
    if config == "lora":
        expected = 0
        for name in encoder_linear_names(model):
            parent_name, child = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            base = getattr(parent, child)
            m, n = base.weight.shape
            setattr(parent, child, LoRALinear(base, rank))
            expected += rank * (m + n)
        assert n_trainable(model) == expected, "LoRA parameter count does not match r(m+n)"
        return expected
    if config == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif config == "head":
        for p in model.decoder.parameters():
            p.requires_grad = True
    elif config.startswith("last"):
        for layer in model.layers[-int(config[4:]):]:
            for p in layer.parameters():
                p.requires_grad = True
    elif config != "base":
        raise ValueError(config)
    return n_trainable(model)
