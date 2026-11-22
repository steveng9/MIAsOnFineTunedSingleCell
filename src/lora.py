"""A minimal LoRA layer: freeze W and learn W_new = W + B @ A.

For W of shape (m, n), B is (m, r) and A is (r, n), so the adapter trains
r * (m + n) parameters instead of m * n. PyTorch stores a Linear weight as
(out_features, in_features), so m = out_features and n = in_features.
"""
import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int):
        super().__init__()
        m, n = base.weight.shape
        assert 0 < r <= min(m, n), f"rank {r} invalid for shape {(m, n)}"
        self.base, self.r = base, r
        for p in self.base.parameters():
            p.requires_grad = False
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Parameter(torch.empty(r, n, **factory))
        self.lora_B = nn.Parameter(torch.zeros(m, r, **factory))  # B = 0, so the model starts unchanged
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T) @ self.lora_B.T

    def delta(self):
        return self.lora_B @ self.lora_A


def encoder_linear_names(model):
    """Names of every nn.Linear inside the transformer encoder."""
    return [n for n, mod in model.named_modules() if n.startswith("bert.encoder.") and isinstance(mod, nn.Linear)]


def inject_lora(model, r):
    """Freeze the model and wrap each encoder Linear. Returns [(name, m, n)]."""
    for p in model.parameters():
        p.requires_grad = False
    wrapped = []
    for name in encoder_linear_names(model):
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        base = getattr(parent, child)
        m, n = base.weight.shape
        setattr(parent, child, LoRALinear(base, r))
        wrapped.append((name, m, n))
    return wrapped


def n_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_config(model, config, rank=None):
    """Set requires_grad for one experimental config and return the trainable count."""
    if config == "lora":
        wrapped = inject_lora(model, rank)
        expected = sum(rank * (m + n) for _, m, n in wrapped)
        assert n_trainable(model) == expected, "LoRA parameter count does not match r(m+n)"
        return expected

    for p in model.parameters():
        p.requires_grad = False
    if config == "base":
        pass
    elif config == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif config == "head":
        # The decoder weight is tied to the gene embeddings, so named_parameters lists it
        # under bert.embeddings and it stays frozen here.
        for name, p in model.named_parameters():
            if name.startswith("cls."):
                p.requires_grad = True
        assert not model.bert.embeddings.word_embeddings.weight.requires_grad
    elif config.startswith("last"):
        k = int(config[4:])
        for layer in model.bert.encoder.layer[-k:]:
            for p in layer.parameters():
                p.requires_grad = True
    else:
        raise ValueError(config)
    return n_trainable(model)
