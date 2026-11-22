"""Checks LoRALinear against the formula, the base model, and HuggingFace PEFT."""
import os
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import BertForMaskedLM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from gf_tokenizer import MODEL_DIR  # noqa: E402
from lora import LoRALinear, apply_config, encoder_linear_names, inject_lora, n_trainable  # noqa: E402

DEVICE = "cuda"
RANKS = [1, 2, 4, 8, 16, 32, 64]


def fresh():
    return BertForMaskedLM.from_pretrained(MODEL_DIR).to(DEVICE).eval()


torch.manual_seed(0)
x = torch.randint(2, 25426, (4, 128), device=DEVICE)
attn = torch.ones_like(x)
with torch.no_grad():
    base_out = fresh()(input_ids=x, attention_mask=attn).logits

model = fresh()
print("cls parameters:", [(n, tuple(p.shape)) for n, p in model.named_parameters() if n.startswith("cls.")])
names = encoder_linear_names(model)
print("encoder Linear modules:", len(names), "| layer 0 shapes:",
      [(n.split("layer.0.")[1], tuple(model.get_submodule(n).weight.shape)) for n in names if ".layer.0." in n])

print(f"\n{'config':<10}{'trainable':>12}{'% of full':>11}")
full = sum(p.numel() for p in model.parameters())
for cfg in ["base", "head", "last1", "last2", "last4", "full"]:
    m = fresh()
    count = apply_config(m, cfg)
    print(f"{cfg:<10}{count:>12,}{100 * count / full:>10.2f}%")

for r in RANKS:
    custom = fresh()
    count = apply_config(custom, "lora", r)
    with torch.no_grad():
        out = custom(input_ids=x, attention_mask=attn).logits
    assert torch.equal(out, base_out), f"custom LoRA r={r} changed outputs at init"

    peft_model = get_peft_model(fresh(), LoraConfig(
        r=r, lora_alpha=r, lora_dropout=0.0, bias="none", target_modules=names, init_lora_weights=True))
    peft_model.eval()
    peft_count = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    assert peft_count == count, f"r={r}: PEFT {peft_count} vs custom {count}"
    with torch.no_grad():
        peft_init = peft_model(input_ids=x, attention_mask=attn).logits
    assert torch.equal(peft_init, base_out), f"PEFT r={r} changed outputs at init"

    # Give both implementations the same nonzero adapters; their outputs must then agree.
    torch.manual_seed(r)
    peft_modules = dict(peft_model.base_model.model.named_modules())
    with torch.no_grad():
        for name in names:
            mine = custom.get_submodule(name)
            assert isinstance(mine, LoRALinear)
            mine.lora_A.copy_(0.01 * torch.randn_like(mine.lora_A))
            mine.lora_B.copy_(0.01 * torch.randn_like(mine.lora_B))
            theirs = peft_modules[name]
            theirs.lora_A["default"].weight.copy_(mine.lora_A)
            theirs.lora_B["default"].weight.copy_(mine.lora_B)
        a = custom(input_ids=x, attention_mask=attn).logits
        b = peft_model(input_ids=x, attention_mask=attn).logits
    gap = (a - b).abs().max().item()
    moved = (a - base_out).abs().max().item()
    assert gap < 1e-3 and moved > 1e-3, f"r={r}: gap {gap}, moved {moved}"
    print(f"lora r={r:<3}{count:>12,}{100 * count / full:>10.2f}%  init identical | PEFT max gap {gap:.1e} | moved {moved:.3f}")

print("ALL LORA TESTS PASSED")
