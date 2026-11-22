# Learning log

Running teaching notes, written as the work happens.

## Phase 0: setup and data audit (2026-11-22)

### Workspace
- **Did:** used `~/MIAsOnFineTunedSingleCell` with conda env `scfm_` (Python 3.10, torch 2.14.1, transformers 5.20.2, peft 0.21.3, anndata 0.11.4).
- **Why:** Steven's request to use this repo and push to GitHub overrides the protocol's `~/scfm_finetune/` and "do not push" lines.
- **Limits:** GPU 1 only (GPU 0 stays free for daniilf), CPU cores 0-11 via `taskset`.
- **Watch:** the disk had 99 GB free after the env install, so save adapters and metrics only.

### Data audit
- **Commands:** `scripts/audit_data.py`, `scripts/audit_splits_raw.py`. Output is in `logs/` (not committed).

| Dataset | Cells | Genes | Donors | Cell types | Count source | Gene IDs |
|---|---|---|---|---|---|---|
| `ok` | 1,267,733 | 25,834 | 981 | 14 | `X`, integer | Symbols only |
| `aida` | 1,058,909 | 35,240 | 508 | 33 | `.raw.X`, integer | Ensembl |
| `cg` | 108,838 | 32,802 | 22 | 9 | `.raw.X`, integer | Ensembl |

- **What broke:** `a.raw[idx].X` returns an unread `_CSRDataset` in backed mode. Indexing `a.raw.X[idx]` works.
- **Surprise 1:** `X` in AIDA and `cg` is log-normalized (max about 7.7). Tokenizing `X` would silently produce wrong ranks and depth normalization. Use `.raw.X`.
- **Surprise 2:** auxiliary donors overlap holdout or train in some splits. This is documented as intentional in the scMAMA-MIA `CLAUDE.md`. Train and holdout never overlap.
  - AIDA 200d: 44-48 auxiliary donors are in train and 44-48 in holdout.
  - OneK1K 490d: all 200 auxiliary donors are in holdout.
  - `cg` 10d, 11d, 20d: auxiliary donors are drawn from train and holdout. The `20d` folders actually contain 11 train donors.
- **Decision:** validation and shadow donors are `auxiliary - train - holdout`, because holdout must not influence training.
- **Open:** OneK1K needs a symbol-to-Ensembl mapping, which the protocol says to raise with Steven.

### Model choice and gene coverage
- **Did:** downloaded `Geneformer-V1-10M` at revision `1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5`, with the `gc30M` dictionaries. The V1 checkpoint is the smallest, as the protocol prefers for a pilot.
- **Shape to show Steven:** 6 layers, hidden size 256, 4 heads, feed-forward 512, vocabulary 25,426, 10,288,722 parameters.
- **Safety:** dictionaries are pickles, so `src/gf_tokenizer.py` uses an unpickler that only permits numpy scalars.
- **Coverage (`scripts/check_gene_mapping.py`, 2,000-cell sample):**
  - `ok`: 62.0% of genes, 89.4% of counts, through symbol lookup
  - `aida`: 55.4% of genes, 95.8% of counts
  - `cg`: 60.1% of genes, 90.9% of counts

### Phase 0 gate (`scripts/phase0_gate.py`): passed
- One AIDA naive CD4 T cell: 2,017 nonzero genes became 1,943 tokens. Top genes were JUNB, ZFP36L2, KLF2, JUND.
- Shapes: input `(1, 1943)`, logits `(1, 1943, 25426)`, 7 hidden states of `(1, 1943, 256)`.
- With position 971 masked, the true gene ranked 5th of 25,426.

## Phase 1: pilot (`scripts/phase1_pilot.py`)
- **Deviation:** the protocol's pilot is OneK1K. It runs on AIDA instead because OneK1K's gene-ID mapping is awaiting Steven's decision.
- **Why these masks:** evaluation masks are seeded by `(mask_seed, cell_id, mask_index)`, so base and fine-tuned losses compare the same masked genes.
- **Positive control:** AIDA 10d, all weights, 30 epochs, lr 5e-4, no early stopping.
- **Pilot:** AIDA 50d, all weights, lr 1e-4, up to 10 epochs, early stopping on auxiliary donors outside train and holdout.
- **Null-spread reference:** with 50 vs 50 donors, the null AUC standard deviation is about 0.058.
