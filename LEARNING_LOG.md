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
