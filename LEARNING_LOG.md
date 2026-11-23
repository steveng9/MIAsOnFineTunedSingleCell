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

### Phase 1 outcome
- **Gate:** null, positive, shuffle, and leakage checks passed. Numbers are in `RESULTS.md`.
- **Surprise 1:** the 50-donor model separates seen cells almost perfectly (calibrated AUC 0.975), yet unseen cells from those donors give 0.453. Here it looks like cell memorization, not donor memorization, which is the opposite of the scMAMA-MIA generator pattern. One trial only.
- **Surprise 2:** calibration matters a lot for seen cells (raw 0.649, calibrated 0.975). Subtracting the base loss removes how hard each cell is to predict.
- **Worth showing Steven:** in the positive control, train loss fell from 2.664 to 2.491 while validation loss stayed near 2.59. A small loss gap was enough for AUC 1.0 on seen cells.
- **What broke:** PyTorch logged 4 allocator OOM retries during the first evaluation at batch size 24, with GPU memory full of cached blocks. The run recovered with no errors. Lower `EVAL_BATCH` before sweeps.
- **Control weakness found:** the calibrated attack is exactly zero on the base model, so its null is untested. Next control: fine-tune, then attack two disjoint groups of non-members.

## Phases 2 and 3: freezing and LoRA sweeps (`scripts/run_sweep.py`)

### Decisions after the Phase 1 check-in
- Steven approved proceeding, mapping OneK1K to Ensembl, adding a real calibrated null, and raising the epoch cap.
- **OneK1K mapping (`scripts/audit_ok_mapping.py`):** 16,006 symbols map one-to-one, with no duplicate Ensembl IDs, covering 89.49% of counts. MALAT1 (6.67% of counts) is the largest loss and is absent from the V1 vocabulary. Old symbols such as GNB2L1 and H3F3B are also lost.
- **Calibrated null:** fine-tune all weights on up to 20 unrelated auxiliary donors for 10 epochs, then attack members vs holdout. Neither group was trained on, so AUC should be near 0.5. This replaces the earlier idea of splitting holdout in two, because it exercises the whole training pipeline.
- **Training:** up to 30 epochs, patience 3, on auxiliary-donor validation loss.
- **Learning rates:** 1e-4 for frozen and full configs, 1e-3 for LoRA, since the adapters start from zero. These were not tuned, so the LoRA and non-LoRA curves differ in learning rate as well as parameter count.

### LoRA (`src/lora.py`, `scripts/test_lora.py`)
- **Shapes in one Geneformer layer:** query, key, value, and attention output are `(256, 256)`. Intermediate is `(512, 256)`, output is `(256, 512)`.
- **Parameter count:** each layer has `4r(512) + 2r(768) = 3584r`. Six layers give `21,504r`, so r=64 trains 1,376,256 parameters (13.38% of the model).
- **Tests passed:** at every rank, outputs before training exactly equal the base model, counts match `r(m+n)`, and outputs match PEFT when both hold the same nonzero adapters.
- **Worth showing Steven:** with B=0, the first gradient for A is exactly zero, because `d(BA)/dA` is scaled by B. B has to move first.
- **Head config:** the decoder weight is tied to the gene embeddings, so "head only" trains the transform and bias (91,730 parameters) and leaves the embeddings frozen.

### Sweep outcome (50 donors, trial 1)
- **Gates:** OneK1K positive control 1.000 (seen). Calibrated null: OneK1K 0.524 / 0.505, AIDA 0.418 / 0.406 (seen / unseen).
- **Surprise 1:** seen-cell AUC climbs smoothly with log parameters, but unseen-cell AUC is flat. Ask Steven to predict this before showing the figure.
- **Surprise 2:** AIDA unseen AUCs are all about 0.42-0.445, and a model trained on unrelated donors gets 0.406. A consistent below-chance value across configs is a split effect, and it is a good example for "how do you know your attack isn't picking up batch effects?"
- **Surprise 3:** LoRA r=64 had worse holdout loss than r=32 in both datasets, and LoRA was worse than head-only on OneK1K at every rank. The learning-rate difference may explain this, so don't teach it as a LoRA property yet.
- **Next:** replicate at 50 donors over trials 2-5 with `head`, `last2`, `full`, LoRA r=8 and r=64. The whole Phase 4 grid at the current cost would be well over 24 hours, so that needs Steven's decision.

## Phase 4: donor-count grid (`scripts/run_phase4.py`, `scripts/launch_phase4.sh`)
- **Approved by Steven:** launch after the replicates, cut to the five configs, and include a small learning-rate check.
- **Grid:** donor counts {10, 20, 50, 100, 200}, trials 1-5, OneK1K and AIDA. Each cohort runs base, unrelated-donor null, `head`, `last2`, `full`, LoRA r=8 and r=64.
- **LR check:** 50 donors, trials 1-2. Freezing configs add lr 3e-4 and 1e-3, LoRA adds 1e-4 and 3e-4, so both families share the grid {1e-4, 3e-4, 1e-3}. Any "best lr" must be chosen on auxiliary validation loss, never holdout.
- **Why the null gate changed:** with dozens of cohorts and two threat models, a 2 SD rule would stop the queue by chance. The queue now stops only beyond 4 null SDs, and the recorded nulls get analyzed across trials afterward.
- **Why save per-cell scores:** per-cell-type and AIDA subgroup analyses need cell-level losses. They go to `scores/`, which is not committed.
- **Disk:** Phase 4 keeps LoRA adapters but no full checkpoints.
- **Not in this queue:** objective O2 (cell-type head), per-cell-type and subgroup analysis, Phases 5 and 6.
