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

### Debugging the failed calibrated null (`scripts/diagnose_null.py`, `scripts/diagnose_null_stratified.py`)
- **What broke:** AIDA 50d trial 4 null hit 0.629, so the replicate queue exited 1 and Phase 4 correctly refused to start (`logs/phase4.exit` = 99).
- **How it was found:** first the pooled test, then 3 retraining seeds, then donor covariates. Seeds agreed (Spearman about 0.99), so the signal is a stable donor property.
- **Cause:** country explains 76.5% of donor null-score variance. A model tuned on some donors improves most on donors from similar sites. Trial 4 members had 22 Japanese donors vs 12 in holdout.
- **Fix tested:** count only same-country member-holdout pairs. Trial 4 fell to 0.537 / 0.536, and the SD across trials fell from about 0.10 to about 0.06.
- **Best teaching example:** this is the literal answer to "how do you know your attack isn't picking up batch effects?" Ask Steven to predict what a model trained on unrelated donors should score, then show 0.63.
- **Correction to an earlier assumption:** a random donor split balances batch only in expectation. Any one split can be imbalanced, so a single-trial AUC can look like leakage.
- **Still open:** trial 1 stays low within country (0.396 unseen, z = -1.66), which is within chance. OneK1K batch structure is unchecked.
- **Useful for Phase 4:** per-cell scores are saved, so within-country AUCs can be computed afterward without retraining.

### Phase 4 relaunch (2026-11-23)
- **Decision:** Steven approved launching with within-country AUC as AIDA's primary metric, reporting pooled AUC alongside it. Training is unchanged.
- **OneK1K batch check:** the metadata has no batch column, but the barcode suffix has 75 values. 980 of 981 donors have one suffix, and each suffix covers 9-14 donors. This looks like a sequencing pool, but that is inferred from the structure and not documented in the data.
- **Next analysis:** test whether this suffix explains OneK1K null scores, as country did for AIDA. With 9-14 donors per suffix, within-pool comparisons will have fewer pairs.
- **Smoke test:** passed on GPU 1, then the real queue started with the learning-rate check.

## Phases 5 and 6 setup (2026-11-24)
- **Steven's instruction:** proceed into the next phases when Phase 4 finishes, without waiting for a go-ahead.
- **Queue (`scripts/launch_after_phase4.sh`):** Phase 4 analysis, Phase 5, Phase 6, a score backfill, then the analysis again. Each step logs its exit code to `logs/chain.status`, and a failed step doesn't stop the later ones.

### Phase 5 (`scripts/run_phase5.py`)
- **ICL-A:** a frozen base Geneformer embeds cells (mean of the second-to-last layer). Classification uses 10 nearest neighbours or cell-type prototypes over a support set of member cells.
- **O2:** a linear cell-type head on the mean-pooled final layer, with head-only, last 2 layers, full, and LoRA r=8.
- **Support sizes:** k = 1, 5, 10, or 50 cells per type, plus all fit cells. k = 0 has no ICL-A analogue.
- **Attack:** cross-entropy of the true type. It is calibrated against the same method built on unrelated auxiliary donors, and that reference alone gives a null AUC.
- **ICL-B dropped:** a 50-gene cell sentence for one example per cell type fills most of the 2,048-4,096-token context of the candidate models (C2S Pythia, OLMo-2). The k = 5 to 50 sweep can't fit, so the protocol's "drop it if it becomes a time sink" rule applies.

### Phase 6: scGPT (`src/scgpt_model.py`, `scripts/test_scgpt.py`, `scripts/run_scgpt.py`)
- **Checkpoint:** whole-human, 12 layers, width 512, 8 heads, 51.3M parameters, 60,697-gene symbol vocabulary. The source and sha256 are in `scgpt_model.py`.
- **LoRA pitfall confirmed:** attention stores q, k, and v in one packed `Wqkv` matrix, so it isn't an ordinary `nn.Linear`. The model was reimplemented with separate q/k/v layers, and its outputs match the official `TransformerModel` exactly on real cells. LoRA trains 73,728 parameters per rank.
- **Surprise:** masked-value prediction on the base model scored worse than a per-cell mean. The checkpoint was pretrained only with the generative objective from the `dev-temp` branch, where genes to predict get a flag embedding instead of a value. O1 for scGPT uses that objective.
- **Still weak after the fix:** base predictions are nearly constant (SD 1.8 bins vs 14 for the truth, Pearson 0.37). This could be a weak model or an input mismatch with the unpublished pretraining pipeline. One epoch of fine-tuning on 20 cells already drops holdout MSE from 227 to 143, so the "utility" gain from fine-tuning scGPT is mostly recalibration.
- **Plan:** positive controls, then 50 donors trial 1 with every config, then trials 2-3 with five configs. Expect about a day of runtime.
- **Gene coverage:** 21,949 OneK1K genes and 23,782 AIDA genes are in the vocabulary.

### Backfill (`scripts/backfill_scores.py`)
- The pre-Phase 4 runs at 50 donors didn't save per-cell scores, so within-strata AUCs need reruns. Rows go to `logs/backfill_metrics.csv`, and reruns aren't bit-exact.
- **OneK1K strata caveat:** pools hold about 13 donors, so at 10-20 donors there are only a handful of same-pool pairs. `phase4_summary.csv` reports pair counts.

## Chain results and follow-ups (2026-11-27)
- **The chain ran cleanly:** Phase 4 finished at 04:00 UTC, then Phase 5, Phase 6, the backfill, and the analysis by 19:08, all with exit 0. The results are written up in `RESULTS.md`.
- **Steven's instruction:** run the follow-ups (scGPT fix, Phase 5 diagnosis, write-up, pretraining check). 3 trials per setting is enough, since this isn't a paper.

### scGPT OneK1K positive control
- **Failure:** at lr 5e-4 with no early stopping, full fine-tuning diverged at epoch 20. Train loss jumped from 121 to 203 and the model fell back to constant predictions, so seen AUC was 0.5. The gate dropped OneK1K from Phase 6 automatically, which the gate is designed to do.
- **Fix:** positive controls now use 1e-4, the same as the main full fine-tune. Gradient clipping was already on. Both datasets' controls were rerun, and the old rows are archived in `results/metrics_scgpt_positive_lr5e-4.csv`.
- **Lesson:** a positive control with a higher learning rate and no early stopping can fail for optimisation reasons. A failure says "training broke", not "nothing leaks". Check the loss curve before concluding anything.

### Phase 5 composition confound
- **Symptom:** the reference-only null AUC was 0.03-0.2 at small k, when it should be about 0.5.
- **Cause:** k cells per type over-samples rare types in the members' seen cells. Rare types are harder to classify, so members look worse to any classifier, including the reference.
- **Fix:** `scripts/analyze_phase5.py` subtracts the per-type mean score before averaging per donor. The reference null is then 0.42-0.58 for k ≥ 50. OneK1K at k ≤ 10 and every k = 1 result stay unreliable.
- **Lesson:** whenever the member cells are chosen by a different rule than the holdout cells, the per-donor average mixes composition with membership. Score within cell type.

### Pretraining overlap
- **scGPT:** checked against the CELLxGENE Census LTS `2023-05-15` via `cellxgene_census`, installed in a scratch venv so `scfm_` stays untouched. Every OneK1K and AIDA donor of ours is in it, labelled `normal` and primary. scGPT used the `2023-05-08` weekly build.
- **Geneformer V1:** probably neither dataset is in its corpus, but this is unverified.
- **Consequence:** for scGPT, holdouts are already pretraining members, so our AUC measures what fine-tuning adds on top of pretraining.

### Filling to 3 trials (`scripts/fill_trials.py`, `scripts/launch_fill.sh`)
- **Runs added:** Phase 2-3 sweep configs outside the grid (last1, last4, LoRA r1/2/4/16/32) get trials 2-3, and the LR check gets trial 3. These run on the existing 50-donor cohorts and queue after the scGPT rerun.

### Things learned from Phase 4
- **LoRA vs freezing:** at equal learning rate, they leak the same for the same parameter budget. The Phase 2-3 "LoRA leaks more" reading was the learning-rate confound.
- **Head and LoRA utility:** the unrelated-donor null gets most of their utility gain, which suggests those configs mostly learn the dataset, not the donors.
- **AIDA 10-donor nulls sit at about 0.7:** unexplained, and noted as a caveat.

### scGPT results (2026-11-28)
- **OneK1K rerun:** positive control at 1e-4 gave seen AUC 1.00, and all nulls passed. The rerun finished at 02:13 UTC.
- **Main finding:** at 50 donors only full fine-tuning leaks, and only a little (0.59-0.63). Every config, including the null on unrelated donors, reaches the same holdout MSE, so fine-tuning recalibrates the base model rather than learning donors.
- **Confound to keep in mind:** validation loss plateaus within 1-2 epochs, so early stopping ends runs at 4-17 epochs, while Geneformer often used all 30. A fixed-epoch comparison (say 30 epochs for both models) would separate "the model resists memorization" from "it stopped early".
- **Trials 2-3 for the 7 non-core configs** are queued after the Geneformer fill (`scripts/launch_fill_scgpt.sh`, tmux `scfm_fill_scgpt`).

### Fill to 3 trials done (2026-11-28)
- **Runs:** the Geneformer fill ran 02:15-12:57 UTC, and the scGPT fill 13:05-17:27. All exited 0, and every setting now has at least 3 trials.
- **Geneformer sweep:** with 3 trials both datasets agree within 0.03 for every config, and seen AUC rises smoothly with parameter count.
- **scGPT sweep:** every non-full config matches the null.
- **LR check with trial 3:** same best rates as before, and the equal-LR comparison still shows LoRA ≈ freezing. `analyze_phase4.py` now reads trials 1-3 for the LR check.
