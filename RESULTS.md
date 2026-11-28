# Results

All numbers below are **preliminary**: few trials (1-5), one training seed per run, Geneformer V1-10M and scGPT whole-human only. Every number traces to a CSV in `results/` named in its section. Control verdicts are in `results/phase1_gates.json`. Sections are in the order the work was done; later sections supersede earlier ones where they disagree.

## Overview (as of 2026-11-27)

AUC is donor-level membership-inference AUC (0.5 = chance). "Seen" means the attacker holds the exact cells used for fine-tuning; "unseen" means the attacker holds other cells from the same donors.

1. **Exact-cell leakage is large and controlled by cohort size and trained parameters** (Geneformer, Phase 4). Full fine-tuning gives seen AUC 1.00 up to 50 donors and 0.89-0.98 at 200. Head-only falls from 0.96-1.00 at 10 donors to 0.55-0.59 at 200. The ordering full > last2 ≈ LoRA r64 > LoRA r8 > head holds at every size, in both datasets, and survives batch-stratified scoring.
2. **Donor-level leakage (unseen cells) is small.** It is detectable only at 10-20 donors on AIDA (about 0.6 against a null of about 0.47 at 20 donors), and at 50+ donors it is indistinguishable from the null for every config. OneK1K shows none at any size.
3. **LoRA vs freezing is mostly a learning-rate effect.** At equal learning rate, LoRA and freezing with similar parameter counts leak about the same. The earlier "LoRA leaks more" reading came from LoRA using a 10x higher learning rate (Phase 4 LR check).
4. **In-context learning is not automatically private** (Phase 5). kNN over member-cell embeddings leaks the seen cells completely (AUC 1.00) while classifying worse than any fine-tuned model. Prototype averaging leaks less, but has the worst utility. As with fine-tuning, no method leaks donor identity through unseen cells.
5. **scGPT barely leaks at 50 donors** (Phase 6). Only full fine-tuning rises above the null (seen AUC 0.63 OneK1K, 0.59 AIDA, against nulls of 0.54 and 0.46). Every config, including the unrelated-donor null, gives the same utility gain, so fine-tuning mostly recalibrates a weak base model. Caveats: scGPT early-stops after 4-17 epochs, and all its donors were in its pretraining data.
6. **Pretraining overlap:** every donor in both datasets was in scGPT's pretraining corpus. Neither dataset is likely to be in Geneformer V1's. See the pretraining section.

## Phase 1 pilot

**Setup:** Geneformer V1-10M, AIDA, objective O1 (masked-gene loss), trial 1, up to 200 cells per donor. AUCs are donor-level with mean aggregation unless noted. The protocol specified OneK1K, but AIDA was used because OneK1K's gene-ID mapping is awaiting a decision.

### Controls

| Control | Cohort | Result | Verdict |
|---|---|---|---|
| Leakage assertions | 50d and 10d | No holdout donor or cell in fit or validation data | Pass |
| Null, raw loss on base model | 50d | AUC 0.469 (seen), 0.468 (unseen); null SD about 0.058 | Pass |
| Positive, full weights, 30 epochs | 10d | Seen-cell calibrated AUC 1.000 | Pass |
| Label shuffle, 1,000 permutations | 10d | Mean AUC 0.505 | Pass |

### Full-weight fine-tune

| Cohort | Threat model | Raw loss AUC | Calibrated AUC |
|---|---|---|---|
| 50d pilot | Seen cells | 0.649 | 0.975 |
| 50d pilot | Same donor, unseen cells | 0.456 | 0.453 |
| 10d positive control | Seen cells | 1.000 | 1.000 |
| 10d positive control | Same donor, unseen cells | 0.700 | 0.850 |

Masked-gene cross-entropy on the 50 holdout donors fell from 2.605 (base) to 2.506 (fine-tuned).

### Reading

- In the 50-donor pilot, the attack identifies donors from their exact training cells. It shows no signal from the same donors' unseen cells.
- The 10-donor overfit model shows some unseen-cell signal, but it is weak evidence. It is one trial with 10 vs 10 donors (null SD about 0.13), and the calibrated AUC ranges from 0.71 to 0.85 across aggregations.

### Caveats

- The calibrated attack has no informative null yet. On the base model its score is exactly zero, so the null row tests raw loss only.
- The label shuffle permutes finished donor scores. It checks the AUC computation, not the training pipeline.
- The 50-donor pilot hit its 10-epoch cap with validation loss still falling, so it may be undertrained.
- The positive control and pilot differ in donor count, learning rate, and epochs, so they do not isolate the effect of any one factor.

## Phases 2 and 3: freezing and LoRA sweeps

**Setup:** Geneformer V1-10M, objective O1, 50 donors, trial 1 only, up to 30 epochs with early stopping on auxiliary donors. AUCs use the calibrated attack with mean aggregation. Figure: `figures/phase23_params_vs_leakage.png`. **Preliminary.**

| Trained | Parameters | OneK1K seen | OneK1K unseen | OneK1K holdout loss | AIDA seen | AIDA unseen | AIDA holdout loss |
|---|---|---|---|---|---|---|---|
| Base | 0 | n/a | n/a | 2.900 | n/a | n/a | 2.605 |
| Unrelated-donor null | 10,288,722 | 0.524 | 0.505 | 2.760 | 0.418 | 0.406 | 2.527 |
| LoRA r=1 | 21,504 | 0.613 | 0.530 | 2.828 | 0.557 | 0.445 | 2.573 |
| LoRA r=2 | 43,008 | 0.638 | 0.511 | 2.814 | 0.588 | 0.443 | 2.565 |
| LoRA r=4 | 86,016 | 0.696 | 0.523 | 2.802 | 0.634 | 0.418 | 2.555 |
| Head | 91,730 | 0.654 | 0.504 | 2.770 | 0.594 | 0.443 | 2.546 |
| LoRA r=8 | 172,032 | 0.738 | 0.502 | 2.795 | 0.672 | 0.416 | 2.550 |
| LoRA r=16 | 344,064 | 0.787 | 0.517 | 2.790 | 0.770 | 0.421 | 2.541 |
| Last 1 | 527,104 | 0.771 | 0.518 | 2.774 | 0.730 | 0.445 | 2.536 |
| LoRA r=32 | 688,128 | 0.859 | 0.528 | 2.789 | 0.838 | 0.420 | 2.539 |
| Last 2 | 1,054,208 | 0.848 | 0.512 | 2.763 | 0.803 | 0.441 | 2.527 |
| LoRA r=64 | 1,376,256 | 0.884 | 0.522 | 2.796 | 0.869 | 0.431 | 2.543 |
| Last 4 | 2,108,416 | 0.921 | 0.522 | 2.759 | 0.879 | 0.434 | 2.521 |
| Full | 10,288,722 | 1.000 | 0.534 | 2.733 | 0.999 | 0.426 | 2.484 |

### Reading

- Seen-cell leakage rises with trainable parameters in both datasets and both config families.
- Unseen-cell AUC does not rise with parameters in either dataset. These runs show no evidence of donor-level memorization at 50 donors.
- In this trial, LoRA had higher seen-cell AUC than freezing at similar parameter counts, and a smaller reduction in holdout loss. This comparison is **confounded**: LoRA used lr 1e-3, the others 1e-4, and neither was tuned.

### Caveats

- All configs within a dataset share one donor split, so their AUCs are not independent. Agreement across configs is not replication.
- AIDA's unseen-cell AUCs sit at 0.416-0.445 for every config, and its unrelated-donor null is 0.406. A model that never saw either group gives the same below-chance value, so this looks like a property of this donor split rather than an effect of training. More trials are needed to confirm that.
- Because of that offset, AIDA AUCs should be read against its null line as well as against 0.5.

## Calibrated-null failure and diagnosis

The replicate queue stopped at AIDA 50 donors, trial 4, because the calibrated null reached AUC 0.629 (seen) and 0.630 (unseen), past the 2 SD gate. Trial 4 for AIDA is incomplete, and trial 5 has not run for either dataset. Sources: `results/null_diagnostic_aida_50d_t4.json` and `results/null_diagnostic_within_country.json`.

- **Not training noise:** retraining that null with 3 seeds gave unseen-cell AUCs of 0.622-0.630, and donor scores correlated at Spearman 0.987-0.991 across seeds.
- **Country confound:** country explains 76.5% of the variance in donor null scores in that trial. Members included 22 Japanese donors and holdout 12. That imbalance is not itself significant (p = 0.083).
- **Across 8 null cohorts:** mean AUC is 0.492 (seen) and 0.479 (unseen). The chi-square test against the theoretical null gives p = 0.093 and p = 0.056, so the evidence for excess spread is suggestive and not conclusive.

Comparing only member-holdout donor pairs from the same country, for AIDA trials 1-4 (a fresh null run per trial):

| Threat model | Pooled mean | Pooled SD | Within-country mean | Within-country SD | Largest within-country z |
|---|---|---|---|---|---|
| Seen cells | 0.488 | 0.099 | 0.498 | 0.053 | 1.27 |
| Same donor, unseen cells | 0.481 | 0.104 | 0.483 | 0.061 | 1.66 |

Trial 4 moves from 0.629 to 0.537 (seen) and from 0.630 to 0.536 (unseen).

**Reading:** on AIDA, the pooled calibrated attack partly measures which country a donor comes from. Within-country scoring removes the failed null and brings the spread close to the permutation SD of about 0.06. Pooled AIDA AUCs from single trials, including the unseen-cell values above, should not be read as evidence for or against donor leakage. OneK1K has not been checked for an equivalent batch effect.

## Phase 4: donor-count grid and learning-rate check (Geneformer)

**Setup:** Geneformer V1-10M, objective O1. OneK1K and AIDA at 10, 20, 50, 100, and 200 member donors, with 5 trials per cell (different donor splits). Configs are head, last 2 layers, full, and LoRA r=8 and r=64 at the default learning rates (1e-4, or 1e-3 for LoRA). Each cohort also has an unrelated-donor null (full fine-tune on auxiliary donors, 10 epochs). Up to 200 cells per donor; early stopping on auxiliary donors. Sources: `results/metrics.csv` (pooled AUCs, utility), `results/phase4_stratified.csv` (per run), `results/phase4_summary.csv` (means), `results/phase4_lr_check.csv`.

**Batch-stratified AUC:** only member-holdout donor pairs from the same stratum are compared. AIDA strata are country. OneK1K strata are the cell-barcode suffix, which looks like a sequencing pool of about 13 donors; this is inferred, not documented. The 50-donor rows for trials 1-2 come from the score backfill, which reran those cohorts with the same seeds (`logs/backfill_metrics.csv`).

### Seen cells, calibrated attack

Mean ± SD over 5 trials of the within-stratum AUC, with the pooled AUC in parentheses.

| Config | 10 donors | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| **OneK1K** full | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (0.99) | 0.98±0.01 (0.90) |
| last2 | 1.00 (1.00) | 1.00 (0.97) | 0.93±0.05 (0.83) | 0.79±0.02 (0.69) | 0.71±0.02 (0.64) |
| LoRA r64 | 1.00 (1.00) | 1.00 (0.99) | 0.93±0.02 (0.84) | 0.77±0.05 (0.68) | 0.65±0.03 (0.62) |
| LoRA r8 | 1.00 (0.98) | 1.00 (0.93) | 0.82±0.03 (0.72) | 0.64±0.03 (0.60) | 0.60±0.03 (0.58) |
| head | 1.00 (0.85) | 0.78±0.22 (0.77) | 0.78±0.04 (0.63) | 0.56±0.02 (0.56) | 0.59±0.03 (0.56) |
| null | 0.50±0.35 (0.51) | 0.31±0.41 (0.46) | 0.56±0.10 (0.51) | 0.44±0.03 (0.47) | 0.50±0.03 (0.51) |
| **AIDA** full | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 0.97±0.01 (0.96) | 0.89±0.01 (0.88) |
| last2 | 1.00 (1.00) | 0.99±0.02 (0.97) | 0.87±0.02 (0.86) | 0.75±0.03 (0.75) | 0.65±0.03 (0.65) |
| LoRA r64 | 1.00 (1.00) | 0.99±0.02 (0.98) | 0.90±0.02 (0.89) | 0.73±0.04 (0.74) | 0.64±0.02 (0.64) |
| LoRA r8 | 0.99±0.01 (0.99) | 0.97±0.02 (0.95) | 0.76±0.04 (0.76) | 0.65±0.03 (0.65) | 0.58±0.03 (0.58) |
| head | 0.96±0.04 (0.96) | 0.85±0.07 (0.84) | 0.65±0.03 (0.67) | 0.59±0.05 (0.60) | 0.55±0.02 (0.55) |
| null | 0.71±0.11 (0.60) | 0.48±0.19 (0.48) | 0.49±0.04 (0.49) | 0.47±0.06 (0.50) | 0.48±0.02 (0.49) |

Permutation SD of a single within-stratum AUC: AIDA 0.16, 0.10, 0.06, 0.04, 0.03 for 10 to 200 donors. OneK1K 0.42, 0.30, 0.10, 0.06, 0.04, because there are only 2-4 same-pool pairs at 10-20 donors (`pairs_mean` in the summary). **OneK1K stratified values at 10-20 donors are not interpretable.**

### Same donor, unseen cells, calibrated attack (within-stratum)

| Config | AIDA 10 | AIDA 20 | AIDA 50 | AIDA 100 | AIDA 200 | OneK1K 50 | OneK1K 100 | OneK1K 200 |
|---|---|---|---|---|---|---|---|---|
| full | 0.81±0.09 | 0.62±0.14 | 0.52±0.07 | 0.52±0.02 | 0.52±0.03 | 0.60±0.08 | 0.46±0.03 | 0.53±0.02 |
| last2 | 0.83±0.12 | 0.62±0.10 | 0.50±0.05 | 0.51±0.03 | 0.51±0.03 | 0.56±0.12 | 0.44±0.04 | 0.50±0.03 |
| LoRA r64 | 0.84±0.09 | 0.61±0.12 | 0.50±0.06 | 0.51±0.03 | 0.50±0.03 | 0.56±0.13 | 0.41±0.02 | 0.50±0.02 |
| LoRA r8 | 0.81±0.10 | 0.60±0.08 | 0.50±0.06 | 0.50±0.03 | 0.51±0.03 | 0.58±0.08 | 0.44±0.02 | 0.50±0.02 |
| head | 0.81±0.12 | 0.63±0.10 | 0.50±0.06 | 0.51±0.03 | 0.50±0.02 | 0.53±0.06 | 0.42±0.03 | 0.51±0.03 |
| null | 0.66±0.09 | 0.47±0.16 | 0.48±0.05 | 0.47±0.06 | 0.49±0.02 | 0.54±0.08 | 0.43±0.04 | 0.50±0.03 |

### Utility (masked-gene cross-entropy on holdout donors, lower is better)

The base model scores 2.605 on AIDA and 2.91 on OneK1K. At 200 donors: full 2.439 / 2.701, last2 2.510 / 2.756, head 2.537 / 2.768, LoRA r64 2.536 / 2.798, LoRA r8 2.545 / 2.800 (AIDA / OneK1K). Gains grow with donors for full and last2 and barely move for head and LoRA. The unrelated-donor null model reaches 2.53-2.55 / 2.77-2.79, so **the null alone captures most of the gain of head and LoRA fine-tuning**: much of what they learn is "this kind of data" rather than "these donors".

### Learning-rate check (50 donors, trials 1-2)

Best validation loss on auxiliary donors (never holdout) picked these learning rates. For both datasets and both trials: full and last2 at 1e-4, head at 3e-4, and LoRA r8 and r64 at 3e-4. The validation-loss differences between learning rates are small (under 0.01, except full and last2 at 1e-3, which are 0.03-0.08 worse).

Seen AUC rises with learning rate for every config. At **equal** learning rate 1e-4, mean of trials 1-2:

| | OneK1K seen | AIDA seen | Parameters |
|---|---|---|---|
| head | 0.66 | 0.64 | 92k |
| LoRA r8 | 0.67 | 0.67 | 172k |
| last2 | 0.85 | 0.84 | 1.05M |
| LoRA r64 | 0.82 | 0.82 | 1.38M |

**Reading:** at matched learning rate and similar parameter counts, LoRA and freezing leak about the same. The Phase 2-3 gap (LoRA higher) was the learning-rate confound flagged there. The grid uses LoRA at 1e-3 and head at 1e-4, neither of which is the validation-best rate, so cross-family comparisons in the grid tables above carry that confound too. Within-family trends are unaffected.

### Reading

- Exact-cell leakage falls steadily with cohort size for every config, but full fine-tuning still reaches 0.89-0.98 at 200 donors.
- Batch stratification changes little at 50+ donors. Pooled and within-stratum values agree within about 0.02 for AIDA. OneK1K within-pool values are *higher* than pooled at 50-200 donors (for example last2 at 200 donors: 0.71 vs 0.64). That is consistent with pool-level batch variation adding noise across pools rather than creating signal.
- Unseen-cell leakage appears only for AIDA at 10-20 donors, and it is roughly the same for every config, including head-only. At 10 donors the null itself sits at 0.66-0.71 (an unexplained offset at this size), so only the 20-donor gap (about 0.61 vs 0.47, about 3 standard errors over 5 trials) is reasonable evidence.

### Caveats

- At 10 donors, AIDA's stratified null is 0.71±0.11 on seen cells. That is above 0.5 by more than 3 standard errors over 5 trials. Nulls at other sizes are fine. The cause is not diagnosed. Possibilities include something systematic in the 10-donor splits from scMAMA-MIA, or a country stratum that dominates the few pairs.
- The 50-donor trial 1-2 within-stratum values come from backfill reruns, which are not bit-identical to the original runs.
- One training seed per run; trial-to-trial SDs mix donor-split and training noise.

## Phase 5: fine-tuning (O2) vs in-context learning (ICL-A), Geneformer

**Setup:** cell-type classification on 50-donor cohorts, trials 1-3, both datasets. Support/training data are k cells per type from member fit cells (k = 1, 5, 10, 50), or all fit cells.
- **ICL-A** (no weights change): the frozen base model embeds cells (mean-pooled second-to-last layer). A query is classified by kNN over the support embeddings (10 neighbours, cosine), or by softmax over cosine to cell-type prototypes (class means).
- **O2 fine-tuning:** a linear head on the final layer, with the encoder frozen (head), the last 2 layers trained, full, or LoRA r8.
- **Attack:** a query returns class probabilities, and the score is the cross-entropy of the true type. The calibrated score uses a reference built the same way on auxiliary donors. The reference alone gives a null (`reference_null`).
- **ICL-B** (LLM prompting on cell sentences) was not run: one example per cell type already fills most of the candidate local LLMs' context.

Sources: `results/metrics_phase5.csv` (raw), and `results/phase5_typecentered.csv` and `results/phase5_summary.csv` (from `scripts/analyze_phase5.py`).

**A correction was needed.** With k cells per type, a member's seen cells are balanced across types, so rare types are over-represented compared with holdout donors. Rare types are harder to classify, so members look "high loss" to any classifier: the reference-only null fell as low as 0.03. The re-analysis subtracts each cell type's mean score before averaging per donor (**type-centered**). That brings the reference null to 0.42-0.58 for k = 50 and "all" in both datasets, and for k ≥ 5 on AIDA. On OneK1K at k ≤ 10 it is still up to 0.9, so **OneK1K results at k ≤ 10 and all k = 1 results are not interpretable**. The tables use type-centered scores.

### Main results (type-centered, calibrated; mean ± SD over 3 trials)

| Method | k | OneK1K F1 | OneK1K seen | OneK1K unseen | AIDA F1 | AIDA seen | AIDA unseen |
|---|---|---|---|---|---|---|---|
| Fine-tune LoRA r8 | all | 0.80 | 0.84±0.07 | 0.54 | 0.69 | 0.90±0.10 | 0.56 |
| Fine-tune last2 | all | 0.77 | 0.93±0.01 | 0.55 | 0.62 | 0.91±0.05 | 0.56 |
| Fine-tune full | all | 0.76 | 0.89±0.08 | 0.56 | 0.64 | 0.98±0.01 | 0.54 |
| Fine-tune full | 50 | 0.66 | 0.93±0.01 | 0.50 | 0.50 | 1.00 | 0.56 |
| Fine-tune head | all | 0.61 | 0.58±0.06 | 0.51 | 0.43 | 0.64±0.13 | 0.57 |
| Fine-tune head | 50 | 0.57 | 0.53±0.12 | 0.51 | 0.38 | 0.61±0.09 | 0.55 |
| ICL kNN | all | 0.48 | 1.00 | 0.51 | 0.33 | 1.00 | 0.57 |
| ICL kNN | 50 | 0.43 | 0.83±0.07 | 0.52 | 0.32 | 0.96 | 0.56 |
| ICL prototype | all | 0.30 | 0.58±0.06 | 0.54 | 0.28 | 0.63±0.03 | 0.56 |
| ICL prototype | 50 | 0.30 | 0.70±0.07 | 0.51 | 0.27 | 0.80±0.01 | 0.56 |
| Reference null | all | | 0.49±0.10 | | | 0.47±0.12 | |

Macro-F1 is on holdout donors. Unseen SDs are 0.01-0.11, and the permutation SD of a single AUC is about 0.06.

### Reading

- **kNN in-context learning is the leakiest method tested.** The support cells are stored verbatim, and each one is its own nearest neighbour, so the attacker identifies seen cells perfectly. It also has much lower F1 than any fine-tuned model.
- **Prototypes** average cells away, so leakage drops as k grows (0.80 at k = 50 to 0.63 with all cells on AIDA). But F1 stays near 0.3.
- **Among fine-tuned models,** head-only is the least leaky (0.53-0.64) but gives up 0.2 F1. LoRA r8 has the best F1 and leaks less than full or last2 on OneK1K, but not on AIDA.
- **Unseen-cell AUC is 0.50-0.57 for every method.** On AIDA it is 0.54-0.57 for every method alike, including head and prototypes. Meanwhile the reference-only unseen null sits at 0.44-0.46, and the calibrated score adds the reference loss, so that offset pushes the calibrated AUC up by about the same amount. No method shows donor-level leakage beyond this shared offset.
- Utility vs leakage: no method dominates. The useful contrast is "where are member cells stored": verbatim (kNN: worst), averaged (prototypes, head), or diffused into weights (full, LoRA).

### Caveats

- Type-centering uses the score mean over all scored cells in a run, including members. That centering is symmetric between groups, but it is a post-hoc fix, chosen after seeing the failed null.
- With k cells per type, not every member donor contributes support cells: at k = 10 about 45 of 50 donors on OneK1K and about 50 on AIDA; at k = 1 only 12-24.
- Class-probability access is a stronger threat model than label-only access, which was not run.

## Pretraining overlap

The protocol asked whether OneK1K and AIDA donors were in the models' pretraining data. Checked 2026-11-27 using public metadata only (no cell data left the machine).

- **scGPT whole-human:** pretrained on normal (non-disease) cells from the CELLxGENE Census, version `2023-05-08` (`external/scGPT/data/cellxgene/data_config.py`). That weekly build is no longer hosted. The nearest long-term release, `2023-05-15`, contains:
  - **OneK1K:** dataset `3faad104-2ab8-4434-816d-474d8d2641db`, 981 donors, all labelled `normal` and primary data. All 981 match our donor IDs. The dataset is also listed by name in the scGPT repo's `metainfo.json`, and it was on CELLxGENE from 2022-04-08.
  - **AIDA Phase 1:** dataset `b0e547f0-462b-4f81-b31b-5b0a5d96f537`, 508 donors, 1,058,909 cells, all `normal` and primary. Our copy has the same 508 donors and the same cell count. It was published on CELLxGENE on 2023-05-04, four days before scGPT's build, so it was **very likely, not certainly**, in the 2023-05-08 build.
  - **So for scGPT, our holdout donors are pretraining members that were not fine-tuned,** and our members are pretraining members that were also fine-tuned. There is no never-seen donor group in these datasets.
- **Geneformer V1 (Genecorpus-30M):** assembled in 2021 from public data, with sources listed in the Methods of Theodoris et al., Nature 2023 (not checked line by line). AIDA's first public release we found is 2023, and OneK1K's data was published in 2022, so **probably neither is in V1's corpus**. Unverified.
- **Consequence:** the Geneformer results measure fine-tuning leakage for donors new to the model. The scGPT results measure the extra leakage that fine-tuning adds on top of pretraining. The calibrated score subtracts the base model's loss, so pretraining membership cancels out of it by design. A pretraining-membership attack on the base model needs a never-seen comparison group from the same pipeline, which we don't have.
- **Prior work:** the protocol cites arXiv:2607.20572 ("Auditing pretraining contamination in single-cell foundation model benchmarks") as the closest work. A web search on 2026-11-27 did not find it. **Steven: please check that citation.**

## Phase 6: scGPT (whole-human)

**Setup:** scGPT whole-human (51.3M parameters), fine-tuned with its own pretraining objective. That objective predicts the expression bins of hidden genes from the visible ones (generative mode; see `LEARNING_LOG.md`), and the per-cell loss is MSE on the hidden bins, with 3 fixed masks and 25% of genes hidden. Same 50-donor cohorts, nulls, and attacks as Geneformer. Trial 1 runs all 12 configs; trials 2-3 run head, last2, full, and LoRA r8 and r64 (trials 2-3 for the rest are queued). Learning rates are 1e-4, or 1e-3 for LoRA, with early stopping (patience 3) on auxiliary donors. Source: `results/metrics_scgpt.csv` (pooled AUCs).

**Controls:**
- **Positive control** (10 donors, full, 30 epochs at lr 1e-4): seen AUC 1.00 on both datasets, so the attack works when scGPT overfits.
- **Earlier failure:** the first OneK1K positive control at lr 5e-4 diverged and was rerun; see the learning log.
- **Nulls:** 0.50-0.59 on OneK1K and 0.45-0.48 on AIDA, all within the stop rule.

### 50 donors, calibrated attack, mean ± SD over 3 trials

| Config | Parameters | OneK1K seen | OneK1K unseen | AIDA seen | AIDA unseen | Holdout MSE OneK1K / AIDA |
|---|---|---|---|---|---|---|
| Base | 0 | n/a | n/a | n/a | n/a | 224.7 / 226.3 |
| Null (unrelated donors) | 50.8M | 0.54±0.05 | 0.53±0.05 | 0.46±0.01 | 0.45±0.06 | 119.3 / 136.8 |
| Head (decoder) | 526k | 0.55±0.05 | 0.53±0.06 | 0.48±0.02 | 0.46±0.07 | 117.8 / 135.9 |
| LoRA r8 | 590k | 0.56±0.05 | 0.53±0.07 | 0.51±0.00 | 0.46±0.06 | 117.4 / 135.1 |
| Last 2 | 3.16M | 0.55±0.05 | 0.53±0.06 | 0.49±0.02 | 0.46±0.07 | 117.6 / 135.6 |
| LoRA r64 | 4.72M | 0.55±0.05 | 0.53±0.07 | 0.50±0.00 | 0.46±0.07 | 118.2 / 136.4 |
| Full | 50.8M | 0.63±0.09 | 0.53±0.06 | 0.59±0.03 | 0.46±0.06 | 117.8 / 135.9 |

The trial-1-only configs (last1, last4, LoRA r1/2/4/16/32) are all within 0.52-0.54 seen on OneK1K and 0.47-0.50 on AIDA.

### Reading

- **At 50 donors, scGPT barely leaks.** Only full fine-tuning rises above its null: about +0.09 on OneK1K and +0.13 on AIDA, where the permutation SD is about 0.06. Every other config is indistinguishable from the null. Geneformer at the same size and learning rate gives 1.00 for full and 0.65-0.90 for the rest.
- **Fine-tuning is almost entirely recalibration.** Every config, *including the null trained on unrelated donors*, cuts holdout MSE from about 225 to 117-119 (OneK1K) or 135-137 (AIDA), and the configs differ by less than 2%. Training on the members teaches scGPT nothing about them that unrelated donors don't.
- **Training length differs from Geneformer.** Validation loss plateaus after the first epoch or two, so early stopping ends scGPT runs after 4-17 epochs. Geneformer runs often used all 30. So "scGPT leaks less" partly means "scGPT trains for less time before it stops improving". The positive control shows that 30 epochs on 10 donors does make it memorize.
- **Pretraining:** all these donors were in scGPT's pretraining data (see the pretraining section). The AUCs here measure only what fine-tuning adds on top of that.

### Caveats

- **Weak base model.** Base predictions are nearly constant (Pearson 0.37 with the truth). That might reflect the model, or a mismatch between our preprocessing and scGPT's unpublished pipeline. A better-matched input pipeline could make the base model stronger and change the leakage picture.
- **Different objective and score scale from Geneformer.** scGPT uses MSE on bins; Geneformer uses cross-entropy on gene ranks. Cross-model comparisons are qualitative.
- **Pooled AUCs only.** Stratified (within-country or within-pool) scoring has not been run for scGPT.

## Open questions and next ideas

1. **Fixed-epoch scGPT vs Geneformer.** Train both for the same number of epochs (no early stopping) at 50 donors. This separates "scGPT resists memorization" from "scGPT stopped early".
2. **Why AIDA's null sits at about 0.7 at 10 donors.** Inspect the 10-donor splits for systematic member/holdout differences (age, sex, country mix, cell counts).
3. **Shadow-model (LiRA-style) attack.** It is in the protocol but not run yet. It could pick up donor-level signal that the calibrated loss misses, especially at 10-20 donors.
4. **Per-cell-type and subgroup leakage.** Which cell types carry the seen-cell signal? Do AIDA countries or age groups differ?
5. **Label-only attacks for Phase 5.** The current attacks use class probabilities, which is a strong assumption for a deployed classifier.
6. **Learning-rate-matched grid.** Rerun LoRA at 3e-4 and head at 3e-4 (their validation-best rates) across donor counts, so cross-family comparisons are fair.
7. **Pretraining membership.** Pretraining-membership attacks need a never-seen group from the same pipeline. One option: a CELLxGENE dataset published after 2023-05 with matched tissue and chemistry, used as non-members for scGPT. It would still be cross-dataset, so it would need careful batch controls.
8. **Better scGPT input pipeline.** Try to match scGPT's preprocessing (normalisation, HVG selection, binning) so the base model's predictions are less constant, then rerun one cohort to see if leakage changes.
9. **HFRA** (third dataset in the protocol) has not been run.
