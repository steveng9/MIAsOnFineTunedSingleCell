# Results

All numbers below are **preliminary**. They come from one trial, one seed, and one dataset, and trace to `results/metrics.csv`. Control verdicts are in `results/phase1_gates.json`.

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
