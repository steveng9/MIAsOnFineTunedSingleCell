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
