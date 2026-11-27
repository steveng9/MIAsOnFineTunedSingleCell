"""Bring every 50-donor Geneformer setting to 3 trials (Steven, 2026-11-27: 3 trials per setting is enough).

Phase 2-3 sweep configs outside the Phase 4 grid (last1, last4, LoRA r=1/2/4/16/32) ran trial 1 only,
and the Phase 4 learning-rate check ran trials 1-2. This adds trials 2-3 for the sweep configs and
trial 3 for the LR check, on the same cohorts (nulls for these cohorts already exist).
Rows go to results/metrics.csv; per-cell scores to scores/ (importing run_phase4 enables saving).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, f"{REPO}/src"]
import run_phase4 as r4  # noqa: E402
import run_sweep as rs  # noqa: E402

CSV = f"{REPO}/results/metrics.csv"
SWEEP_EXTRA = [(c, None, c, None) for c in ("last1", "last4")] + \
              [("lora", r, "lora", None) for r in (1, 2, 4, 16, 32)]

if __name__ == "__main__":
    datas = {n: rs.Data(n) for n in r4.DATASETS}
    for trial in (2, 3):
        for n, d in datas.items():
            r4.run_cohort(d, 50, trial, SWEEP_EXTRA, CSV, False, with_null=False)
    for n, d in datas.items():
        r4.run_cohort(d, 50, 3, r4.lr_jobs(), CSV, False, with_null=False)
    print("FILL TRIALS COMPLETE", flush=True)
