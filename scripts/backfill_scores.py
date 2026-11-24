"""Re-run 50-donor cohorts whose per-cell scores were not saved (runs before Phase 4 added saving).

Needed for batch-stratified AUCs. Rows go to logs/backfill_metrics.csv, not results/metrics.csv,
so the original rows stay the record. Reruns use the same seeds but GPU kernels are not bit-exact,
so AUCs can differ slightly from the originals; the stratified analysis notes which it used.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, f"{REPO}/src"]
import run_phase4 as r4  # noqa: E402  (importing enables per-cell score saving)
import run_sweep as rs  # noqa: E402

CSV = f"{REPO}/logs/backfill_metrics.csv"


def missing(name, trial):
    tags = [lab + (f"_r{r}" if r else "") for _, r, lab, _ in r4.grid_jobs()] + ["null_unrelated_cohort"]
    return [t for t in tags if not os.path.exists(f"{REPO}/scores/{name}_50d_t{trial}_{t}.npz")]


for name in r4.DATASETS:
    data = None
    for trial in r4.TRIALS:
        gaps = missing(name, trial)
        if not gaps:
            continue
        print(f"backfill {name} 50d t{trial}: missing {gaps}", flush=True)
        data = data or rs.Data(name)
        r4.run_cohort(data, 50, trial, r4.grid_jobs(), CSV, False, with_null=True)
print("BACKFILL COMPLETE", flush=True)
