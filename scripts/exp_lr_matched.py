"""Learning-rate-matched grid: head and LoRA r8/r64 at 3e-4, their best validation rate in the
Phase 4 LR check (50 donors, 3 trials), at 10, 20, 100, and 200 donors, trials 1-3.

The Phase 4 grid ran head at 1e-4 and LoRA at 1e-3. full and last2 were already at their best
rate (1e-4), and 50 donors already has these rates from the LR check. Uses the same cohorts;
nulls already exist. Rows: results/metrics.csv with labels head_lr0.0003 / lora_lr0.0003.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, f"{REPO}/src"]
import run_phase4 as r4  # noqa: E402
import run_sweep as rs  # noqa: E402

CSV = f"{REPO}/results/metrics.csv"
JOBS = [("head", None, r4.lr_label("head", 3e-4), 3e-4), ("lora", 8, r4.lr_label("lora", 3e-4), 3e-4),
        ("lora", 64, r4.lr_label("lora", 3e-4), 3e-4)]

if __name__ == "__main__":
    datas = {n: rs.Data(n) for n in r4.DATASETS}
    for trial in (1, 2, 3):
        for nd in (10, 20, 100, 200):
            for n, d in datas.items():
                r4.run_cohort(d, nd, trial, JOBS, CSV, False, with_null=False)
    print("LR MATCHED COMPLETE", flush=True)
