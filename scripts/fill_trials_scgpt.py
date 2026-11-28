"""Bring every scGPT 50-donor setting to 3 trials: adds trials 2-3 for last1, last4, LoRA r=1/2/4/16/32.

Same cohorts, seeds, and nulls as run_scgpt.py (nulls already exist, so they are skipped).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_scgpt as r6  # noqa: E402

EXTRA = [c for c in r6.ALL_CONFIGS if c not in r6.CORE_CONFIGS]

if __name__ == "__main__":
    for trial in (2, 3):
        for name in r6.DATASETS:
            r6.run_cohort(r6.ScData(name), trial, EXTRA, r6.CSV)
    print("SCGPT FILL COMPLETE", flush=True)
