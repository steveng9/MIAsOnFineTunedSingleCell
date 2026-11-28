"""scGPT with pretraining-matched binning (ties in the upper bin, as scGPT's preprocess._digitize).

Phase 6 spread tied counts at random between bin edges, so hidden-bin targets for low counts were
partly noise. With pretraining-style ties the base model's Pearson on hidden bins rises from about
0.39 to 0.56 (scripts/test_scgpt_binning.py). This reruns the Phase 6 core: positive controls, then
50 donors, trials 1-3, head, last2, full, LoRA r8 and r64, base, and the null, early stopping as before.
Rows: results/metrics_scgpt_v2.csv; scores in scores/scgpt_v2/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_scgpt as r6  # noqa: E402

r6.TIES = "left"
r6.OUT_TAG = "scgpt_v2"
CSV = os.path.join(os.path.dirname(HERE), "results", "metrics_scgpt_v2.csv")

if __name__ == "__main__":
    active = [d for d in (r6.ScData(n) for n in r6.DATASETS) if r6.positive_control(d, CSV)]
    for trial in (1, 2, 3):
        active = [d for d in active if r6.run_cohort(d, trial, r6.CORE_CONFIGS, CSV)]
        print(f"SCGPT V2 TRIAL {trial} COMPLETE", flush=True)
    print("SCGPT V2 COMPLETE", flush=True)
