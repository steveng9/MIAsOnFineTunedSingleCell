#!/usr/bin/env bash
# Runs a Phase 4 smoke test, then the full resumable queue.
# Uses GPU 1 and CPU cores 0-11 only.
cd /home/golobs/MIAsOnFineTunedSingleCell || exit 1
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 MKL_NUM_THREADS=10
PY="taskset -c 0-11 conda run --no-capture-output -n scfm_ python"

rm -f logs/phase4.exit
$PY scripts/run_phase4.py --smoke > logs/phase4_smoke.log 2>&1 \
    && $PY scripts/run_phase4.py >> logs/phase4.log 2>&1
echo $? > logs/phase4.exit
