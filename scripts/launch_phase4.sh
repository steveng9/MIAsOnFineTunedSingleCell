#!/usr/bin/env bash
# Waits for the replicate trials, then runs a Phase 4 smoke test and the full queue.
# Uses GPU 1 and CPU cores 0-11 only. Exit code 99 means the replicates did not finish cleanly.
cd /home/golobs/MIAsOnFineTunedSingleCell || exit 1
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 MKL_NUM_THREADS=10
PY="taskset -c 0-11 conda run --no-capture-output -n scfm_ python"

until [ -f logs/replicates.exit ]; do sleep 60; done
if [ "$(cat logs/replicates.exit)" != "0" ]; then
    echo 99 > logs/phase4.exit
    exit 0
fi

$PY scripts/run_phase4.py --smoke > logs/phase4_smoke.log 2>&1 \
    && $PY scripts/run_phase4.py >> logs/phase4.log 2>&1
echo $? > logs/phase4.exit
