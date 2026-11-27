#!/usr/bin/env bash
# Waits for the scGPT rerun, then fills 50-donor settings to 3 trials and reruns both analyses.
cd /home/golobs/MIAsOnFineTunedSingleCell || exit 1
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 MKL_NUM_THREADS=10
PY="taskset -c 0-11 conda run --no-capture-output -n scfm_ python"
until [ -f logs/phase6_rerun.exit ]; do sleep 120; done
echo "$(date -u +%FT%TZ) phase6 rerun exit $(cat logs/phase6_rerun.exit)" >> logs/chain.status
step() {
    local name=$1; shift
    echo "$(date -u +%FT%TZ) start $name" >> logs/chain.status
    "$@" > "logs/$name.log" 2>&1
    echo "$(date -u +%FT%TZ) $name exit $?" >> logs/chain.status
}
step fill_trials $PY scripts/fill_trials.py
step analyze_phase4_fill env CUDA_VISIBLE_DEVICES= $PY scripts/analyze_phase4.py
echo "$(date -u +%FT%TZ) fill chain complete" >> logs/chain.status
