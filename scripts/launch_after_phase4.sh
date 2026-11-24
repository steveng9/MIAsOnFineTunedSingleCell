#!/usr/bin/env bash
# Waits for Phase 4, then runs the Phase 4 analysis, Phase 5, Phase 6 (scGPT), the score backfill,
# and the analysis again. Steps are independent, so a failed step is logged and the chain continues.
# Uses GPU 1 and CPU cores 0-11 only. Status lines go to logs/chain.status.
cd /home/golobs/MIAsOnFineTunedSingleCell || exit 1
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 MKL_NUM_THREADS=10
PY="taskset -c 0-11 conda run --no-capture-output -n scfm_ python"

until [ -f logs/phase4.exit ]; do sleep 300; done
echo "$(date -u +%FT%TZ) phase4 exit $(cat logs/phase4.exit)" >> logs/chain.status

step() {
    local name=$1; shift
    echo "$(date -u +%FT%TZ) start $name" >> logs/chain.status
    "$@" > "logs/$name.log" 2>&1
    echo "$(date -u +%FT%TZ) $name exit $?" >> logs/chain.status
}

step analyze_phase4 env CUDA_VISIBLE_DEVICES= $PY scripts/analyze_phase4.py
step phase5 $PY scripts/run_phase5.py
step phase6_scgpt $PY scripts/run_scgpt.py
step backfill_scores $PY scripts/backfill_scores.py
step analyze_phase4_final env CUDA_VISIBLE_DEVICES= $PY scripts/analyze_phase4.py
echo "$(date -u +%FT%TZ) chain complete" >> logs/chain.status
