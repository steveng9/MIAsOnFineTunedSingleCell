#!/usr/bin/env bash
# Follow-up experiments (approved 2026-11-28). GPU 1 and CPU cores 0-11 only. Status in logs/chain.status.
cd /home/golobs/MIAsOnFineTunedSingleCell || exit 1
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 MKL_NUM_THREADS=10
PY="taskset -c 0-11 conda run --no-capture-output -n scfm_ python"
until [ -f logs/exp_pretrain.exit ]; do sleep 60; done
echo "$(date -u +%FT%TZ) exp_pretrain_membership exit $(cat logs/exp_pretrain.exit)" >> logs/chain.status
step() {
    local name=$1; shift
    echo "$(date -u +%FT%TZ) start $name" >> logs/chain.status
    "$@" > "logs/$name.log" 2>&1
    echo "$(date -u +%FT%TZ) $name exit $?" >> logs/chain.status
}
step exp_pretrain_ties_left $PY scripts/exp_pretrain_membership.py --ties-left
step phase5_rerun $PY scripts/run_phase5.py --rerun
step exp_scgpt_v2 $PY scripts/exp_scgpt_v2.py
step exp_lira $PY scripts/exp_lira.py
step exp_fixed_epochs $PY scripts/exp_fixed_epochs.py
step exp_hfra $PY scripts/exp_hfra.py
step exp_lr_matched $PY scripts/exp_lr_matched.py
step analyze_phase4_followups env CUDA_VISIBLE_DEVICES= $PY scripts/analyze_phase4.py
echo "$(date -u +%FT%TZ) followups complete" >> logs/chain.status
