#!/usr/bin/env bash
# Waits for the Geneformer fill chain, then fills scGPT 50-donor settings to 3 trials.
cd /home/golobs/MIAsOnFineTunedSingleCell || exit 1
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 MKL_NUM_THREADS=10
until grep -q "fill chain complete" logs/chain.status; do sleep 300; done
echo "$(date -u +%FT%TZ) start fill_trials_scgpt" >> logs/chain.status
taskset -c 0-11 conda run --no-capture-output -n scfm_ python scripts/fill_trials_scgpt.py > logs/fill_trials_scgpt.log 2>&1
echo "$(date -u +%FT%TZ) fill_trials_scgpt exit $?" >> logs/chain.status
echo "$(date -u +%FT%TZ) all queued work complete" >> logs/chain.status
