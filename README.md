# Membership Inference on Fine-Tuned Single-Cell Foundation Models

When a single-cell foundation model is fine-tuned on a small donor cohort, does it memorize those donors? This project measures how that changes with the number and shape of the trained parameters, and compares fine-tuning with in-context learning, where no weights change.

**Status:** Phases 1-3 complete for Geneformer V1 on OneK1K and AIDA at 50 donors (one trial, preliminary). Replicates stopped on a failed null, traced to a country confound in AIDA. Phase 4 is running, and its AIDA attacks will be scored within country as well as pooled. When it finishes, a chained queue (`scripts/launch_after_phase4.sh`, progress in `logs/chain.status`) runs the Phase 4 analysis, Phase 5 (fine-tuning vs in-context learning), Phase 6 (scGPT), and a score backfill. See [RESULTS.md](RESULTS.md).

## Design

- **Models:** Geneformer first, then scGPT.
- **Data:** OneK1K, AIDA, and HFRA, using the donor splits from scMAMA-MIA.
- **Trained parameters:** base, head only, last-k layers, full, and LoRA with r from 1 to 64.
- **Attacks:** raw loss, base-calibrated loss, and shadow-model (LiRA-style), aggregated to donor-level AUC.
- **Threat models:** the attacker holds the exact training cells, or other unseen cells from the same donor.

The full protocol is in [docs/scfm_finetuning_instructions.md](docs/scfm_finetuning_instructions.md).

## Layout

| Path | Contents |
|---|---|
| `docs/` | Protocol and design notes |
| `src/` | Library code |
| `scripts/` | Audits and experiment entry points |
| `configs/` | Experiment configurations |
| `learn/` | Small runnable teaching snippets |
| `results/` | Tidy metrics CSV |
| `figures/` | Plots |
| `LEARNING_LOG.md` | Running notes on each step |
| `RESULTS.md` | What was run, what was found, and caveats |

## Environment

```bash
conda create -n scfm_ python=3.10 pip
conda activate scfm_
pip install torch transformers peft datasets accelerate anndata scanpy scikit-learn pandas matplotlib loompy tdigest huggingface_hub
```

## Data policy

Expression data, cell sentences, donor information, and model checkpoints stay on the local machine and are excluded by `.gitignore`. Source datasets are treated as read-only.
