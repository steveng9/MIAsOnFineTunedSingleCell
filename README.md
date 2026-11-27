# Membership Inference on Fine-Tuned Single-Cell Foundation Models

When a single-cell foundation model is fine-tuned on a small donor cohort, does it memorize those donors? This project measures how that changes with the number and shape of the trained parameters, and compares fine-tuning with in-context learning, where no weights change.

**Status (2026-11-27):** Phases 1-6 have run for Geneformer V1 (OneK1K and AIDA, 10-200 donors) and for scGPT (50 donors). The scGPT OneK1K runs are being redone after a failed positive control, and 50-donor sweep settings are being filled to 3 trials (`scripts/launch_fill.sh`, progress in `logs/chain.status`). Headline findings are at the top of RESULTS.md. See [RESULTS.md](RESULTS.md).

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
