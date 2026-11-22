# Fine-tuning single-cell foundation models: instructions for Claude

Steven will start you on this machine (2 GPUs, 48 vCPUs) and point you at this file. Read all of it before doing anything.

## 1. What this is and why

Steven is a final-year PhD student in ML privacy. Their published work attacks synthetic single-cell data (scMAMA-MIA, NeurIPS 2026). They have trained attention models from scratch, but **have never fine-tuned a pretrained model**. This project has two goals, in this order:

1. **Steven learns this properly.** They should come out able to explain every step at a whiteboard: tokenization, checkpoints, freezing, LoRA, in-context learning, and why each result came out as it did.
2. **A small, real research result.** When a single-cell foundation model is fine-tuned on a small donor cohort, does it memorize those donors? How does that change with the *number and shape* of the parameters trained, and how does it compare to in-context learning, where no weights change?

This also prepares Steven for a postdoc application (Ai2 CellOLMo), which asks for hands-on fine-tuning and asks why training was the right call over in-context learning. Everything Steven later says about this work must be true and theirs, so **the learning goal is not optional and the results must be reported honestly.** A null result is a fine result. Do not write application prose unless asked.

## 2. How to work with Steven (read twice)

Two modes.

### Mode A: running experiments
You may run the phases below on your own, without narrating. Keep messages short: what finished, what's next, anything that needs a decision. **Do not explain concepts in this mode unless asked.**

While you work, keep `LEARNING_LOG.md` in the workspace. It is your teaching notes for later, and it should be written as you go, not reconstructed. For each step, record:
- what you did and the exact command or file
- why you chose it over the alternatives
- what broke and how you found out
- what surprised you
- a real number, tensor shape, or example worth showing Steven later

### Mode B: teaching (when Steven asks, e.g. "walk me through it")
This is an **interactive conversation, not a report.**

- **One idea per message.** Usually under about 150 words plus one small example. Then stop and wait.
- **Never dump.** No long summaries, no multi-section explanations, no "here's everything we did." If Steven asks a broad question, answer the first piece and ask where to go next.
- **Leave room.** End most messages with a real opening: a question, or "want to look at that, or move on?" Don't end every one with a quiz.
- **Ask Steven to predict before you reveal.** "This matrix is 512×512 and r=8. How many parameters does the adapter train?" or "Do you expect AUC to go up or down here?" Prediction is where the learning happens. If they're wrong, get curious about why rather than just correcting.
- **Use real objects.** Show actual tokens, actual `.shape` output, an actual parameter count from *our* checkpoint. Prefer a 10-line runnable snippet in `learn/` that Steven can edit and break over a paragraph. Steven loves playing with the matrix dimensions of neural nets, so lean into it.
- **Follow tangents.** Steven's questions set the route. The curriculum in section 9 is a map, not a script.
- **Calibrate.** Steven already knows MIAs, AUC, DP, copulas, scRNA-seq data, AnnData, PyTorch, attention, and GNNs. Do not explain those. Build on them ("this is like the auxiliary data in scMAMA-MIA, except…"). They do *not* yet know the fine-tuning workflow, HuggingFace conventions, LoRA, or LLM vocabulary, so define each term the first time.
- **Say what you don't know,** and check rather than guess. If you made a mistake earlier, say so plainly.
- Plain language. No hype, no flattery.

## 3. Ground rules for this machine

- **Check before using resources.** Run `nvidia-smi`, `free -h`, `df -h`, and look for others' running jobs. Do not kill or disturb any process you did not start. If the GPUs are busy, tell Steven.
- **Existing data is read-only.** Never write to, move, or delete anything under `~/data/` or `~/scRNA-seq_privacy_audits/`. Import or copy from them.
- **Work only in `~/scfm_finetune/`** with its own conda environment. Do not modify existing environments (`camda_conda`, `tabddpm_`, etc.). Use `git init` locally; do not push anywhere.
- **Keep cell data on this machine.** Do not send any expression data, cell sentences, or donor information to external APIs or services. Downloading public checkpoints and code is fine. All LLMs in Phase 5 run locally.
- **Memory.** AIDA's h5ad is 57 GB, so always open it with `backed='r'`. Earlier fork-based workers OOM-killed this box. Never densify a full matrix.
- **Disk.** Save LoRA adapters and final metrics. Don't keep every full checkpoint.
- One job per GPU via `CUDA_VISIBLE_DEVICES`. Roughly 20 CPU workers per job at most. Run long jobs in `tmux` or `nohup` with logs in `logs/`.
- Fix seeds and record them, plus package versions and model revision hashes.

## 4. Existing assets

Verify these; they come from a repo snapshot and may have moved.

- Repo: `~/scRNA-seq_privacy_audits/` (read its `CLAUDE.md` for context; much of it concerns older revision work).
- Data: `~/data/scMAMAMIA/{ok,aida,cg}/full_dataset_cleaned.h5ad`.
  - `ok` = OneK1K, blood, ~1.26M cells, 981 donors, 14 cell types
  - `aida` = AIDA, Asian immune atlas, ~1.06M cells, 508 donors, 33 cell types, has sex and ethnicity metadata
  - `cg` is probably HFRA (fetal retina, ~108K cells, 22 donors). **Confirm.**
- Donor splits: `~/data/scMAMAMIA/{dataset}/splits/{nd}d/{trial}/{train,holdout,auxiliary}.npy`. **Reuse these** so results line up with scMAMA-MIA. Confirm they are disjoint.
- Convention: donor-level ROC AUC, averaged over 5 trials, at donor counts {2, 5, 10, 20, 50, 100, 200}.

**Check before tokenizing:** does `full_dataset_cleaned.h5ad` hold **raw counts** and **all genes** (not only HVGs, not log-normalized)? Geneformer needs raw counts and Ensembl IDs. If the file doesn't qualify, stop and tell Steven what you found.

## 5. Models

Use the official sources, and read their current docs rather than trusting this summary.

- **Geneformer** (Theodoris et al., Nature 2023), `ctheodoris/Geneformer` on HuggingFace. Start here. Cells become genes ranked by normalized expression; pretrained with masked-gene prediction.
  - **Pitfall:** V1 (~30M cells) and V2 (~95–104M cells) use *different* gene dictionaries, median files, input lengths, and special tokens. The tokenizer and checkpoint must match. Pick one (prefer the smaller for the pilot) and record which.
- **scGPT** (Cui et al., Nature Methods 2024), `bowang-lab/scGPT`, whole-human checkpoint. Do this second. Tokens are genes plus binned expression. Its install can be fragile.
  - **Pitfall:** it may use `nn.TransformerEncoder`/`MultiheadAttention` or flash attention, whose internal projections are not ordinary `nn.Linear` modules, so off-the-shelf LoRA may silently not attach to them. Inspect and count trainable parameters.

## 6. Experimental design

### Prior work to read first
- *Auditing pretraining contamination in single-cell foundation model benchmarks* (arXiv:2607.20572). Closest known work: a loss-based MIA on Geneformer, scGPT, and UCE, asking whether *benchmarks were in pretraining*. Our question differs (donors in *fine-tuning*).
- *SoK: Reducing the Vulnerability of Fine-tuned Language Models to MIAs* (arXiv:2403.08481).
- Hu et al., *LoRA* (2021). Carlini et al., *LiRA* (2022).

Run your own literature check too. **Do not claim "first" for anything** without a proper search.

### The pretraining question
OneK1K and AIDA may be in the pretraining corpora. Check as far as you can and record what you find. Because members and non-members come from a **random split of donors in the same dataset**, pretraining exposure should be balanced between them, so it doesn't by itself bias the member-vs-holdout AUC. It could still shrink the signal, since the base model may already fit these donors well. Keep this distinction clear, and treat "was this donor in pretraining?" as a separate question.

### Donor and cell splits
For each (dataset, nd, trial):
- **Members** = `train` donors. **Non-members** = `holdout` donors. **Auxiliary** = `auxiliary` donors.
- Split each member's cells into **fit cells** (used in fine-tuning) and **unseen cells** (never used). This gives two threat models:
  - *Seen cells:* the attacker holds the exact training cells.
  - *Same donor, unseen cells:* the attacker holds other cells from that person. This is the more interesting one. scMAMA-MIA found generators remember the donor rather than the cell, and we want to know if that holds here.
- **Holdout donors must not influence training.** Early stopping and hyperparameter tuning use a validation slice of *auxiliary* donors only.
- Cap cells per donor for attack scoring (e.g. 200), so donors with many cells don't dominate.

### Two fine-tuning objectives
- **O1, continued pretraining:** same masked-gene objective, on member cells. Primary.
- **O2, task head:** add a classifier for cell type and train it. Utility is macro-F1 on holdout donors.

### What gets trained (the main axis)
| Config | Trainable |
|---|---|
| Base | nothing (null control) |
| Head only | classifier/output head |
| Last-k | last k transformer layers (k = 1, 2, …) |
| Full | all weights |
| LoRA | low-rank adapters, r ∈ {1, 2, 4, 8, 16, 32, 64} |

**LoRA:** freeze W and learn `W_new = W + B·A`. For W of shape `(m, n)`: B is `(m, r)`, A is `(r, n)`, so the adapter has `r(m+n)` parameters instead of `mn`.
- **Write a small `LoRALinear` yourself first** so Steven can read it. Then compare against HuggingFace PEFT on the same config to confirm they agree.
- Initialize B to zero, so the model starts *exactly* at the pretrained one. **Test this:** outputs must be identical before training.
- Print a table of trainable parameters per config, and assert the counts match the formula.
- Secondary sweep if time allows: *where* the adapter goes (attention projections vs. feed-forward vs. last layers only).

### Attacks, simplest first
Score each target cell, then aggregate to a donor score.
1. **Raw loss** on the fine-tuned model.
2. **Calibrated loss:** `L_base(x) − L_finetuned(x)`, using the *same fixed masks* for both (seed masks by cell ID; average over several masks). The base checkpoint is public, so this is a realistic attacker.
3. **Shadow / LiRA-style:** fine-tune several shadow models on auxiliary donors, and calibrate with them. Expensive, so do it on a few chosen configs only.

Aggregate by donor mean; also try the median and an equal-weight-per-cell-type mean.

Report: donor-level AUC (mean ± std over trials), and TPR at low FPR **where the number of donors can resolve it** (with 25 non-members the smallest nonzero FPR is 4%, so don't report TPR@1%). Pool across trials when appropriate and say so.

### Required controls (run these before any sweep)
- **Null:** the attack on the *unmodified* base model must give AUC ≈ 0.5.
- **Positive:** deliberately overfit (few donors, many epochs, full weights). AUC should approach 1. If it doesn't, the attack or pipeline is broken.
- **Label shuffle:** permuted member labels give AUC ≈ 0.5.
- **Leakage check:** assert no holdout donor's cell appears in fine-tuning or validation data.

If a control fails, stop and debug. Don't run sweeps on a broken pipeline.

### Phase 5: in-context learning comparison
In-context learning means adapting through inputs at inference, with **no weight updates**. These models don't follow prompts, so compare two honest versions:

- **ICL-A, reference set in embedding space.** Freeze the base model, embed a labeled "support set" of member cells, and classify new cells by nearest neighbors or prototypes. This is the closest analogue for Geneformer and scGPT.
- **ICL-B, real few-shot prompting.** Run an open LLM **locally** on "cell sentences" (genes in rank order) with k labeled examples in the prompt and a constrained list of cell-type answers. Candidates: a C2S-Scale model (Cell2Sentence) and an OLMo model, since the target lab builds OLMo. Check licenses and whether GPU memory allows it. If ICL-B turns into a time sink, drop it and tell Steven.

Compare on the same holdout donors, at k ∈ {0, 1, 5, 10, 50} per cell type:
- utility (macro-F1)
- training and inference cost
- what data must be retained at deployment (ICL keeps the support cells around)
- privacy: if the support set comes from member donors, can an attacker with query access tell? No weights memorize anything here, but outputs still depend on the support set. Attack it the same way, including with same-donor unseen cells.

The question to be able to answer: **what did changing the weights buy over not changing them, and what did it cost in privacy?**

## 7. Phases and gates

Don't start a phase until the previous gate passes.

- **Phase 0, setup.** Workspace, env, GPU check, data audit (raw counts? Ensembl IDs? splits disjoint? overlap with pretraining?). *Gate:* one cell goes through the tokenizer and model, and you can print its tokens and output shapes.
- **Phase 1, pilot on Geneformer.** OneK1K, 50 donors, trial 1, O1, a small subset. *Gate:* null, positive, and shuffle controls all behave.
- **Phase 2, freezing axis.** Base / head / last-k / full. Plot leakage vs. utility.
- **Phase 3, LoRA.** Rank sweep. Main figure: **trainable parameters (log) vs. AUC, and vs. utility**, one curve per config family.
- **Phase 4, scale up.** Donor counts {10, 20, 50, 100, 200}, 5 trials, then AIDA. Per-cell-type AUC and, for AIDA, subgroups: does the "who is exposed first" pattern from Steven's earlier work appear? Include O2.
- **Phase 5, in-context comparison** (above).
- **Phase 6, scGPT.** Repeat the core of Phases 1–3. Do the two models agree?

**Check in with Steven after Phase 1** with a short message and the control results, before spending serious GPU time. Also stop and ask if: data isn't what's expected, a run would take over ~24 hours, or a design choice would change the question. Otherwise proceed. Optional later, only if Steven wants it: LoRA + donor-level DP-SGD.

## 8. Workspace and outputs

```
~/scfm_finetune/
  LEARNING_LOG.md      # your running teaching notes
  RESULTS.md           # short, honest: what was run, what was found, caveats
  learn/               # tiny runnable teaching snippets
  src/  configs/  logs/
  results/             # one tidy CSV: model, version, dataset, nd, trial, objective,
                       #   config, rank, n_trainable, threat_model, attack, aggregation,
                       #   auc, tpr_at_fpr, utility, seed, runtime
  figures/
```

Every number in `RESULTS.md` must trace to a row in the CSV. Mark anything preliminary. Do not round a weak effect into a claim.

## 9. Teaching map (for Mode B)

Rough order; Steven's questions override it.

1. What a pretrained checkpoint actually is: open it, list the layers, count the parameters.
2. A cell becomes a sentence: take one real cell through normalization, ranking, and token IDs.
3. What the model was trained to do: mask a gene, see what it guesses.
4. What "fine-tuning" changes: `requires_grad`, the optimizer, why the learning rate is low.
5. Freezing: which matrices move, and how far they moved (`‖W_after − W_before‖` per layer).
6. LoRA from scratch: dimensions, parameter counts, why B starts at zero, what rank limits. Try the SVD of a real full fine-tuning update `ΔW`: *is it already approximately low-rank?* That's a good experiment for Steven to run themself.
7. Measuring memorization: the base model as reference, then the controls and why each exists.
8. Results: leakage vs. parameters. What did Steven predict, and what happened?
9. In-context learning: what it can and can't do here, and what it costs.
10. Connection to Steven's research: frozen public weights plus a small private update resembles "learn shared structure from public data, spend the budget on the deviation."
11. Practice questions: Steven explains it back, as in an interview. For example:
    - Why fine-tune instead of prompting?
    - What does rank mean here, and how did you choose it?
    - How do you know your attack isn't picking up batch effects?
    - What would you do differently?

## 10. First message to Steven

Keep it to a few lines: confirm you've read this, report GPU/disk availability, state anything that looks off, and say you're starting Phase 0. Don't summarize this file back.
