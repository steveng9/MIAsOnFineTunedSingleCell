"""Phase 2-3 figure: trainable parameters vs leakage and utility, from results/metrics.csv."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ND, TRIAL = 50, 1
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
FAMILIES = {"Freezing": ("#2a78d6", "o"), "LoRA": ("#eb6834", "s")}
DATASETS = {"ok": "OneK1K", "aida": "AIDA"}
FREEZE = ["head", "last1", "last2", "last4", "full"]

d = pd.read_csv(f"{REPO}/results/metrics.csv")
d = d[(d.nd == ND) & (d.trial == TRIAL) & (d.objective == "O1") & (d.aggregation == "mean")]
cal = d[d.attack == "calibrated_loss"]


def family(frame, name):
    rows = frame[frame.config.isin(FREEZE)] if name == "Freezing" else frame[frame.config == "lora"]
    return rows.sort_values("n_trainable")


plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": GRID, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(3, 2, figsize=(10, 10.5), sharex=True, facecolor=SURFACE)
panels = [("seen_cells", "AUC, attacker holds training cells"),
          ("same_donor_unseen_cells", "AUC, attacker holds unseen cells"),
          (None, "Reduction in holdout loss vs base")]

for col, (ds, title) in enumerate(DATASETS.items()):
    frame = cal[cal.dataset == ds]
    base_loss = d[(d.dataset == ds) & (d.config == "base")].utility.iloc[0]
    for row, (threat, ylabel) in enumerate(panels):
        ax = axes[row, col]
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for name, (color, marker) in FAMILIES.items():
            rows = family(frame if threat is None else frame[frame.threat_model == threat], name)
            if threat is None:
                rows = rows.drop_duplicates("n_trainable")
                y = base_loss - rows.utility
            else:
                y = rows.auc
            ax.plot(rows.n_trainable, y, color=color, marker=marker, markersize=7, linewidth=2,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=name)
        if threat is not None:
            null = frame[(frame.config == "null_unrelated_cohort") & (frame.threat_model == threat)].auc.iloc[0]
            ax.axhline(null, color=MUTED, linestyle="--", linewidth=1.2, label="Unrelated-donor null")
            ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1, label="Chance (0.5)")
            ax.set_ylim(0.35, 1.03)
        else:
            ax.set_ylim(0, 0.18)
            ax.set_xlabel("Trainable parameters (log scale)")
        ax.set_xscale("log")
        if col == 0:
            ax.set_ylabel(ylabel)
        if row == 0:
            ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.955))
fig.suptitle(f"Geneformer V1: leakage and utility vs trainable parameters ({ND} donors, one trial, preliminary)",
             x=0.5, y=0.985, fontsize=12.5, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = f"{REPO}/figures/phase23_params_vs_leakage.png"
fig.savefig(out, dpi=170, facecolor=SURFACE)
print(out)
