"""Which cell types and donor subgroups carry the leakage? (CPU, saved Phase 4 scores)

Per cell type: donor score = mean calibrated score (base loss - fine-tuned loss) over that donor's
cells of the type; AUC members vs holdout among donors with >= MIN_CELLS cells of the type.
Per subgroup (AIDA only: country, sex, age band; the cleaned OneK1K file has no donor metadata): AUC using only members
and holdout donors inside the subgroup, with all of their cells.
Configs full, last2, LoRA r8, head and the null; 50, 100, and 200 donors; trials 1-5 where saved.
Outputs: results/celltype_auc.csv, results/subgroup_auc.csv (aggregates only).
"""
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, f"{REPO}/src"]
import phase1_pilot as p1  # noqa: E402

LABELS = ["full", "last2", "lora_r8", "head", "null_unrelated_cohort"]
NDS, TRIALS, MIN_CELLS, MIN_DONORS = [50, 100, 200], [1, 2, 3, 4, 5], 5, 5


def auc(mem, hold):
    if len(mem) < MIN_DONORS or len(hold) < MIN_DONORS:
        return np.nan
    return roc_auc_score(np.r_[np.ones(len(mem)), np.zeros(len(hold))], np.r_[mem, hold])


def main():
    ct_rows, sg_rows = [], []
    for name in ("ok", "aida"):
        obs = ad.read_h5ad(os.path.expanduser(f"~/data/scMAMAMIA/{name}/full_dataset_cleaned.h5ad"), backed="r").obs
        donors = obs["individual"].astype(str).to_numpy()
        ctype = obs["cell_type"].astype(str).to_numpy()
        dob = obs.drop_duplicates("individual").set_index("individual")
        groups = {}
        if name == "aida":  # the cleaned OneK1K file has no age, sex, or batch metadata
            age = pd.to_numeric(dob["age"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
            groups = {"sex": dob["sex"].astype(str), "country": dob["Country"].astype(str),
                      "age_band": pd.cut(age, [0, 35, 50, 200], labels=["<35", "35-50", ">50"]).astype(str)}
        for nd in NDS:
            for trial in TRIALS:
                try:
                    split = p1.load_split(name, nd, trial)
                except FileNotFoundError:
                    continue
                cells = p1.select_cells(donors, split, np.random.default_rng([0, nd, trial]), p1.CELL_CAP,
                                        p1.VAL_CELL_CAP)
                for label in LABELS:
                    f = f"{REPO}/scores/{name}_{nd}d_t{trial}_{label}.npz"
                    if not os.path.exists(f):
                        continue
                    z = np.load(f)
                    s = pd.Series(z["base"] - z["loss"], index=z["cell"])
                    for threat, key in (("seen_cells", "fit"), ("same_donor_unseen_cells", "unseen")):
                        mem = {d: v for d, v in cells[key].items() if len(v)}
                        hold = cells["holdout"]
                        # per cell type
                        for t in np.unique(ctype[s.index]):
                            def per(group):
                                out = []
                                for v in group.values():
                                    v = v[ctype[v] == t]
                                    if len(v) >= MIN_CELLS:
                                        out.append(s.loc[v].mean())
                                return np.array(out)
                            m, h = per(mem), per(hold)
                            ct_rows.append({"dataset": name, "nd": nd, "trial": trial, "config": label,
                                            "threat_model": threat, "cell_type": t, "auc": auc(m, h),
                                            "n_member_donors": len(m), "n_holdout_donors": len(h)})
                        # per subgroup
                        dm = {d: s.loc[v].mean() for d, v in mem.items()}
                        dh = {d: s.loc[v].mean() for d, v in hold.items()}
                        for gname, g in groups.items():
                            for level in sorted(set(g.loc[list(dm) + list(dh)])):
                                m = np.array([x for d, x in dm.items() if g.loc[d] == level])
                                h = np.array([x for d, x in dh.items() if g.loc[d] == level])
                                sg_rows.append({"dataset": name, "nd": nd, "trial": trial, "config": label,
                                                "threat_model": threat, "grouping": gname, "level": level,
                                                "auc": auc(m, h), "n_member_donors": len(m),
                                                "n_holdout_donors": len(h)})
        print(f"{name} done", flush=True)
    pd.DataFrame(ct_rows).to_csv(f"{REPO}/results/celltype_auc.csv", index=False)
    pd.DataFrame(sg_rows).to_csv(f"{REPO}/results/subgroup_auc.csv", index=False)
    print("CELLTYPE/SUBGROUP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
