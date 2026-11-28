"""Why is AIDA's calibrated null about 0.7 at 10 donors? (CPU, saved scores)

For each 10-donor trial: the null model's calibrated donor scores (base loss - null loss), members
vs holdout. Checks: (1) cell-type composition (type-centered scores), (2) donor covariates
(country, sex, age, cells available), (3) the same for 20 donors as a contrast.
Output: results/null_diagnostic_10d.csv (aggregates only).
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

obs = ad.read_h5ad(os.path.expanduser("~/data/scMAMAMIA/aida/full_dataset_cleaned.h5ad"), backed="r").obs
donors = obs["individual"].astype(str).to_numpy()
ctype = obs["cell_type"].astype(str).to_numpy()
dob = obs.drop_duplicates("individual").set_index("individual")
age = pd.to_numeric(dob["age"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
ncell = pd.Series(donors).value_counts()
rows = []
for nd in (10, 20):
    for trial in range(1, 6):
        split = p1.load_split("aida", nd, trial)
        cells = p1.select_cells(donors, split, np.random.default_rng([0, nd, trial]), p1.CELL_CAP, p1.VAL_CELL_CAP)
        z = np.load(f"{REPO}/scores/aida_{nd}d_t{trial}_null_unrelated_cohort.npz")
        s = pd.Series(z["base"] - z["loss"], index=z["cell"])
        centered = s - s.groupby(ctype[s.index]).transform("mean")
        for how, sc in (("mean", s), ("type_centered", centered)):
            dm = {d: sc.loc[v].mean() for d, v in cells["fit"].items()}
            dh = {d: sc.loc[v].mean() for d, v in cells["holdout"].items()}
            y = np.r_[np.ones(len(dm)), np.zeros(len(dh))]
            v = np.r_[list(dm.values()), list(dh.values())]
            ds = list(dm) + list(dh)
            rows.append({"nd": nd, "trial": trial, "aggregation": how, "null_auc_seen": roc_auc_score(y, v),
                         "corr_score_age": np.corrcoef(v, age.loc[ds])[0, 1],
                         "corr_score_ncells": np.corrcoef(v, np.log(ncell.loc[ds]))[0, 1],
                         "member_age": age.loc[list(dm)].mean(), "holdout_age": age.loc[list(dh)].mean(),
                         "member_cells": ncell.loc[list(dm)].median(), "holdout_cells": ncell.loc[list(dh)].median(),
                         "member_female": (dob.loc[list(dm), "sex"].astype(str) == "female").mean(),
                         "holdout_female": (dob.loc[list(dh), "sex"].astype(str) == "female").mean(),
                         "member_countries": dob.loc[list(dm), "Country"].astype(str).value_counts().to_dict(),
                         "holdout_countries": dob.loc[list(dh), "Country"].astype(str).value_counts().to_dict()})
out = pd.DataFrame(rows)
out.to_csv(f"{REPO}/results/null_diagnostic_10d.csv", index=False)
pd.set_option("display.width", 250)
print(out.round(3).to_string())
