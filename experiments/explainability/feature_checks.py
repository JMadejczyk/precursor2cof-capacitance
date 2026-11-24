"""Check the chemistry the SHAP analysis points at directly in the data, without any model.

For each substructure (SMARTS), compare substrates with and without it:
- n_with, n_ctf_with          how many substrates have it (and how many of those are CTFs)
- median_with / median_without   median capacity_max
- d_resid                     median difference of the size residual (capacity minus the straight-line fit on
                              heavy-atom count): the effect once monomer size is accounted for
- p_resid                     Mann-Whitney p-value of that difference (two-sided, no multiple-testing correction:
                              the groups were picked after looking at SHAP, so treat these as descriptive)
- d_resid_nonctf              the same difference within the 85 non-CTF substrates only

Output: results/feature_checks.csv. Run from the repo root with PYTHONPATH=.
"""

import os

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import mannwhitneyu, spearmanr

from experiments.explainability.common import TARGET, WORK_DIR, load_data, size_residual

GROUPS = {
    "aromatic S (thiophene)": "[s]",
    "aromatic N, no H (pyridine/triazine)": "[n;H0;X2]",
    "triazine ring": "c1ncncn1",
    "pyridine ring": "c1ccncc1",
    "hydroxyl (OH)": "[OX2H]",
    "phenol (aromatic OH)": "c[OX2H]",
    "primary amine (NH2)": "[NX3;H2]",
    "imine (C=N, acyclic)": "[CX3]=[NX2;!R]",
    "carbonyl (C=O)": "[CX3]=[OX1]",
    "nitrile (C#N)": "C#N",
    "aldehyde (CHO)": "[CX3H1](=O)",
    "halogen": "[F,Cl,Br,I]",
}

if __name__ == "__main__":
    data = load_data()
    y = data[TARGET].to_numpy(dtype=float)
    heavy = data["heavy_atoms"].to_numpy(dtype=float)
    resid = size_residual(y, heavy)
    ctf = data["is_ctf"].to_numpy()
    mols = [Chem.MolFromSmiles(s) for s in data["smiles"]]
    rows = [
        {
            "group": "heavy-atom count (Spearman)",
            "n_with": len(y),
            "rho": round(float(spearmanr(heavy, y).statistic), 3),
            "rho_ctf": round(float(spearmanr(heavy[ctf], y[ctf]).statistic), 3),
            "rho_nonctf": round(float(spearmanr(heavy[~ctf], y[~ctf]).statistic), 3),
        }
    ]
    for name, smarts in GROUPS.items():
        patt = Chem.MolFromSmarts(smarts)
        has = np.array([m.HasSubstructMatch(patt) for m in mols])
        row = {"group": name, "smarts": smarts, "n_with": int(has.sum()), "n_ctf_with": int((has & ctf).sum())}
        if 3 <= has.sum() <= len(has) - 3:
            row |= {
                "median_with": float(np.median(y[has])),
                "median_without": float(np.median(y[~has])),
                "heavy_with": float(np.median(heavy[has])),
                "heavy_without": float(np.median(heavy[~has])),
                "d_resid": round(float(np.median(resid[has]) - np.median(resid[~has])), 1),
                "p_resid": float(mannwhitneyu(resid[has], resid[~has]).pvalue),
            }
            nc = ~ctf
            if 3 <= (has & nc).sum() <= nc.sum() - 3:
                row["d_resid_nonctf"] = round(float(np.median(resid[has & nc]) - np.median(resid[~has & nc])), 1)
                row["p_resid_nonctf"] = float(mannwhitneyu(resid[has & nc], resid[~has & nc]).pvalue)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(WORK_DIR, "feature_checks.csv"), index=False)
    with pd.option_context("display.width", 250):
        print(out.drop(columns="smarts").round(3).to_string(index=False))
