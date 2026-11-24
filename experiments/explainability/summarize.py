"""Summarise cv.py into one table: results/summary.csv.

Per target x table x model:
- rmse, r2, spearman          mean over the 10 CV repeats
- p_r2, p_spearman            permutation p-value: share of shuffled-target runs at least as good as the real run
                              (both averaged over the same first NULL_REPEATS repeats); (1 + k) / (1 + n_perm)
- q_r2, q_spearman            Benjamini-Hochberg over every configuration of the target (we pick winners from ~60)
- d_rmse_vs_size              mean per-repeat RMSE minus that of BRR on heavy-atom count (same folds); < 0 is better
- win_vs_size                 share of the 10 repeats in which the model has lower RMSE than the size baseline
- spearman_ctf, spearman_nonctf   Spearman of out-of-fold predictions within the 29 CTFs and within the 85 others,
                              so a model that only separates the two classes does not count as ranking
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.explainability.common import MODEL_NAMES, TARGET, WORK_DIR, load_data, size_residual
from experiments.explainability.cv import NULL_REPEATS


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    out = np.empty_like(q)
    out[order] = q
    return out


def within(y: np.ndarray, oof: np.ndarray, mask: np.ndarray) -> float:
    """Mean over repeats of the Spearman correlation inside one group of rows."""
    return float(np.mean([spearmanr(y[mask], p[mask]).statistic for p in oof]))


if __name__ == "__main__":
    data = load_data()
    y_cap = data[TARGET].to_numpy(dtype=float)
    ys = {"capacity": y_cap, "size_resid": size_residual(y_cap, data["heavy_atoms"].to_numpy(dtype=float))}
    ctf = data["is_ctf"].to_numpy()
    rows = []
    for target in sorted(os.listdir(os.path.join(WORK_DIR, "cv"))):
        y = ys[target]
        size_dir = os.path.join(WORK_DIR, "cv", target, "size", "bayes")
        size_rmse = pd.read_csv(os.path.join(size_dir, "metrics.csv"))["rmse"].to_numpy()
        for table in sorted(os.listdir(os.path.join(WORK_DIR, "cv", target))):
            for model in sorted(os.listdir(os.path.join(WORK_DIR, "cv", target, table))):
                d = os.path.join(WORK_DIR, "cv", target, table, model)
                if not os.path.exists(os.path.join(d, "null.csv")):
                    continue
                real, null, oof = pd.read_csv(os.path.join(d, "metrics.csv")), pd.read_csv(os.path.join(d, "null.csv")), np.load(os.path.join(d, "oof.npy"))
                head = real.iloc[:NULL_REPEATS].mean()
                row = {"target": target, "table": table, "model": MODEL_NAMES[model], "n_perm": len(null)}
                row |= real.mean().round(4).to_dict()
                row["spearman_sd"] = round(real["spearman"].std(), 4)
                row["p_r2"] = (1 + (null["r2"] >= head["r2"]).sum()) / (1 + len(null))
                row["p_spearman"] = (1 + (null["spearman"] >= head["spearman"]).sum()) / (1 + len(null))
                row["d_rmse_vs_size"] = round(float(np.mean(real["rmse"].to_numpy() - size_rmse)), 2)
                row["win_vs_size"] = float(np.mean(real["rmse"].to_numpy() < size_rmse))
                if model != "median":
                    row["spearman_ctf"], row["spearman_nonctf"] = round(within(y, oof, ctf), 3), round(within(y, oof, ~ctf), 3)
                rows.append(row)
    summary = pd.DataFrame(rows)
    # the median model's Spearman is an artefact (a constant per fold), so it is left out of the tests
    summary.loc[summary["model"] == "Median", ["spearman", "p_spearman"]] = np.nan
    for target, idx in summary.groupby("target").groups.items():
        for col in ["r2", "spearman"]:
            ok = summary.loc[idx, f"p_{col}"].notna()
            sel = summary.loc[idx][ok].index
            summary.loc[sel, f"q_{col}"] = bh(summary.loc[sel, f"p_{col}"].to_numpy())
    summary = summary.sort_values(["target", "r2"], ascending=[True, False])
    summary.to_csv(os.path.join(WORK_DIR, "summary.csv"), index=False)
    cols = ["target", "table", "model", "rmse", "r2", "spearman", "p_r2", "q_r2", "q_spearman", "d_rmse_vs_size", "win_vs_size", "spearman_ctf", "spearman_nonctf"]
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        for target, df in summary.groupby("target"):
            print(f"\n=== {target}")
            print(df[cols].head(25).round(3).to_string(index=False))
