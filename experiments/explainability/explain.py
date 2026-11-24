"""How do the models that pass cv.py make their predictions? Out-of-fold SHAP for chosen configurations.

Every CV fold model (10 repeats x 5 folds = 50 fits, same folds as cv.py) explains only its own test rows, so every
SHAP value is for a substrate the model did not see. Written to results/explain/<target>__<table>__<model>/:
- shap_oof.csv      per substrate: SHAP of every feature, averaged over the 10 repeats (units: mAh/g or F/g, as capacity_max)
- importance.csv    per feature: readable name, family, mean |SHAP|, share of the total, direction (Spearman of the
                    feature value with its SHAP: > 0 means more of it -> higher predicted capacitance), number of
                    substrates that have it, rho_heavy_atoms (Spearman with heavy-atom count, i.e. is it a size proxy),
                    top10_freq (share of the 50 fold fits where it is in the top 10: stability), and for ECFP bits the
                    substructures that set the bit
- family_share.csv  share of mean |SHAP| per feature family

Run from the repo root:  PYTHONPATH=. .venv/bin/python experiments/explainability/explain.py --config capacity:rf_best:random_forest ...
"""

import argparse
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from experiments.electrocatalysis.final_model import ecfp_environments
from experiments.explainability.common import TARGET, WORK_DIR, cv_splits, family, fit_predict, load_data, load_table, readable, size_residual
from experiments.explainability.cv import N_REPEATS

parser = argparse.ArgumentParser()
parser.add_argument("--config", nargs="+", required=True, help="target:table:model, e.g. capacity:rf_best:random_forest")


def fold_shap(model_name: str, model: object, X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """SHAP values of the test rows (inputs already scaled)."""
    if model_name in ("random_forest", "xgboost"):
        return shap.TreeExplainer(model).shap_values(X_test)
    if model_name in ("bayes", "lasso", "linear"):
        return shap.LinearExplainer(model, X_train).shap_values(X_test)
    # model-agnostic fallback (KNN, TabPFN): permutation explainer with the training fold as background
    explainer = shap.PermutationExplainer(model.predict, shap.maskers.Independent(X_train, max_samples=50))
    return explainer(X_test, max_evals=2 * X_test.shape[1] + 1).values


def spearman_or_nan(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation, NaN when either side is constant."""
    return float(spearmanr(a, b).statistic) if np.std(a) > 0 and np.std(b) > 0 else np.nan


if __name__ == "__main__":
    args = parser.parse_args()
    data = load_data()
    y_cap = data[TARGET].to_numpy(dtype=float)
    heavy = data["heavy_atoms"].to_numpy(dtype=float)
    ys = {"capacity": y_cap, "size_resid": size_residual(y_cap, heavy)}
    repeats = cv_splits(y_cap, n_repeats=N_REPEATS)
    envs = None

    for config in args.config:
        target, table, model_name = config.split(":")
        X_df = load_table(table)
        X, y, n_feat = X_df.to_numpy(dtype=float), ys[target], X_df.shape[1]
        values = np.zeros((N_REPEATS, len(y), n_feat))
        fit_importance = []  # mean |SHAP| of each fold fit, for stability
        for r, folds in enumerate(repeats):
            for train_idx, test_idx in folds:
                _, model, scaler = fit_predict(model_name, X[train_idx], y[train_idx], X[test_idx])
                sv = fold_shap(model_name, model, scaler.transform(X[train_idx]), scaler.transform(X[test_idx]))
                values[r, test_idx] = sv
                fit_importance.append(np.abs(sv).mean(axis=0))
        shap_oof = values.mean(axis=0)
        fit_importance = np.array(fit_importance)
        top10 = np.zeros(n_feat)
        for imp in fit_importance:
            top10[np.argsort(-imp)[:10]] += 1

        mean_abs = np.abs(values).mean(axis=(0, 1))
        importance = pd.DataFrame(
            {
                "feature": X_df.columns,
                "name": [readable(c) for c in X_df.columns],
                "family": [family(c) for c in X_df.columns],
                "mean_abs_shap": mean_abs,
                "share": mean_abs / mean_abs.sum(),
                "direction": [spearman_or_nan(X[:, j], shap_oof[:, j]) for j in range(n_feat)],
                "n_present": (X > 0).sum(axis=0),
                "rho_heavy_atoms": [spearman_or_nan(X[:, j], heavy) for j in range(n_feat)],
                "top10_freq": top10 / len(fit_importance),
            }
        ).sort_values("mean_abs_shap", ascending=False)

        ecfp = importance["feature"].str.fullmatch(r"ecfp_\d+")
        if ecfp.any():
            if envs is None:
                envs = ecfp_environments(data["smiles"].tolist(), "full")
            bits = importance.loc[ecfp, "feature"].map(lambda f: int(re.match(r"ecfp_(\d+)", f).group(1)))
            importance.loc[ecfp, "substructures"] = bits.map(lambda b: " | ".join(f"{s} ({n})" for s, n in sorted(envs.get(b, Counter()).items(), key=lambda kv: (-kv[1], kv[0]))[:3]))

        out_dir = os.path.join(WORK_DIR, "explain", f"{target}__{table}__{model_name}")
        os.makedirs(out_dir, exist_ok=True)
        pd.concat([data[["smiles", TARGET, "heavy_atoms", "is_ctf"]], pd.DataFrame(shap_oof, columns=X_df.columns)], axis=1).to_csv(os.path.join(out_dir, "shap_oof.csv"), index=False)
        importance.to_csv(os.path.join(out_dir, "importance.csv"), index=False)
        fam = importance.groupby("family")["share"].sum().sort_values(ascending=False)
        fam.to_csv(os.path.join(out_dir, "family_share.csv"))

        print(f"\n=== {config}   families: {fam.round(2).to_dict()}")
        cols = ["name", "mean_abs_shap", "share", "direction", "n_present", "rho_heavy_atoms", "top10_freq"] + (["substructures"] if ecfp.any() else [])
        with pd.option_context("display.width", 250, "display.max_colwidth", 70):
            print(importance[cols].head(12).round(3).to_string(index=False))
