"""Held-out SHAP for the electrocatalysis (J) model that ranks materials within a paper.

Leave-one-paper-out with the electrocatalysis experiment's own code (corrections, models, SHAP). Each fold
model explains only the materials of the held-out paper. Needs experiments/electrocatalysis/features.py to have
been run. Output in results/electro/<subset>__<method>__<model>__<table>/:
- predictions.csv   per material: paper, J, held-out prediction (on the J scale of an average paper)
- shap_oof.csv      held-out SHAP of every feature (units: within-paper standard deviations of J for zscore)
- importance.csv    mean |SHAP|, direction (Spearman of feature value with SHAP; > 0 = higher, less negative J,
                    i.e. a WORSE catalyst), materials with the feature, share of |SHAP| from the largest paper,
                    importance without that paper, and the substructures behind ECFP bits

Run from the repo root:  PYTHONPATH=. .venv/bin/python experiments/explainability/electro_shap.py
"""

import argparse
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from experiments.electrocatalysis.common import SUBSETS, get_model, load_config, load_data, shap_values, within_paper_accuracy
from experiments.electrocatalysis.corrections import fit_predict
from experiments.electrocatalysis.final_model import ecfp_environments
from experiments.explainability.common import WORK_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--config", nargs="+", default=["clean:zscore:random_forest:a1_full__ecfp"], help="subset:method:model:table")

if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config()
    data_all = load_data(config["work_dir"])
    for spec in args.config:
        subset, method, model_name, table = spec.split(":")
        rows = np.asarray(SUBSETS[subset](data_all))
        data = data_all.loc[rows].reset_index(drop=True)
        X_df = pd.read_csv(os.path.join(config["work_dir"], "features", f"{table}.csv")).loc[rows].reset_index(drop=True)
        X, j, papers = X_df.to_numpy(dtype=float), data["j"].to_numpy(), data["paper"].to_numpy()

        pred, shap_oof = np.zeros(len(j)), np.zeros(X.shape)
        for tr, te in LeaveOneGroupOut().split(X, groups=papers):
            scaler = StandardScaler().fit(X[tr])
            model, shap_method = get_model(model_name)
            pred[te], model, _, keep = fit_predict(method, model, scaler.transform(X[tr]), j[tr], papers[tr], scaler.transform(X[te]))
            shap_oof[te] = shap_values(model, shap_method, scaler.transform(X[tr])[keep], scaler.transform(X[te]))

        preds = data[["number", "name", "paper", "type", "j", "smiles"]].assign(y_pred=pred)
        wpa = within_paper_accuracy(preds)
        # only papers with >= 2 materials have within-paper pairs; single-material papers say nothing about ranking
        multi = preds.groupby("paper")["j"].transform("size").to_numpy() >= 2
        biggest = preds.loc[multi, "paper"].value_counts().index[0]
        in_big = papers == biggest

        abs_shap = np.abs(shap_oof[multi])
        importance = pd.DataFrame(
            {
                "feature": X_df.columns,
                "mean_abs_shap": abs_shap.mean(axis=0),
                "direction": [spearmanr(X[multi, k], shap_oof[multi, k]).statistic if X[multi, k].std() > 0 and shap_oof[multi, k].std() > 0 else np.nan for k in range(X.shape[1])],
                "n_present": (X[multi] > 0).sum(axis=0),
                f"share_from_{biggest}": np.abs(shap_oof[in_big]).sum(axis=0) / np.maximum(np.abs(shap_oof[multi]).sum(axis=0), 1e-12),
                f"mean_abs_shap_without_{biggest}": np.abs(shap_oof[multi & ~in_big]).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)

        ecfp = importance["feature"].str.fullmatch(r"ecfp_\d+")
        if ecfp.any():
            envs = ecfp_environments(data["smiles"].tolist(), "full")
            importance.loc[ecfp, "substructures"] = importance.loc[ecfp, "feature"].map(
                lambda f, envs=envs: " | ".join(f"{s} ({n})" for s, n in sorted(envs.get(int(re.match(r"ecfp_(\d+)", f).group(1)), Counter()).items(), key=lambda kv: (-kv[1], kv[0]))[:3])
            )

        out_dir = os.path.join(WORK_DIR, "electro", spec.replace(":", "__"))
        os.makedirs(out_dir, exist_ok=True)
        preds.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
        pd.concat([preds[["number", "name", "paper"]], pd.DataFrame(shap_oof, columns=X_df.columns)], axis=1).to_csv(os.path.join(out_dir, "shap_oof.csv"), index=False)
        importance.to_csv(os.path.join(out_dir, "importance.csv"), index=False)
        rank_corr = spearmanr(importance["mean_abs_shap"], importance[f"mean_abs_shap_without_{biggest}"]).statistic
        print(f"\n=== {spec}: WPA {wpa:.3f}, {multi.sum()} materials in multi-material papers, largest paper {biggest} ({in_big.sum()})")
        print(f"importance rank correlation with vs without {biggest}: {rank_corr:.2f}")
        with pd.option_context("display.width", 250, "display.max_colwidth", 75):
            print(importance.head(12).round(3).to_string(index=False))
