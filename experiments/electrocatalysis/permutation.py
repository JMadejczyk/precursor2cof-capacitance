"""Within-paper permutation test for the best leave-one-paper-out configurations.

J is shuffled inside each paper and the same correction, model and folds are rerun. If the observed within-paper
accuracy is rarely reached by shuffled data, the model has learned a real within-paper ordering.
Output: results/permutation.csv.
"""

import os
from glob import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from experiments.electrocatalysis.common import SUBSETS, get_model, load_config, load_data, within_paper_accuracy
from experiments.electrocatalysis.corrections import fit_predict


def predict_paper_out(X: pd.DataFrame, j: np.ndarray, papers: np.ndarray, rows: np.ndarray, method: str, model_name: str) -> np.ndarray:
    """Leave-one-paper-out predictions for the given rows (same order as rows)."""
    pred = pd.Series(np.nan, index=rows)
    for tr, te in LeaveOneGroupOut().split(rows, groups=papers[rows]):
        tr, te = rows[tr], rows[te]
        scaler = StandardScaler().fit(X.iloc[tr])
        model, _ = get_model(model_name)
        pred[te] = fit_predict(method, model, scaler.transform(X.iloc[tr]), j[tr], papers[tr], scaler.transform(X.iloc[te]))[0]
    return pred.to_numpy()


if __name__ == "__main__":
    config = load_config()
    work_dir, cfg = config["work_dir"], config["permutation"]
    data = load_data(work_dir)
    j, papers = data["j"].to_numpy(), data["paper"].to_numpy()

    # pick the top configurations by within-paper accuracy, per subset
    scores = []
    for path in glob(os.path.join(work_dir, "runs", "*", "paper_out", "*", "*", "*", "predictions.csv")):
        p = pd.read_csv(path)
        if p["model"].iloc[0] in ("median", "tabpfn"):
            continue
        scores.append({k: p[k].iloc[0] for k in ["subset", "method", "model", "approach", "feature_set"]} | {"wpa": within_paper_accuracy(p)})
    top = pd.DataFrame(scores).sort_values("wpa", ascending=False).groupby("subset").head(cfg["top_k"])

    rng = np.random.default_rng(42)
    results = []
    for _, c in top.iterrows():
        rows = np.asarray(SUBSETS[c["subset"]](data))
        X = pd.read_csv(os.path.join(work_dir, "features", f"{c['approach']}__{c['feature_set']}.csv"))
        frame = data.loc[rows, ["paper", "j"]].copy()
        frame["y_pred"] = predict_paper_out(X, j, papers, rows, c["method"], c["model"])
        observed = within_paper_accuracy(frame)
        null = []
        for _ in range(cfg["n_perm"]):
            j_perm = j.copy()
            for paper in np.unique(papers[rows]):
                idx = rows[papers[rows] == paper]
                j_perm[idx] = rng.permutation(j[idx])
            shuffled = pd.DataFrame({"paper": papers[rows], "j": j_perm[rows], "y_pred": predict_paper_out(X, j_perm, papers, rows, c["method"], c["model"])})
            null.append(within_paper_accuracy(shuffled))
        null = np.array(null)
        results.append(c.drop("wpa").to_dict() | {"observed": observed, "null_mean": null.mean(), "null_95": np.percentile(null, 95), "p_value": (1 + (null >= observed).sum()) / (1 + len(null))})
        print(results[-1])
    pd.DataFrame(results).to_csv(os.path.join(work_dir, "permutation.csv"), index=False)
