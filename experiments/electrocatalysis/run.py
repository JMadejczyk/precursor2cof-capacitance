"""Run every model on every feature table and correction method; store out-of-fold predictions and SHAP importance.

Output per run: results/runs/<subset>/<cv>/<method>/<model>/<table>/predictions.csv (+ importance.csv for SHAP models).
cv = 'paper_out' (leave one paper out, every method) or 'random' (repeated stratified 5-fold, raw J only).
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from experiments.electrocatalysis.common import SUBSETS, get_model, load_config, load_data, shap_values
from experiments.electrocatalysis.corrections import fit_predict
from modules.data.utils import custom_discretization

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=str, nargs="*", default=None, help="Override the models from the config")
parser.add_argument("--methods", type=str, nargs="*", default=None, help="Override the correction methods from the config")


def make_folds(data: pd.DataFrame, rows: np.ndarray, cv: str, cfg: dict) -> list:
    """
    :return: list of (repeat, train_rows, test_rows), indices are positions in `data`.
    """
    if cv == "paper_out":
        return [(0, rows[tr], rows[te]) for tr, te in LeaveOneGroupOut().split(rows, groups=data.loc[rows, "paper"])]
    binned = custom_discretization(data.loc[rows, ["j"]].reset_index(drop=True), cfg["num_bins"])
    rskf = RepeatedStratifiedKFold(n_splits=cfg["num_folds"], n_repeats=cfg["num_repeats"], random_state=cfg["random_state"])
    return [(i // cfg["num_folds"], rows[tr], rows[te]) for i, (tr, te) in enumerate(rskf.split(np.zeros(len(rows)), binned))]


if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config()
    work_dir = config["work_dir"]
    models = ["median"] + (args.models or config["models"] + (["tabpfn"] if config["tabpfn"]["enabled"] else []))
    methods = args.methods or config["methods"]
    data = load_data(work_dir)
    j, papers = data["j"].to_numpy(), data["paper"].to_numpy()
    runs = [("random", "none")] + [("paper_out", m) for m in methods]

    for table_file in sorted(os.listdir(os.path.join(work_dir, "features"))):
        table = table_file.replace(".csv", "")
        approach, feature_set = table.split("__")
        X_all = pd.read_csv(os.path.join(work_dir, "features", table_file))
        for subset, select in SUBSETS.items():
            rows = np.asarray(select(data))
            for cv, method in runs:
                folds = make_folds(data, rows, cv, config["random_cv"])
                for model_name in models:
                    if model_name == "median" and method != "none":
                        continue
                    run_dir = os.path.join(work_dir, "runs", subset, cv, method, model_name, table)
                    if os.path.exists(os.path.join(run_dir, "predictions.csv")):
                        continue
                    os.makedirs(run_dir, exist_ok=True)
                    out, importance, n_explained = [], np.zeros(X_all.shape[1]), 0
                    for fold, (repeat, tr, te) in enumerate(folds):
                        if model_name == "median":
                            pred = np.full(len(te), np.median(j[tr]))
                        else:
                            scaler = StandardScaler().fit(X_all.iloc[tr])
                            X_tr, X_te = scaler.transform(X_all.iloc[tr]), scaler.transform(X_all.iloc[te])
                            model, shap_method = get_model(model_name, config["tabpfn"]["device"])
                            pred, model, _, keep = fit_predict(method, model, X_tr, j[tr], papers[tr], X_te)
                            if model_name in config["shap_models"]:
                                values = shap_values(model, shap_method, X_tr[keep], X_te)
                                if values is not None:
                                    importance += np.abs(values).sum(axis=0)
                                    n_explained += len(te)
                        out.extend({"fold": fold, "repeat": repeat, "row": r, "y_pred": p} for r, p in zip(te, pred))
                    out = pd.DataFrame(out).merge(data[["number", "paper", "type", "has_side_chain", "suspect", "j"]], left_on="row", right_index=True)
                    out.assign(subset=subset, cv=cv, method=method, model=model_name, approach=approach, feature_set=feature_set).to_csv(os.path.join(run_dir, "predictions.csv"), index=False)
                    if n_explained:
                        pd.DataFrame({"feature": X_all.columns, "mean_abs_shap": importance / n_explained}).to_csv(os.path.join(run_dir, "importance.csv"), index=False)
                    print(f"{subset:5s} {cv:9s} {method:7s} {model_name:13s} {table}")
