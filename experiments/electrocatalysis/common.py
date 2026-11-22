"""Shared helpers for the electrocatalysis experiment: config, data subsets, models, metrics."""

import os

import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.base import clone
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

from modules.training_and_evaluation.model_factory import Models

CONFIG_PATH = "experiments/electrocatalysis/config.yaml"

# row subsets of the labelled data: all 55 materials, or without the 6 suspect rows flagged by clean.py
SUBSETS = {"all": lambda d: d.index, "clean": lambda d: d.index[d["suspect"] == ""]}

MODEL_NAMES = {"median": "Median", "bayes": "BRR", "random_forest": "RF", "xgboost": "XGBoost", "knn": "KNN", "tabpfn": "TabPFN"}


def load_config(path: str = CONFIG_PATH) -> dict:
    """
    :param path: path to the experiment config.
    :return: config dict.
    """
    with open(path, "r", encoding="UTF-8") as f:
        return yaml.safe_load(f)


def load_data(work_dir: str) -> pd.DataFrame:
    """
    Labelled materials as written by features.py (one row per material, same order as the feature tables).
    :param work_dir: results directory.
    :return: dataframe with number, name, paper, type, j, suspect, smiles, has_side_chain.
    """
    data = pd.read_csv(os.path.join(work_dir, "data.csv"))
    data["suspect"] = data["suspect"].fillna("")
    return data


def get_model(name: str, tabpfn_device: str = "cpu") -> tuple:
    """
    Fresh, unfitted model.
    :param name: model name from modules.training_and_evaluation.model_factory, or 'median' / 'tabpfn'.
    :param tabpfn_device: device for TabPFN.
    :return: (model or None for 'median', SHAP method or None).
    """
    if name == "median":
        return None, None
    if name == "tabpfn":
        from tabpfn import TabPFNRegressor  # slow import, only when used

        return TabPFNRegressor(device=tabpfn_device, random_state=42, ignore_pretraining_limits=True), None
    _, model, _, shap_method = Models().get_model(name)
    return clone(model), shap_method


def shap_values(model: object, method: str | None, X_background: np.ndarray, X_explain: np.ndarray) -> np.ndarray | None:
    """
    SHAP values with the explainer the model factory assigns to the model ('tree' or 'linear').
    :return: array (n_explain, n_features) or None for models without a cheap explainer.
    """
    if method == "tree":
        return shap.TreeExplainer(model).shap_values(X_explain)
    if method == "linear":
        return shap.LinearExplainer(model, X_background).shap_values(X_explain)
    return None


def variance_filter(df_features: pd.DataFrame, remove_threshold: float) -> pd.DataFrame:
    """
    Remove low-variance features, same rule as modules.features.feature_factory.generate_features.
    """
    scaled = MinMaxScaler().fit_transform(df_features)
    selector = VarianceThreshold(threshold=1 - remove_threshold).fit(scaled)
    return df_features.iloc[:, selector.get_support(indices=True)]


def paper_pairs(df: pd.DataFrame, true_col: str = "j", pred_col: str = "y_pred") -> pd.DataFrame:
    """
    All pairs of materials from the same paper with whether the prediction orders them correctly.
    Ties in the truth are skipped; ties in the prediction count 0.5.
    :return: dataframe with paper and score per pair.
    """
    rows = []
    for paper, g in df.groupby("paper"):
        v = g[[true_col, pred_col]].to_numpy()
        for a in range(len(v)):
            for b in range(a + 1, len(v)):
                dt, dp = v[a][0] - v[b][0], v[a][1] - v[b][1]
                if dt != 0:
                    rows.append({"paper": paper, "score": 0.5 if dp == 0 else float(np.sign(dt) == np.sign(dp))})
    return pd.DataFrame(rows, columns=["paper", "score"])


def within_paper_accuracy(df: pd.DataFrame, true_col: str = "j", pred_col: str = "y_pred") -> float:
    """
    Within-paper pairwise accuracy (WPA): share of same-paper pairs ordered correctly. 0.5 = chance.
    """
    pairs = paper_pairs(df, true_col, pred_col)
    return pairs["score"].mean()


def metrics(df: pd.DataFrame, rng: np.random.Generator, n_boot: int = 1000) -> pd.Series:
    """
    Metrics of pooled out-of-fold predictions.
    r2_abs: R2 of J. r2_within: R2 after subtracting each paper's mean from truth and prediction (papers with >= 2 materials).
    wpa: within-paper pairwise accuracy with a 95% CI from a bootstrap over papers.
    """
    err = df["y_pred"] - df["j"]
    r2_abs = 1 - (err**2).sum() / ((df["j"] - df["j"].mean()) ** 2).sum()
    multi = df[df.groupby("paper")["j"].transform("size") >= 2]
    t = multi["j"] - multi.groupby("paper")["j"].transform("mean")
    p = multi["y_pred"] - multi.groupby("paper")["y_pred"].transform("mean")
    r2_within = 1 - ((p - t) ** 2).sum() / (t**2).sum()
    by_paper = paper_pairs(df).groupby("paper")["score"].agg(["sum", "count"])
    idx = rng.integers(0, len(by_paper), size=(n_boot, len(by_paper)))
    boots = by_paper["sum"].to_numpy()[idx].sum(1) / by_paper["count"].to_numpy()[idx].sum(1)
    return pd.Series({
        "r2_abs": r2_abs,
        "rmse": np.sqrt((err**2).mean()),
        "r2_within": r2_within,
        "wpa": by_paper["sum"].sum() / by_paper["count"].sum(),
        "wpa_lo": np.percentile(boots, 2.5),
        "wpa_hi": np.percentile(boots, 97.5),
        "pairs": int(by_paper["count"].sum()),
    })
