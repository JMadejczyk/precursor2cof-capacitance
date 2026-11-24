"""Shared helpers for the capacitance explainability experiment: data, feature tables, readable names, models, CV."""

import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.EState.AtomTypes import _rawD
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LassoCV, LinearRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DATA_CSV = "cof_matched/data_substrate_cofs.csv"
FEATURES_DIR = "cof_matched/rth_0.99/features"
WORK_DIR = "experiments/explainability/results"
TARGET = "capacity_max"

# feature tables from the main pipeline (rth 0.99); combinations are joined column-wise (all files have the same row order)
BASE_TABLES = {
    "descriptor": "features_rth_0.99_descriptor_False_False.csv",
    "custom_pattern": "features_rth_0.99_custom_pattern_True.csv",
    "estate": "features_rth_0.99_estate_sum.csv",
    "functional_groups": "features_rth_0.99_functional_groups_True.csv",
    "maccs": "features_rth_0.99_maccs_True.csv",
    "ecfp": "features_rth_0.99_ecfp_2_1024_True.csv",
}
TABLES = {
    "size": ["size"],  # heavy-atom count only: the "how big is the monomer" baseline
    "descriptor": ["descriptor"],
    "custom_pattern": ["custom_pattern"],
    "estate": ["estate"],
    "functional_groups": ["functional_groups"],
    "maccs": ["maccs"],
    "ecfp": ["ecfp"],
    "rf_best": ["estate", "functional_groups"],  # best RF set from greedy selection
    "brr_best": ["descriptor", "ecfp", "estate", "functional_groups"],  # best BRR set from greedy selection
}

MODEL_NAMES = {"median": "Median", "bayes": "BRR", "lasso": "Lasso", "random_forest": "RF", "xgboost": "XGBoost", "knn": "KNN", "tabpfn": "TabPFN"}

CUSTOM_PATTERNS = ["c2ccc1ccccc1c2", "c2ccc(c1ccccc1)cc2", "c1ccncc1", "c1ncncn1", "N#Cc1ccccc1", "C/C=C/C=C/C", "C=CC(=O)C=C"]
FR_NAMES = [n for n, _ in Descriptors.descList if n.startswith("fr_")]  # same order as skfp FunctionalGroupsFingerprint (checked)
ESTATE_NAMES = [n for n, _ in _rawD]


def load_data() -> pd.DataFrame:
    """
    :return: smiles, capacity_max, heavy_atoms, is_ctf (has a nitrile), one row per substrate in file order.
    """
    data = pd.read_csv(DATA_CSV)
    mols = [Chem.MolFromSmiles(s) for s in data["smiles"]]
    data["heavy_atoms"] = [m.GetNumHeavyAtoms() for m in mols]
    data["is_ctf"] = [m.HasSubstructMatch(Chem.MolFromSmarts("C#N")) for m in mols]
    return data


def load_table(name: str) -> pd.DataFrame:
    """
    :param name: key of TABLES.
    :return: unscaled feature dataframe, rows in the order of load_data().
    """
    data = load_data()
    parts = []
    for part in TABLES[name]:
        if part == "size":
            parts.append(data[["heavy_atoms"]])
            continue
        df = pd.read_csv(os.path.join(FEATURES_DIR, BASE_TABLES[part]))
        assert (df["smiles"].to_numpy() == data["smiles"].to_numpy()).all(), f"row order of {part} differs from {DATA_CSV}"
        parts.append(df.drop(columns=["smiles", TARGET]))
    return pd.concat(parts, axis=1)


def readable(feature: str) -> str:
    """Human-readable name of a feature column (ECFP bits are decoded separately into substructures)."""
    if feature.startswith("estate_"):
        return f"EState {ESTATE_NAMES[int(feature.split('_')[1])]}"
    if feature.startswith("functional_groups_"):
        return FR_NAMES[int(feature.split("_")[-1])]
    if feature.startswith("custom_pattern_"):
        return f"pattern {CUSTOM_PATTERNS[int(feature.split('_')[-1])]}"
    return feature


def family(feature: str) -> str:
    """Feature family, for importance shares."""
    for prefix in ["ecfp", "estate", "functional_groups", "maccs", "custom_pattern"]:
        if feature.startswith(prefix):
            return prefix
    return "descriptor"


def get_model(name: str, n_jobs: int = 1, device: str = "cuda") -> object | None:
    """
    Fresh, unfitted model with fixed hyperparameters (no tuning: ~90 training rows per fold, and the
    permutation null must use exactly the same procedure as the real run).
    :return: model, or None for the median baseline.
    """
    if name == "median":
        return None
    if name == "tabpfn":
        from tabpfn import TabPFNRegressor  # pylint: disable=import-outside-toplevel  # slow import, only when used

        return TabPFNRegressor(device=device, random_state=42, ignore_pretraining_limits=True)
    models = {
        "bayes": BayesianRidge(),
        "lasso": LassoCV(cv=5, n_alphas=30, max_iter=20000, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.33, random_state=42, n_jobs=n_jobs),
        "xgboost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.5, min_child_weight=3, random_state=42, n_jobs=n_jobs),
        "knn": KNeighborsRegressor(n_neighbors=5),
        "linear": LinearRegression(),
    }
    return clone(models[name])


def cv_splits(y: np.ndarray, n_splits: int = 5, n_repeats: int = 10, n_bins: int = 5, random_state: int = 42) -> list:
    """
    Repeated stratified K-fold on the binned target. Each repeat predicts every row exactly once.
    :return: list of repeats, each a list of (train_idx, test_idx).
    """
    bins = pd.qcut(y, n_bins, labels=False, duplicates="drop")
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    folds = list(splitter.split(np.zeros(len(y)), bins))
    return [folds[r * n_splits : (r + 1) * n_splits] for r in range(n_repeats)]


def fit_predict(model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, device: str = "cuda") -> tuple:
    """
    Scale on the training fold, fit, predict.
    :return: (predictions, fitted model or None, fitted scaler).
    """
    scaler = StandardScaler().fit(X_train)
    if model_name == "median":
        return np.full(len(X_test), np.median(y_train)), None, scaler
    model = get_model(model_name, device=device)
    model.fit(scaler.transform(X_train), y_train)
    return np.asarray(model.predict(scaler.transform(X_test))).ravel(), model, scaler


def oof_predictions(model_name: str, X: np.ndarray, y: np.ndarray, repeats: list, device: str = "cuda") -> np.ndarray:
    """
    :param repeats: output of cv_splits (or a slice of it).
    :return: (n_repeats, n_rows) out-of-fold predictions.
    """
    out = np.zeros((len(repeats), len(y)))
    for r, folds in enumerate(repeats):
        for train_idx, test_idx in folds:
            out[r, test_idx], _, _ = fit_predict(model_name, X[train_idx], y[train_idx], X[test_idx], device=device)
    return out


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """Pooled out-of-fold metrics of one repeat. R2 is against the overall mean (a median model scores slightly below 0)."""
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    r2 = float(1 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2))
    rho = float(spearmanr(y, pred).statistic) if np.std(pred) > 0 else 0.0
    return {"rmse": rmse, "r2": r2, "spearman": rho}


def size_residual(y: np.ndarray, heavy_atoms: np.ndarray) -> np.ndarray:
    """
    Capacitance minus a straight-line fit on heavy-atom count (fitted on all rows: 2 parameters, so the
    leak into CV is negligible). A model that predicts this has learned something beyond monomer size.
    """
    fit = LinearRegression().fit(heavy_atoms.reshape(-1, 1), y)
    return y - fit.predict(heavy_atoms.reshape(-1, 1))
