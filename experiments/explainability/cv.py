"""Which models predict capacitance better than chance? Repeated CV plus a y-scrambling null for every
target x feature table x model.

For each configuration, written to results/cv/<target>/<table>/<model>/:
- oof.npy        (10, n) out-of-fold predictions, one row per CV repeat
- metrics.csv    RMSE, R2 and Spearman of each repeat (pooled over its 5 folds)
- null.csv       the same metrics with the target shuffled, one row per permutation, averaged over
                 NULL_REPEATS CV repeats whose folds are stratified on the shuffled capacity. The p-value compares
                 it with the real metric averaged over the first NULL_REPEATS repeats, so both go through an
                 identical procedure.

Targets:
- capacity       capacity_max as it is
- size_resid     capacity_max minus a straight-line fit on heavy-atom count: is there anything beyond monomer size?

Run from the repo root:  PYTHONPATH=. .venv/bin/python experiments/explainability/cv.py [--models ...] [--n_jobs 22]
TabPFN runs on the GPU in this process (one configuration at a time); run it separately with --models tabpfn.
"""

import argparse
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from experiments.explainability.common import TABLES, TARGET, WORK_DIR, cv_splits, load_data, load_table, metrics, oof_predictions, size_residual

N_REPEATS = 10
NULL_REPEATS = 2
N_PERM = {"default": 100, "tabpfn": 30}
CPU_MODELS = ["median", "bayes", "lasso", "random_forest", "xgboost", "knn"]

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=CPU_MODELS)
parser.add_argument("--tables", nargs="+", default=list(TABLES))
parser.add_argument("--targets", nargs="+", default=["capacity", "size_resid"])
parser.add_argument("--n_jobs", type=int, default=22)


def targets() -> dict:
    """:return: {target name: array}."""
    data = load_data()
    y = data[TARGET].to_numpy(dtype=float)
    return {"capacity": y, "size_resid": size_residual(y, data["heavy_atoms"].to_numpy(dtype=float))}


def run_real(model: str, X: np.ndarray, y: np.ndarray, repeats: list) -> tuple:
    """:return: (oof predictions, per-repeat metrics dataframe)."""
    oof = oof_predictions(model, X, y, repeats)
    return oof, pd.DataFrame([metrics(y, p) for p in oof])


def run_null(model: str, X: np.ndarray, y: np.ndarray, strat: np.ndarray, perm_seeds: list) -> list:
    """
    Shuffle the rows of the target, and stratify the folds on the shuffled capacity, exactly as the real run
    stratifies on the real capacity (the identity permutation reproduces the real run's first NULL_REPEATS repeats).
    :param strat: capacity, used for stratification.
    :return: list of metric dicts, one per permutation (averaged over the null repeats).
    """
    rows = []
    for seed in perm_seeds:
        perm = np.random.default_rng(seed).permutation(len(y))
        y_perm = y[perm]
        oof = oof_predictions(model, X, y_perm, cv_splits(strat[perm], n_repeats=NULL_REPEATS))
        rows.append(pd.DataFrame([metrics(y_perm, p) for p in oof]).mean().to_dict() | {"seed": seed})
    return rows


if __name__ == "__main__":
    args = parser.parse_args()
    ys = targets()
    configs = []
    for target in args.targets:
        y = ys[target]
        # splits are stratified on the real capacity for both targets, so every target and model sees the same folds
        strat = ys["capacity"]
        repeats = cv_splits(strat, n_repeats=N_REPEATS)
        for table in args.tables:
            X = load_table(table).to_numpy(dtype=float)
            for model in args.models:
                out_dir = os.path.join(WORK_DIR, "cv", target, table, model)
                if os.path.exists(os.path.join(out_dir, "null.csv")):
                    continue
                configs.append((out_dir, model, X, y, repeats, strat))
    print(f"{len(configs)} configurations to run")

    if "tabpfn" in args.models:
        # GPU: sequential, one configuration at a time
        for out_dir, model, X, y, repeats, strat in configs:
            os.makedirs(out_dir, exist_ok=True)
            oof, real = run_real(model, X, y, repeats)
            null = run_null(model, X, y, strat, list(range(N_PERM["tabpfn"])))
            np.save(os.path.join(out_dir, "oof.npy"), oof)
            real.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
            pd.DataFrame(null).to_csv(os.path.join(out_dir, "null.csv"), index=False)
            print(out_dir, real.mean().round(3).to_dict(), flush=True)
        raise SystemExit

    # CPU: the real run and chunks of 10 permutations are independent jobs
    chunks = [list(range(s, s + 10)) for s in range(0, N_PERM["default"], 10)]
    jobs = []
    for c, (out_dir, model, X, y, repeats, strat) in enumerate(configs):
        jobs.append(delayed(lambda *a: ("real", a[0], run_real(*a[1:])))(c, model, X, y, repeats))
        for chunk in chunks:
            jobs.append(delayed(lambda *a: ("null", a[0], run_null(*a[1:])))(c, model, X, y, strat, chunk))
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(jobs)

    for c, (out_dir, model, X, y, repeats, strat) in enumerate(configs):
        os.makedirs(out_dir, exist_ok=True)
        null = []
        for kind, idx, res in results:
            if idx != c:
                continue
            if kind == "real":
                np.save(os.path.join(out_dir, "oof.npy"), res[0])
                res[1].to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
            else:
                null.extend(res)
        pd.DataFrame(null).sort_values("seed").to_csv(os.path.join(out_dir, "null.csv"), index=False)
    print("done")
