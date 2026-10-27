"""Script to compare cross-validation results of models trained on different feature sets.

Reads <results_dir>/<model>/<dataset>/scores.pickle written by the training pipelines and
prints mean +- std per metric. With two results dirs (e.g. baseline and split features),
it also compares the best feature set of each per model on the same folds.

Usage: python scripts/compare_results.py --results_dirs results/baseline_cof_matched_rth_0.99 results/split_cof_matched_rth_0.99
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--results_dirs", type=str, nargs="+", required=True, help="Results directories to compare (first one is the reference)")
parser.add_argument("--metrics", type=str, nargs="+", default=["mae", "rmse", "r2_score", "smape", "pairwise_accuracy_score"], help="Metrics to report")
parser.add_argument("--rank_metric", type=str, default="mae", help="Metric used to pick the best feature set per model (lower is better unless it is r2_score, pairwise_accuracy_score or ndcg_score)")
parser.add_argument("--output", type=str, default="", help="Optional path to save the full table as CSV")

HIGHER_IS_BETTER = {"r2_score", "pairwise_accuracy_score", "ndcg_score"}


def load_scores(results_dirs: list[str]) -> dict:
    """
    Load per-fold scores for every results dir, model and dataset.
    :param results_dirs: list of results directories.
    :return: dict (results_dir, model, dataset) -> dict metric -> list of fold scores.
    """
    scores = {}
    for results_dir in results_dirs:
        for model in sorted(os.listdir(results_dir)):
            model_dir = os.path.join(results_dir, model)
            if not os.path.isdir(model_dir):
                continue
            for dataset in sorted(os.listdir(model_dir)):
                scores_path = os.path.join(model_dir, dataset, "scores.pickle")
                if os.path.exists(scores_path):
                    scores[(results_dir, model, dataset)] = joblib.load(scores_path)
    return scores


def summary_table(scores: dict, metrics: list[str]) -> pd.DataFrame:
    """
    Mean and std of every metric, plus the median-prediction baseline.
    :param scores: output of load_scores.
    :param metrics: metrics to report.
    :return: dataframe with one row per (results_dir, model, dataset).
    """
    rows = []
    for (results_dir, model, dataset), s in scores.items():
        row = {"results": results_dir, "model": model, "dataset": dataset.replace("features_", ""), "dataset_dir": dataset}
        for metric in metrics:
            row[f"{metric}_mean"] = np.mean(s[metric])
            row[f"{metric}_std"] = np.std(s[metric])
            row[f"baseline_{metric}"] = np.mean(s[f"baseline_{metric}"])
        rows.append(row)
    return pd.DataFrame(rows)


def best_per_model(table: pd.DataFrame, rank_metric: str) -> pd.DataFrame:
    """
    Best feature set per results dir and model.
    :param table: output of summary_table.
    :param rank_metric: metric used for ranking.
    :return: dataframe with the best row per (results, model).
    """
    ascending = rank_metric not in HIGHER_IS_BETTER
    ranked = table.sort_values(f"{rank_metric}_mean", ascending=ascending)
    return ranked.groupby(["results", "model"], sort=False).head(1).sort_values(["model", "results"])


def paired_comparison(scores: dict, best: pd.DataFrame, results_dirs: list[str], rank_metric: str) -> pd.DataFrame:
    """
    Per model, compare the best feature set of the second results dir against the first on the same folds.
    :param scores: output of load_scores.
    :param best: output of best_per_model.
    :param results_dirs: the two results directories.
    :param rank_metric: metric compared.
    :return: dataframe with mean fold difference and number of folds won.
    """
    sign = -1 if rank_metric in HIGHER_IS_BETTER else 1
    rows = []
    for model in sorted(best["model"].unique()):
        picks = best[best["model"] == model].set_index("results")
        if not all(d in picks.index for d in results_dirs):
            continue
        ref, new = (scores[(d, model, picks.loc[d, "dataset_dir"])][rank_metric] for d in results_dirs)
        diff = np.array(new) - np.array(ref)
        rows.append(
            {
                "model": model,
                "reference_dataset": picks.loc[results_dirs[0], "dataset"],
                "compared_dataset": picks.loc[results_dirs[1], "dataset"],
                f"{rank_metric}_diff_mean": diff.mean(),
                f"{rank_metric}_diff_std": diff.std(),
                "folds_won": int((sign * diff < 0).sum()),
                "folds": len(diff),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    args = parser.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 80)

    all_scores = load_scores(args.results_dirs)
    table = summary_table(all_scores, args.metrics)
    if table.empty:
        raise SystemExit("No scores.pickle files found.")
    if args.output:
        table.to_csv(args.output, index=False)
        print(f"Full table saved to {args.output}")

    best = best_per_model(table, args.rank_metric)
    columns = ["results", "model", "dataset"] + [c for m in args.metrics for c in (f"{m}_mean", f"{m}_std")] + [f"baseline_{args.rank_metric}"]
    print(f"Best feature set per model (by {args.rank_metric}):")
    print(best[columns].round(3).to_string(index=False))

    if len(args.results_dirs) == 2:
        print(f"\nPaired comparison on the same folds ({args.rank_metric}, second minus first):")
        print(paired_comparison(all_scores, best, args.results_dirs, args.rank_metric).round(3).to_string(index=False))
