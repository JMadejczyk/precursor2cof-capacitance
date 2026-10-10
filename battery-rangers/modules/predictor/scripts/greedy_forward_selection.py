"""Greedy forward feature-set selection based on SMAPE and pairwise accuracy ranking."""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
import yaml

from modules.predictor.data.utils import custom_data_kfold
from modules.predictor.training_and_evaluation.chemprop_pipeline import ChempropTrainingPipeline
from modules.predictor.training_and_evaluation.sklearn_pipeline import SklearnTrainingPipeline
from modules.predictor.training_and_evaluation.tabpfn_pipeline import TabPFNTrainingPipeline

# pylint: disable=import-error


TARGET_COLUMN = "capacity_max"
DEFAULT_ID_COLUMN = "smiles"


@dataclass
class CandidateEvaluation:
    """Container for one candidate evaluation result."""

    model: str
    candidate: tuple[str, ...]
    smape: float
    pa: float
    results: dict[str, float]

    def to_record(
        self,
        step: int,
        selected_so_far: list[str],
        combined_rank: float | None = None,
    ) -> dict[str, Any]:
        """Serialize the evaluation into a JSON-safe record."""
        record = {
            "step": step,
            "model": self.model,
            "candidate": list(self.candidate),
            "selected_so_far": selected_so_far,
            "smape": self.smape,
            "pa": self.pa,
            "results": self.results,
        }
        if combined_rank is not None:
            record["combined_rank"] = combined_rank
        return record


def _resolve_feature_set_name(feature_set_name: str, available_feature_files: set[str]) -> str:
    """Resolve a feature set to the existing csv stem."""
    stem = feature_set_name.replace(".csv", "")
    if stem not in available_feature_files:
        raise ValueError(f"Feature set '{feature_set_name}' was not found in the features directory.")
    return stem


def _load_feature_sets(datasets_dir: str, feature_sets: list[str]) -> dict[str, tuple[pd.DataFrame, dict[str, list[str]]]]:
    """Load requested feature sets and their feature type definitions."""
    feature_dir = os.path.join(datasets_dir, "features")
    types_dir = os.path.join(datasets_dir, "feature_types")

    available_files = {file_name.replace(".csv", "") for file_name in os.listdir(feature_dir)}
    loaded: dict[str, tuple[pd.DataFrame, dict[str, list[str]]]] = {}

    for feature_set_name in feature_sets:
        resolved_name = _resolve_feature_set_name(feature_set_name, available_files)
        dataset = pd.read_csv(os.path.join(feature_dir, f"{resolved_name}.csv"))
        type_file = resolved_name.replace("features", "feature_types") + ".yaml"
        with open(os.path.join(types_dir, type_file), "r", encoding="UTF-8") as file:
            feature_types = yaml.safe_load(file)
        loaded[resolved_name] = (dataset, feature_types)

    return loaded


def _merge_feature_types(feature_types_list: list[dict[str, list[str]]], selected_columns: list[str]) -> dict[str, list[str]]:
    """Merge feature types for all selected feature sets while keeping column order stable."""
    merged = {"numerical": [], "categorical": [], "binary": []}
    selected_set = set(selected_columns)

    for f_types in feature_types_list:
        for key, _ in merged.items():
            for column in f_types.get(key, []):
                if column in selected_set and column not in merged[key]:
                    merged[key].append(column)

    return merged


def _build_combined_dataset(
    loaded_feature_sets: dict[str, tuple[pd.DataFrame, dict[str, list[str]]]],
    selected_feature_sets: list[str],
    target_column: str = TARGET_COLUMN,
    id_column: str = DEFAULT_ID_COLUMN,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build a combined dataset from a list of feature-set names."""
    if len(selected_feature_sets) == 0:
        raise ValueError("selected_feature_sets cannot be empty.")

    first_df, first_types = loaded_feature_sets[selected_feature_sets[0]]

    if target_column not in first_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset '{selected_feature_sets[0]}'.")

    keep_columns = [target_column]
    if id_column in first_df.columns:
        keep_columns.insert(0, id_column)

    combined = first_df[keep_columns].copy()
    selected_columns: list[str] = []
    feature_types_to_merge: list[dict[str, list[str]]] = [first_types]

    for feature_set_name in selected_feature_sets:
        dataset, f_types = loaded_feature_sets[feature_set_name]
        feature_types_to_merge.append(f_types)
        candidate_columns = [c for c in dataset.columns if c not in {target_column, id_column}]
        new_columns = [c for c in candidate_columns if c not in selected_columns]

        if len(new_columns) == 0:
            continue

        if id_column in combined.columns and id_column in dataset.columns:
            indexed = dataset.set_index(id_column)
            aligned = indexed.reindex(combined[id_column])
            combined[new_columns] = aligned[new_columns].to_numpy()
        else:
            if len(dataset) != len(combined):
                raise ValueError(f"Cannot align feature set '{feature_set_name}' without id column '{id_column}'.")
            combined[new_columns] = dataset.reset_index(drop=True)[new_columns]

        selected_columns.extend(new_columns)

    merged_feature_types = _merge_feature_types(feature_types_to_merge, selected_columns)
    return combined, merged_feature_types


def _build_pipeline(
    model_name: str,
    X: pd.DataFrame,  # pylint: disable=invalid-name
    y: pd.DataFrame,
    feature_types: dict[str, list[str]],
    folds: list,
    metrics: list[str],
    save_dir: str,
    hyperparam_opt: bool,
    num_bins_hyperopt: int,
    verbose: bool,
    chemprop_model_path: str | None = None,
):
    """Create model-specific training pipeline."""
    if model_name == "tabpfn":
        return TabPFNTrainingPipeline(
            X=copy.deepcopy(X),
            y=copy.deepcopy(y),
            feature_types=feature_types,
            folds=copy.deepcopy(folds),
            metrics=metrics,
            save_dir=save_dir,
            data_name="greedy_combined",
            hyperparam_opt=hyperparam_opt,
            num_bins=num_bins_hyperopt,
            verbose=verbose,
        )

    if model_name == "chemprop":
        if not chemprop_model_path:
            raise ValueError("Please provide 'chemprop_params.chemprop_model_path' in config for chemprop.")
        return ChempropTrainingPipeline(
            X=copy.deepcopy(X),
            y=copy.deepcopy(y),
            feature_types=feature_types,
            folds=copy.deepcopy(folds),
            metrics=metrics,
            save_dir=save_dir,
            data_name="greedy_combined",
            model_path=chemprop_model_path,
            hyperparam_opt=hyperparam_opt,
            num_bins=num_bins_hyperopt,
            verbose=verbose,
        )

    return SklearnTrainingPipeline(
        X=copy.deepcopy(X),
        y=copy.deepcopy(y),
        feature_types=feature_types,
        folds=copy.deepcopy(folds),
        metrics=metrics,
        save_dir=save_dir,
        data_name="greedy_combined",
        hyperparam_opt=hyperparam_opt,
        num_bins=num_bins_hyperopt,
        verbose=verbose,
    )


def _candidate_key(candidate: list[str]) -> tuple[str, ...]:
    """Canonical key for candidate caching."""
    return tuple(sorted(candidate))


def _evaluate_candidate(
    model_name: str,
    candidate: list[str],
    loaded_feature_sets: dict[str, tuple[pd.DataFrame, dict[str, list[str]]]],
    evaluation_cache: dict[tuple[str, tuple[str, ...]], CandidateEvaluation],
    config: dict[str, Any],
) -> CandidateEvaluation:
    """Train and evaluate one model on one candidate feature-set combination."""
    cache_key = (model_name, _candidate_key(candidate))
    if cache_key in evaluation_cache:
        return evaluation_cache[cache_key]

    combined_dataset, feature_types = _build_combined_dataset(loaded_feature_sets, candidate)
    if model_name == "chemprop":
        if DEFAULT_ID_COLUMN not in combined_dataset.columns:
            raise ValueError(f"Model '{model_name}' requires '{DEFAULT_ID_COLUMN}' column in combined feature sets.")
        X = combined_dataset.drop(columns=[TARGET_COLUMN], errors="ignore")  # pylint: disable=invalid-name
    else:
        X = combined_dataset.drop(columns=[TARGET_COLUMN, DEFAULT_ID_COLUMN], errors="ignore")  # pylint: disable=invalid-name
    y = combined_dataset[[TARGET_COLUMN]]

    data_config = config.get("dataset_params", {})
    num_folds = data_config.get("num_folds", 5)
    num_bins = data_config.get("num_bins", 5)
    num_bins_hyperopt = data_config.get("num_bins_hyperparam_opt", 5)

    folds = custom_data_kfold(X, y, num_splits=num_folds, num_bins=num_bins, random_state=42)

    candidate_name = "__".join(candidate)
    save_dir = os.path.join(config["save_dir"], "greedy_candidates", model_name, candidate_name)
    os.makedirs(save_dir, exist_ok=True)

    pipeline = _build_pipeline(
        model_name=model_name,
        X=X,
        y=y,
        feature_types=feature_types,
        folds=folds,
        metrics=config["metrics"],
        save_dir=save_dir,
        hyperparam_opt=config.get("hyperparam_opt", True),
        num_bins_hyperopt=num_bins_hyperopt,
        verbose=config.get("verbose", False),
        chemprop_model_path=config.get("chemprop_params", {}).get("chemprop_model_path"),
    )

    results, _, _, _ = pipeline.train_pipeline(model_name=model_name)

    smape_metric = config.get("ranking_metrics", {}).get("smape", "smape")
    pa_metric = config.get("ranking_metrics", {}).get("pa", "pairwise_accuracy_score")

    if smape_metric not in results or pa_metric not in results:
        raise ValueError(f"Ranking metrics not found in results. Required: '{smape_metric}' and '{pa_metric}'.")

    evaluation = CandidateEvaluation(
        model=model_name,
        candidate=tuple(candidate),
        smape=float(results[smape_metric]),
        pa=float(results[pa_metric]),
        results=results,
    )
    evaluation_cache[cache_key] = evaluation
    return evaluation


def _rank_model_candidates(model_name: str, evaluations: list[CandidateEvaluation]) -> pd.DataFrame:
    """Rank candidates for one model by SMAPE and pairwise accuracy."""
    table = pd.DataFrame(
        {
            "model": [model_name] * len(evaluations),
            "candidate": ["__".join(ev.candidate) for ev in evaluations],
            "smape": [ev.smape for ev in evaluations],
            "pa": [ev.pa for ev in evaluations],
        }
    )
    table["smape_rank"] = table["smape"].rank(method="average", ascending=True)
    table["pa_rank"] = table["pa"].rank(method="average", ascending=False)
    table["combined_rank"] = (table["smape_rank"] + table["pa_rank"]) / 2.0
    # Tie-break equal combined rank by better PA rank (lower is better), then SMAPE rank.
    return table.sort_values(
        ["combined_rank", "pa_rank", "smape_rank", "candidate"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def _append_evaluation_record(log_path: str, record: dict[str, Any]) -> None:
    """Append one evaluation record to a JSONL log file."""
    with open(log_path, "a", encoding="UTF-8") as file:
        file.write(json.dumps(record) + "\n")


def greedy_forward_feature_set_selection(  # pylint: disable=too-many-statements
    feature_sets: list[str],
    models: list[str],
    config: dict[str, Any],
    selection_mode: str = "per_model",
) -> dict[str, Any]:
    """Run greedy forward selection for feature-set combinations.

    selection_mode:
      - per_model: independent greedy path for each model
      - global: one shared greedy path chosen by mean combined rank across models
    """
    if selection_mode not in {"per_model", "global"}:
        raise ValueError("selection_mode must be one of: 'per_model', 'global'.")

    loaded_feature_sets = _load_feature_sets(config["datasets_dir"], feature_sets)
    resolved_feature_sets = list(loaded_feature_sets.keys())
    evaluation_cache: dict[tuple[str, tuple[str, ...]], CandidateEvaluation] = {}

    os.makedirs(config["save_dir"], exist_ok=True)

    final_output: dict[str, Any] = {
        "selection_mode": selection_mode,
        "models": models,
        "feature_sets": resolved_feature_sets,
        "iterations": [],
        "selected": {},
    }
    evaluation_records: list[dict[str, Any]] = []
    evaluations_jsonl_path = os.path.join(config["save_dir"], "greedy_forward_selection_evaluations.jsonl")
    with open(evaluations_jsonl_path, "w", encoding="UTF-8"):
        pass

    if selection_mode == "per_model":
        for model_name in models:
            selected: list[str] = []
            remaining = resolved_feature_sets.copy()
            step = 0

            while len(remaining) > 0:
                candidate_lists = [selected + [candidate] for candidate in remaining]
                evaluations = [
                    _evaluate_candidate(
                        model_name=model_name,
                        candidate=candidate,
                        loaded_feature_sets=loaded_feature_sets,
                        evaluation_cache=evaluation_cache,
                        config=config,
                    )
                    for candidate in candidate_lists
                ]

                ranking = _rank_model_candidates(model_name, evaluations)
                ranking_map = {row["candidate"]: row["combined_rank"] for row in ranking.to_dict(orient="records")}
                for candidate, evaluation in zip(candidate_lists, evaluations):
                    record = evaluation.to_record(
                        step=step,
                        selected_so_far=selected.copy(),
                        combined_rank=ranking_map.get("__".join(candidate)),
                    )
                    evaluation_records.append(record)
                    _append_evaluation_record(evaluations_jsonl_path, record)
                best_candidate_name = ranking.iloc[0]["candidate"]
                best_candidate = best_candidate_name.split("__")
                next_feature_set = [name for name in best_candidate if name not in selected][0]

                selected.append(next_feature_set)
                remaining.remove(next_feature_set)
                step += 1

                final_output["iterations"].append(
                    {
                        "model": model_name,
                        "selected_so_far": selected.copy(),
                        "ranking": ranking.to_dict(orient="records"),
                    }
                )

            final_output["selected"][model_name] = selected

    else:
        selected = []
        remaining = resolved_feature_sets.copy()
        step = 0

        while len(remaining) > 0:
            candidate_lists = [selected + [candidate] for candidate in remaining]
            all_rankings = []

            per_model_rankings: dict[str, pd.DataFrame] = {}
            for model_name in models:
                evaluations = [
                    _evaluate_candidate(
                        model_name=model_name,
                        candidate=candidate,
                        loaded_feature_sets=loaded_feature_sets,
                        evaluation_cache=evaluation_cache,
                        config=config,
                    )
                    for candidate in candidate_lists
                ]
                ranking = _rank_model_candidates(model_name, evaluations)
                per_model_rankings[model_name] = ranking
                all_rankings.append(ranking[["candidate", "combined_rank"]].rename(columns={"combined_rank": model_name}))
                ranking_map = {row["candidate"]: row["combined_rank"] for row in ranking.to_dict(orient="records")}
                for candidate, evaluation in zip(candidate_lists, evaluations):
                    record = evaluation.to_record(
                        step=step,
                        selected_so_far=selected.copy(),
                        combined_rank=ranking_map.get("__".join(candidate)),
                    )
                    evaluation_records.append(record)
                    _append_evaluation_record(evaluations_jsonl_path, record)

            merged = all_rankings[0]
            for table in all_rankings[1:]:
                merged = merged.merge(table, on="candidate", how="inner")

            model_columns = [column for column in merged.columns if column != "candidate"]
            merged["global_mean_rank"] = merged[model_columns].mean(axis=1)
            merged = merged.sort_values("global_mean_rank", ascending=True).reset_index(drop=True)

            best_candidate = merged.iloc[0]["candidate"].split("__")
            next_feature_set = [name for name in best_candidate if name not in selected][0]

            selected.append(next_feature_set)
            remaining.remove(next_feature_set)
            step += 1

            final_output["iterations"].append(
                {
                    "model": "GLOBAL",
                    "selected_so_far": selected.copy(),
                    "global_ranking": merged.to_dict(orient="records"),
                    "per_model_ranking": {model_name: table.to_dict(orient="records") for model_name, table in per_model_rankings.items()},
                }
            )

        final_output["selected"] = {"global": selected}

    output_path = os.path.join(config["save_dir"], "greedy_forward_selection_results.json")
    with open(output_path, "w", encoding="UTF-8") as file:
        json.dump(final_output, file, indent=2)

    evaluations_path = os.path.join(config["save_dir"], "greedy_forward_selection_evaluations.json")
    with open(evaluations_path, "w", encoding="UTF-8") as file:
        json.dump(evaluation_records, file, indent=2)

    return final_output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration yaml file")
    return parser.parse_args()


def main() -> None:
    """Run greedy feature-set selection from config."""
    args = _parse_args()

    with open(args.config_path, "r", encoding="UTF-8") as file:
        config = yaml.safe_load(file)

    feature_sets = config.get("feature_sets", [])
    models = config.get("models", [])
    selection_mode = config.get("selection_mode", "per_model")

    if len(feature_sets) == 0:
        raise ValueError("Please provide a non-empty 'feature_sets' list in the config.")
    if len(models) == 0:
        raise ValueError("Please provide a non-empty 'models' list in the config.")

    result = greedy_forward_feature_set_selection(
        feature_sets=feature_sets,
        models=models,
        config=config,
        selection_mode=selection_mode,
    )

    print("Greedy feature selection finished.")
    print(json.dumps(result["selected"], indent=2))


if __name__ == "__main__":
    main()
