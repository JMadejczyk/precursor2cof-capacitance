"""Run train/eval for missing greedy feature-set requests."""

from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any

import yaml

from scripts.greedy_forward_selection import _evaluate_candidate, _load_feature_sets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to greedy yaml config.")
    parser.add_argument(
        "--requests_path",
        type=str,
        required=True,
        help="Path to JSON file with records: {'model': str, 'datasets': list[str]}.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where missing evaluation outputs (json/jsonl) are written.",
    )
    parser.add_argument(
        "--save_dir_override",
        type=str,
        default=None,
        help="Optional override for config['save_dir'] used by training pipelines.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip requests already present in output jsonl.",
    )
    return parser.parse_args()


def _candidate_key(candidate: list[str]) -> tuple[str, ...]:
    return tuple(sorted([name.replace(".csv", "") for name in candidate]))


def _load_existing_jsonl(path: str) -> tuple[list[dict[str, Any]], set[tuple[str, tuple[str, ...]]]]:
    if not os.path.exists(path):
        return [], set()

    records: list[dict[str, Any]] = []
    keys: set[tuple[str, tuple[str, ...]]] = set()
    with open(path, "r", encoding="UTF-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            records.append(record)
            if "model" in record and "candidate" in record:
                keys.add((record["model"], _candidate_key(record["candidate"])))
    return records, keys


def main() -> None:
    args = _parse_args()

    with open(args.config_path, "r", encoding="UTF-8") as file:
        config = yaml.safe_load(file)

    with open(args.requests_path, "r", encoding="UTF-8") as file:
        requests = json.load(file)

    if not isinstance(requests, list):
        raise ValueError("Requests file must contain a JSON list.")

    run_config = copy.deepcopy(config)
    if args.save_dir_override:
        run_config["save_dir"] = args.save_dir_override

    all_feature_sets = {dataset.replace(".csv", "") for request in requests for dataset in request.get("datasets", [])}
    if len(all_feature_sets) == 0:
        raise ValueError("No datasets found in requests.")

    loaded_feature_sets = _load_feature_sets(run_config["datasets_dir"], sorted(all_feature_sets))
    evaluation_cache = {}

    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl_path = os.path.join(args.output_dir, "missing_evaluation_results.jsonl")
    output_json_path = os.path.join(args.output_dir, "missing_evaluation_results.json")

    existing_records, existing_keys = _load_existing_jsonl(output_jsonl_path)
    records = existing_records.copy()

    evaluated_count = 0
    skipped_count = 0
    for request_index, request in enumerate(requests):
        model = request.get("model")
        datasets = request.get("datasets")

        if not isinstance(model, str) or not isinstance(datasets, list) or len(datasets) == 0:
            raise ValueError("Each request must be: {'model': <str>, 'datasets': <non-empty list[str]>}.")

        candidate = [name.replace(".csv", "") for name in datasets]
        key = (model, _candidate_key(candidate))
        if args.skip_existing and key in existing_keys:
            skipped_count += 1
            continue

        evaluation = _evaluate_candidate(
            model_name=model,
            candidate=candidate,
            loaded_feature_sets=loaded_feature_sets,
            evaluation_cache=evaluation_cache,
            config=run_config,
        )

        record = evaluation.to_record(step=-1, selected_so_far=[])
        record["request_index"] = request_index
        record["source"] = "missing_evaluation_request"

        with open(output_jsonl_path, "a", encoding="UTF-8") as file:
            file.write(json.dumps(record) + "\n")

        records.append(record)
        existing_keys.add(key)
        evaluated_count += 1

    with open(output_json_path, "w", encoding="UTF-8") as file:
        json.dump(records, file, indent=2)

    print(f"Evaluated requests: {evaluated_count}")
    print(f"Skipped existing requests: {skipped_count}")
    print(f"JSONL output: {output_jsonl_path}")
    print(f"JSON output: {output_json_path}")


if __name__ == "__main__":
    main()
