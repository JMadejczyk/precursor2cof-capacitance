"""Script for training machine learning models for the predictor module."""
import argparse
import copy
import os

import pandas as pd
import yaml

from modules.predictor.data.utils import cft_cof_filter, custom_data_kfold, custom_data_kfold_ctf_cof
from modules.predictor.training_and_evaluation.chemprop_pipeline import ChempropTrainingPipeline
from modules.predictor.training_and_evaluation.sklearn_pipeline import SklearnTrainingPipeline
from modules.predictor.training_and_evaluation.tabpfn_pipeline import TabPFNTrainingPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration yaml file")

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="UTF-8") as file:
        config = yaml.safe_load(file)

    datasets_dir = config["datasets_dir"]
    feature_dir = os.path.join(datasets_dir, "features")
    types_dir = os.path.join(datasets_dir, "feature_types")
    results_dir = config["save_dir"]
    models_to_train = config["models"]
    hyperparameters = config["hyperparam_opt"]
    metrics = config["metrics"]

    data_config = config.get("dataset_params", {})
    num_bins_hyperopt = data_config.get("num_bins_hyperparam_opt", 5)
    num_bins = data_config.get("num_bins", 5)
    num_folds = data_config.get("num_folds", 5)

    # load datasets and feature types
    dataset_names = os.listdir(feature_dir)
    datasets = {}
    for d_name in dataset_names:
        dataset = pd.read_csv(os.path.join(feature_dir, d_name))
        with open(os.path.join(types_dir, d_name.replace("features", "feature_types").replace(".csv", ".yaml")), "r", encoding="UTF-8") as f:
            feature_types = yaml.safe_load(f)
        datasets[d_name.replace(".csv", "")] = (dataset, feature_types)

    for model in models_to_train:
        if model == "chemprop":
            chemprop_config = config.get("chemprop_params", {})
            chemprop_path = chemprop_config.get("chemprop_model_path", None)
            if chemprop_path is None:
                raise ValueError("Please provide chemprop_model_path for Chemprop model in the configuration file.")
            data_path = chemprop_config.get("chemprop_dataset_path", None)
            if data_path is None:
                raise ValueError("Please provide data_path for Chemprop model in the configuration file.")
            smiles_dataset = pd.read_csv(data_path)
            datasets_chemprop = copy.deepcopy(datasets)
            datasets_chemprop["smiles"] = (smiles_dataset, "")
            for d_name in datasets_chemprop:
                dataset, feature_types = datasets_chemprop[d_name]
                X = dataset.drop(columns=["capacity_max"])
                y = dataset[["capacity_max"]]
                folds = custom_data_kfold(X, y, num_splits=num_folds, num_bins=num_bins, random_state=42)

                save_dir = os.path.join(results_dir, model, d_name)
                print(model, save_dir)
                os.makedirs(save_dir, exist_ok=True)
                if os.path.exists(os.path.join(save_dir, "aggregated_results.txt")):
                    continue

                pipeline = ChempropTrainingPipeline(
                    X=copy.deepcopy(X),
                    y=copy.deepcopy(y),
                    feature_types=feature_types,
                    folds=copy.deepcopy(folds),
                    metrics=metrics,
                    save_dir=save_dir,
                    data_name="smiles",
                    model_path=chemprop_path,
                    verbose=True,
                )

                results, scores, f_importance, model_params = pipeline.train_pipeline(model_name=model)
                print(f"Results for dataset smiles with model {model}:")
                print(results)
                print(f"Scores:")
                print(scores)
                print("Model hyperparameters:")
                print(model_params)
                print("=" * 50)
                print("\n")

        else:
            for d_name in datasets:
                dataset, feature_types = datasets[d_name]
                X = dataset.drop(columns=["capacity_max", "smiles"])
                y = dataset[["capacity_max"]]
                folds = custom_data_kfold(X, y, num_splits=num_folds, num_bins=num_bins, random_state=42)

                save_dir = os.path.join(results_dir, model, d_name)
                print(d_name, model, save_dir)
                os.makedirs(save_dir, exist_ok=True)
                if os.path.exists(os.path.join(save_dir, "aggregated_results.txt")):
                    continue
                if model == "tabpfn":
                    pipeline = TabPFNTrainingPipeline(
                        X=copy.deepcopy(X),
                        y=copy.deepcopy(y),
                        feature_types=feature_types,
                        folds=copy.deepcopy(folds),
                        metrics=metrics,
                        save_dir=save_dir,
                        data_name=d_name,
                        hyperparam_opt=hyperparameters,
                        num_bins=num_bins_hyperopt,
                        verbose=True,
                    )
                else:
                    pipeline = SklearnTrainingPipeline(
                        X=copy.deepcopy(X),
                        y=copy.deepcopy(y),
                        feature_types=feature_types,
                        folds=copy.deepcopy(folds),
                        metrics=metrics,
                        save_dir=save_dir,
                        data_name=d_name,
                        hyperparam_opt=hyperparameters,
                        num_bins=num_bins_hyperopt,
                        verbose=True,
                    )
                try:
                    results, scores, f_importance, model_params = pipeline.train_pipeline(model_name=model)
                    print(f"Results for dataset {d_name} with model {model}:")
                    print(results)
                    print(f"Scores:")
                    print(scores)
                    print("Model hyperparameters:")
                    print(model_params)
                    print("=" * 50)
                    print("\n")
                except Exception as e:
                    print(f"An error occurred while training model {model} on dataset {d_name}: {e}")
                    print("=" * 50)
                    print("\n")
