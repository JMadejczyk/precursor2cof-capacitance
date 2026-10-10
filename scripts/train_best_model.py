"""Script for training machine learning models for the predictor module."""
import copy
import os

import pandas as pd

from modules.data.utils import custom_data_kfold
from scripts.greedy_forward_selection import _build_combined_dataset, _load_feature_sets
from modules.training_and_evaluation.sklearn_pipeline import SklearnTrainingPipeline
from modules.training_and_evaluation.tabpfn_pipeline import TabPFNTrainingPipeline

if __name__ == "__main__":
    datasets_dir = "../../../data/predictor/cof_matched/rth_0.99"
    results_dir = "best_model_bayes/"
    models = ["bayes"]
    hyperparameters = True

    data_config = {}
    num_bins_hyperopt = data_config.get("num_bins_hyperparam_opt", 10)
    num_bins = data_config.get("num_bins", 10)
    num_folds = data_config.get("num_folds", 10)

    # Provide one or many feature dataset names; they will be merged into one training table.
    dataset_names = "features_rth_0.99_ecfp_2_1024_True__features_rth_0.99_functional_groups_True__features_rth_0.99_descriptor_False_False__features_rth_0.99_estate_sum"
    dataset_names = dataset_names.split("__")
    loaded_feature_sets = _load_feature_sets(datasets_dir, dataset_names)
    selected_feature_sets = list(loaded_feature_sets.keys())
    dataset, feature_types = _build_combined_dataset(loaded_feature_sets, selected_feature_sets)
    combined_dataset_name = "__".join(selected_feature_sets)

    features_to_keep = [col for col in dataset.columns if col not in ["capacity_max", "smiles"]]
    features_to_keep = pd.Series(features_to_keep)

    for model in models:
        X = dataset.drop(columns=["capacity_max", "smiles"])
        # print(X)
        y = dataset[["capacity_max"]]
        folds = custom_data_kfold(X, y, num_splits=num_folds, num_bins=num_bins, random_state=42)

        save_dir = os.path.join(results_dir, model, combined_dataset_name)
        print(combined_dataset_name, model, save_dir)
        os.makedirs(save_dir, exist_ok=True)
        if os.path.exists(os.path.join(save_dir, "aggregated_results.txt")):
            continue
        if model == "tabpfn":
            pipeline = TabPFNTrainingPipeline(
                X=copy.deepcopy(X),
                y=copy.deepcopy(y),
                feature_types=feature_types,
                folds=copy.deepcopy(folds),
                metrics=[],
                save_dir=save_dir,
                data_name=combined_dataset_name,
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
                metrics=[],
                save_dir=save_dir,
                data_name=combined_dataset_name,
                hyperparam_opt=hyperparameters,
                num_bins=num_bins_hyperopt,
                verbose=True,
            )
        try:
            pipeline.train_and_save_model(model_name=model)

            features_to_keep.to_csv(os.path.join(save_dir, "features_to_keep.csv"), index=False)
        except Exception as e:
            print(f"An error occurred while training model {model} on dataset {combined_dataset_name}: {e}")
            print("=" * 50)
            print("\n")
