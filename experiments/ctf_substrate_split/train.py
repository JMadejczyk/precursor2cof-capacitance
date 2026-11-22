"""Train models on the CTF substrate split feature tables with shared repeated stratified k-fold CV."""

import argparse
import copy
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import RepeatedStratifiedKFold
from tabpfn import TabPFNRegressor

from modules.data.utils import custom_discretization
from modules.training_and_evaluation.sklearn_pipeline import SklearnTrainingPipeline
from modules.training_and_evaluation.tabpfn_pipeline import TabPFNTrainingPipeline

TARGET_COL = "capacity_max"

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="experiments/ctf_substrate_split/config.yaml", help="Path to the experiment config")
parser.add_argument("--models", type=str, nargs="*", default=None, help="Override the list of models from the config")


class SklearnPipelineOptionalShap(SklearnTrainingPipeline):
    """
    Sklearn pipeline that can skip SHAP and keeps out-of-fold predictions.
    """

    def __init__(self, *args, compute_shap: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_shap = compute_shap

    def calculate_f_importance(self, model: object, method: str, X_test: pd.DataFrame, X_train: pd.DataFrame) -> tuple:
        if not self.compute_shap:
            return None, None
        return super().calculate_f_importance(model, method, X_test, X_train)

    def fold_predictions(self) -> list:
        """
        Recompute out-of-fold predictions from the saved fold models.
        :return: list of (test_idx, y_pred) per fold.
        """
        preds = []
        for i, (train_idx, test_idx) in enumerate(self.folds):
            model = joblib.load(os.path.join(self.save_dir, "models", f"model_{i}.joblib"))
            X_train = self.X.loc[train_idx, :].reset_index(drop=True)
            X_test = self._data_preparation(X_train, self.X.loc[test_idx, :].reset_index(drop=True))
            preds.append((test_idx, model.predict(X_test.to_numpy()).flatten()))
        return preds


class PlainTabPFNPipeline(TabPFNTrainingPipeline):
    """
    TabPFN with default settings (no TunedTabPFN search), configurable device and optional SHAP.
    """

    def __init__(self, *args, device: str = "cpu", compute_shap: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.compute_shap = compute_shap
        self.predictions = []

    def train_pipeline(self, model_name: str, model_path: str | None = None) -> tuple:
        self.init_scores()
        model_params = []
        for train_idx, test_idx in self.folds:
            X_train = self.X.loc[train_idx, :].reset_index(drop=True)
            y_train = self.y.loc[train_idx, TARGET_COL].to_numpy()
            X_test = self.X.loc[test_idx, :].reset_index(drop=True)
            y_test = self.y.loc[test_idx, TARGET_COL].to_numpy()

            model = TabPFNRegressor(device=self.device, random_state=42, ignore_pretraining_limits=True)
            model.fit(X_train.to_numpy(), y_train)
            y_pred = model.predict(X_test.to_numpy()).flatten()

            f_imp = self.calculate_f_importance(model, "TabPFN", X_test, X_train) if self.compute_shap else None
            baseline = np.median(y_train) * np.ones_like(y_test)
            self.update_scores(self.eval_model(y_pred, y_test), self.eval_model(baseline, y_test), f_imp)
            self.predictions.append((test_idx, y_pred))
            model_params.append(model.get_params())

        results = self.aggregate_scores()
        self.save_results(results, "TabPFN Regressor", model_params)
        return results, self.scores, self.feature_importance, model_params

    def fold_predictions(self) -> list:
        return self.predictions


def make_folds(data: pd.DataFrame, cv_config: dict) -> list:
    """
    Repeated stratified k-fold on the binned target of CTF rows; identical for all approaches and train scopes.
    Test sets contain only CTFs, non-CTF rows (if present) are always added to training.
    :param data: dataframe with target and is_ctf column.
    :param cv_config: cv section of the config.
    :return: list of (train_idx, test_idx).
    """
    ctf_rows = data.index[data["is_ctf"]].to_numpy()
    non_ctf_rows = data.index[~data["is_ctf"]].to_numpy()
    binned = custom_discretization(data.loc[ctf_rows, [TARGET_COL]].reset_index(drop=True), cv_config["num_bins"])
    rskf = RepeatedStratifiedKFold(n_splits=cv_config["num_folds"], n_repeats=cv_config["num_repeats"], random_state=cv_config["random_state"])
    return [(np.concatenate([ctf_rows[train], non_ctf_rows]), ctf_rows[test]) for train, test in rskf.split(np.zeros(len(ctf_rows)), binned)]


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config_path, "r", encoding="UTF-8") as f:
        config = yaml.safe_load(f)

    work_dir = config["work_dir"]
    features_dir = os.path.join(work_dir, "features")
    runs_dir = os.path.join(work_dir, "runs")
    models = args.models or config["models"]
    shap_models = set(config.get("shap_models", []))
    num_folds = config["cv"]["num_folds"]

    data = pd.read_csv(os.path.join(work_dir, "data.csv"))
    folds = make_folds(data, config["cv"])
    joblib.dump(folds, os.path.join(work_dir, "folds.pickle"))

    for d_file in sorted(os.listdir(features_dir)):
        d_name = d_file.replace(".csv", "")
        approach, rep_name = d_name.split("__")
        dataset = pd.read_csv(os.path.join(features_dir, d_file))
        X = dataset.drop(columns=[TARGET_COL, "smiles"])
        y = dataset[[TARGET_COL]]
        feature_types = {"numerical": X.columns.tolist(), "categorical": [], "binary": []}

        for model in models:
            save_dir = os.path.join(runs_dir, model, d_name)
            if os.path.exists(os.path.join(save_dir, "predictions.csv")):
                print(f"skip {model} {d_name}")
                continue
            os.makedirs(save_dir, exist_ok=True)
            print(f"{model} {d_name} ({X.shape[1]} features)")

            common = dict(
                X=copy.deepcopy(X),
                y=copy.deepcopy(y),
                feature_types=feature_types,
                folds=copy.deepcopy(folds),
                metrics=config["metrics"],
                save_dir=save_dir,
                data_name=d_name,
                hyperparam_opt=config["hyperparam_opt"],
                num_bins=config["cv"]["num_bins"],
                verbose=False,
            )
            if model == "tabpfn":
                pipeline = PlainTabPFNPipeline(**common, device=config["tabpfn"]["device"], compute_shap=model in shap_models)
            else:
                pipeline = SklearnPipelineOptionalShap(**common, compute_shap=model in shap_models)

            results, _, _, _ = pipeline.train_pipeline(model_name=model)
            print({k: v for k, v in results.items() if not k.startswith("baseline_")})

            rows = []
            for fold_i, (test_idx, y_pred) in enumerate(pipeline.fold_predictions()):
                for idx, pred in zip(test_idx, y_pred):
                    rows.append({
                        "approach": approach,
                        "representation": rep_name,
                        "model": model,
                        "fold": fold_i,
                        "repeat": fold_i // num_folds,
                        "row": idx,
                        "idx": data.loc[idx, "idx"],
                        "has_side_chain": isinstance(data.loc[idx, "side_chain_smiles"], str),
                        "y_true": data.loc[idx, TARGET_COL],
                        "y_pred": pred,
                    })
            pd.DataFrame(rows).to_csv(os.path.join(save_dir, "predictions.csv"), index=False)
