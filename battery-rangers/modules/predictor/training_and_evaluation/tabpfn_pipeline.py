"""TabPFN training pipeline module."""
import copy
import os
import random

import numpy as np
import pandas as pd
import shap
import torch
from tabpfn import load_fitted_tabpfn_model, save_fitted_tabpfn_model
from tabpfn_extensions import TunedTabPFNRegressor, interpretability
from tabpfn_extensions.hpo import TabPFNSearchSpace

from modules.predictor.training_and_evaluation.training_pipeline import ModelTrainingPipeline

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TabPFNTrainingPipeline(ModelTrainingPipeline):
    """
    Cross-validation pipeline class for sklearn models.
    """

    def tune_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame, model: object, param_grid: dict | None) -> object:
        return None

    def train_pipeline(self, model_name: str, model_path: str | None = None) -> tuple:
        """
        Train the model.
        :param model_name: name of the model.
        :param model_path: path to saved model.
        :return: tuple with results, scores, explanations, model parameters.
        """
        custom_space = TabPFNSearchSpace.get_classifier_space(n_ensemble_range=(2, 8))
        if len(self.X.columns) > 500:
            custom_space["ignore_pretraining_limits"] = [True]
        proper_model_name = "TabPFN Regressor"

        self.init_scores()

        if self.verbose:
            print(f"Training model {proper_model_name}")

        model_params = []

        for i, fold in enumerate(self.folds):
            model = TunedTabPFNRegressor(random_state=42, n_validation_size=0.3, device="cuda", n_trials=100, search_space=custom_space, metric="rmse")

            train_idx, test_idx = fold

            # train-test split
            X_train = copy.deepcopy(self.X.loc[train_idx, :]).reset_index(drop=True)
            y_train = copy.deepcopy(self.y.loc[train_idx, :]).reset_index(drop=True)
            X_test = copy.deepcopy(self.X.loc[test_idx, :]).reset_index(drop=True)
            y_test = copy.deepcopy(self.y.loc[test_idx, :]).reset_index(drop=True)

            model.fit(X_train.to_numpy(), y_train[y_train.columns[0]].to_numpy())
            model = model.best_model_
            model.fit(X_train.to_numpy(), y_train[y_train.columns[0]].to_numpy())

            y_pred = model.predict(X_test.to_numpy()).flatten()

            f_imp = self.calculate_f_importance(model, "TabPFN", X_test, X_train)

            # model eval
            y_test_numpy = y_test.to_numpy().flatten()
            y_pred_eval = self.eval_model(y_pred, y_test_numpy)

            baseline = np.median(y_train[y_train.columns[0]].to_numpy()) * np.ones_like(y_test_numpy)
            baselines = self.eval_model(baseline, y_test_numpy)

            self.update_scores(y_pred_eval, baselines, f_imp)
            if len(self.save_dir) > 0:
                self.save_model(model, fold_num=i)
            model_params.append(model.get_params())

        results = self.aggregate_scores()
        self.save_results(results, proper_model_name, model_params)
        return results, self.scores, self.feature_importance, model_params

    def save_model(self, model: object, fold_num: int = None) -> object:
        model_save_dir = os.path.join(self.save_dir, "models")
        os.makedirs(model_save_dir, exist_ok=True)
        save_model_path = os.path.join(model_save_dir, f"model_{fold_num}.tabpfn_fit")
        model.device = "cpu"
        save_fitted_tabpfn_model(model, save_model_path)

    def load_model(self, fold_num: int = None) -> object:
        model_save_dir = os.path.join(self.save_dir, "models")
        save_model_path = os.path.join(model_save_dir, f"model_{fold_num}.tabpfn_fit")
        reg_cpu = load_fitted_tabpfn_model(save_model_path, device="cpu")
        return reg_cpu

    def calculate_f_importance(self, model: object, method: str, X_test: pd.DataFrame, X_train: pd.DataFrame) -> tuple:
        """
        Calculate feature importance using SHAP.
        :param model: model to be explained.
        :param method: type of SHAP explainer.
        :param X_test: test data.
        :param X_train: train data.
        :return: tuple with method and SHAP values.
        """
        found_ecfp = [1 for col in X_test.columns if "ecfp" in col]
        found_bcut = [1 for col in X_test.columns if "bcut" in col]
        if len(found_ecfp) > 0 or len(found_bcut) > 0:
            return method, []
        check_explainer = shap.Explainer(model.predict, X_test.to_numpy())
        if isinstance(check_explainer, shap.ExactExplainer):
            shap_values = interpretability.shap.get_shap_values(
                estimator=model,
                test_x=X_test.to_numpy(),
                attribute_names=X_test.columns.tolist(),
            )
        else:
            shap_values = interpretability.shap.get_shap_values(
                estimator=model,
                test_x=X_test.to_numpy(),
                attribute_names=X_test.columns.tolist(),
                background=X_train.to_numpy(),
                max_evals=2 * len(X_test.columns) + 100,
            )
        return method, shap_values

    def train_and_save_model(self, model_name: str, model_path: str | None = None) -> object:
        """
        Train the model on the entire dataset and save it.
        :param model_name: name of the model.
        :param model_path: path to saved model.
        :return: trained model.
        """
        X, y = copy.deepcopy(self.X), copy.deepcopy(self.y)
        custom_space = TabPFNSearchSpace.get_classifier_space(n_ensemble_range=(2, 8))
        if len(self.X.columns) > 500:
            custom_space["ignore_pretraining_limits"] = [True]
        proper_model_name = "TabPFN Regressor"
        model = TunedTabPFNRegressor(random_state=42, n_validation_size=0.3, device="cuda", n_trials=100, search_space=custom_space, metric="rmse")

        model.fit(X.to_numpy(), y[y.columns[0]].to_numpy())
        model = model.best_model_
        model.fit(X.to_numpy(), y[y.columns[0]].to_numpy())

        self.save_model(model, fold_num="final")

        return model
