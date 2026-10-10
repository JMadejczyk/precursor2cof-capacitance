"""Cross-validation training pipeline."""
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd

from modules.predictor.training_and_evaluation.evaluation_metrics import EvalMetrics

optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore")


class ModelTrainingPipeline(ABC):
    """
    Cross-validation pipeline class.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        feature_types: dict,
        folds: list,
        metrics: list,
        save_dir: str,
        data_name: str,
        hyperparam_opt: bool = True,
        num_bins: int = 5,
        verbose: bool = False,
    ):
        """
        Initialize the cross-validation pipeline.
        :param X: dataframe with features.
        :param y: dataframe with target variable.
        :param folds: list with cross-validation folds.
        :param metrics: list with metrics to evaluate.
        :param save_dir: path to save scores.
        :param data_name: name of the dataset.
        :param hyperparam_opt: whether to perform hyperparameter optimization.
        :param num_bins: number of bins for stratified splitting for hyperparam optimization.
        :param verbose: whether to print model scores.
        """
        self.X = X
        self.y = y
        self.feature_types = feature_types
        self.folds = folds
        self.data_name = data_name
        self.save_dir = save_dir
        self.verbose = verbose
        self.metrics = metrics
        self.hyperparam_opt = hyperparam_opt
        self.num_bins = num_bins
        self.scores = None
        self.feature_importance = None

    @abstractmethod
    def tune_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame, model: object, param_grid: dict | None) -> object:
        pass

    def eval_model(self, y_pred: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model.
        :param y_pred: predicted values.
        :param y_test: true values.
        :return: dictionary with evaluation metrics.
        """
        partial_scores = {}
        for metric in self.metrics:
            partial_scores[metric] = EvalMetrics().evaluate(metric, y_test, y_pred)
        return partial_scores

    def init_scores(self):
        """
        Initialize scores dictionary and shap.
        """
        scores = {}
        for metric in self.metrics:
            scores[metric] = []
            scores[f"baseline_{metric}"] = []
        self.scores = scores
        self.feature_importance = []

    def update_scores(self, model_scores: dict, baselines: dict, f_importance: tuple | None = None):
        """
        Update scores dictionary.
        :param model_scores: dictionary with model scores.
        :param baselines: dictionary with baseline scores.
        :param f_importance: tuple with method of calculate and feature importance scores.
        """
        for metric in self.metrics:
            self.scores[metric].append(model_scores[metric])
            self.scores[f"baseline_{metric}"].append(baselines[f"{metric}"])
        if f_importance is not None:
            self.feature_importance.append(f_importance)
        else:
            self.feature_importance.append((None, None))

    def aggregate_scores(self):
        """
        Aggregate scores.
        :return: aggregated scores.
        """
        results = {}
        for metric in self.metrics:
            metric_scores = self.scores[metric]
            results[metric] = round(sum(metric_scores) / len(metric_scores), 4)
            results[f"baseline_{metric}"] = round(sum(self.scores[f"baseline_{metric}"]) / len(self.scores[f"baseline_{metric}"]), 4)
        return results

    def save_model(self, model: object, fold_num: int):
        """
        Save the trained model to a file.
        :param model: trained model.
        :param fold_num: fold number for saving.
        """
        model_save_dir = os.path.join(self.save_dir, "models")
        os.makedirs(model_save_dir, exist_ok=True)
        save_model_path = os.path.join(model_save_dir, f"model_{fold_num}.joblib")
        joblib.dump(model, save_model_path)
        if self.verbose:
            print(f"Model saved to {save_model_path}")

    def save_results(self, results: dict, model_name: str, model_params: Any):
        """
        Save the results to a file.
        :param results: dictionary with results.
        :param model_name: name of the model.
        :param model_params: parameters of the model.
        """
        results_save_dir = self.save_dir
        os.makedirs(results_save_dir, exist_ok=True)
        results_path = os.path.join(results_save_dir, f"aggregated_results.txt")
        with open(results_path, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Parameters: {model_params}\n")
            for metric, score in results.items():
                f.write(f"{metric}: {score}\n")
        if self.verbose:
            print(f"Results saved to {results_path}")
        # save partial scores
        partial_scores_path = os.path.join(results_save_dir, f"scores.pickle")
        with open(partial_scores_path, "wb") as f:
            joblib.dump(self.scores, f)
        if self.verbose:
            print(f"Partial scores saved to {partial_scores_path}")
        # save feature importance
        f_importance_path = os.path.join(results_save_dir, f"feature_importance.pickle")
        with open(f_importance_path, "wb") as f:
            joblib.dump(self.feature_importance, f)
        if self.verbose:
            print(f"Feature importance saved to {f_importance_path}")

    @abstractmethod
    def train_pipeline(self, model_name: str, model_path: str | None = None) -> tuple:
        pass

    @abstractmethod
    def calculate_f_importance(self, model: object, method: str, X_test: pd.DataFrame, X_train: pd.DataFrame) -> tuple:
        pass
