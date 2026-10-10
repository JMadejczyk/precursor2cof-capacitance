"""Evaluation metrics for model performance assessment."""
from typing import Callable

import numpy as np
import pandas as pd
import sklearn


class EvalMetrics:
    """
    Class for evaluation metrics.
    """

    _eval_metrics: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Register an evaluation metric function with a given name.

        This method is used as a decorator to register an evaluation metric function
        under a specified name. The registered function can later be retrieved
        and used to evaluate models' results.

        :param name: The name to register the eval function under.
        :return: A decorator that registers the eval function.
        """

        def decorator(func: Callable) -> Callable:
            cls._eval_metrics[name] = func
            return func

        return decorator

    def evaluate(self, name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Evaluate the model performance using the evaluation metric function with the given name.
        :param name: name of the evaluation metric function.
        :param y_true: Ground truth labels.
        :param y_pred: Predictions.
        :return: Evaluation metric value.
        """
        if name in self._eval_metrics:
            return self._eval_metrics[name](y_true, y_pred)
        raise ValueError(f"Evaluation metric function '{name}' is not defined.")


@EvalMetrics.register("pairwise_accuracy_score")
def average_ranking_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates average ranking score given predictions and ground truth labels.
    :param y_true: Ground truth labels.
    :param y_pred: Predictions.
    :return: Average ranking score.
    """
    df = pd.DataFrame({"Test": y_true, "Pred": y_pred})
    sorted_rank = df.sort_values(by="Test", ascending=False).reset_index(drop=True)
    sorted_rank["rank"] = sorted_rank.index + 1

    cartesian_df = sorted_rank.merge(sorted_rank, how="cross", suffixes=("_1", "_2"))
    cartesian_df = cartesian_df.loc[cartesian_df["Test_1"] != cartesian_df["Test_2"]]
    cartesian_df["target_test"] = cartesian_df["Test_1"] > cartesian_df["Test_2"]
    cartesian_df["target_pred"] = cartesian_df["Pred_1"] > cartesian_df["Pred_2"]
    cartesian_df["result"] = cartesian_df["target_test"] == cartesian_df["target_pred"]

    return cartesian_df["result"].mean()


@EvalMetrics.register("ndcg_score")
def ndcg_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates normalized discounted cumulative gain given predictions and ground truth labels.
    :param y_true: Ground truth labels.
    :param y_pred: Predictions.
    :return: Normalized discounted cumulative gain.
    """
    # y_true_rank = (
    #     pd.DataFrame(y_true, columns=["target"])
    #     .rank(ascending=True)
    #     .to_numpy()
    #     .reshape(
    #         -1,
    #     )
    # )
    # y_pred_rank = (
    #     pd.DataFrame(y_pred, columns=["target"])
    #     .rank(ascending=True)
    #     .to_numpy()
    #     .reshape(
    #         -1,
    #     )
    # )

    return sklearn.metrics.ndcg_score([y_true], [y_pred])


@EvalMetrics.register("rmse")
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates root mean squared error given predictions and ground truth labels.
    :param y_true: Ground truth labels.
    :param y_pred: Predictions.
    :return: Root mean squared error.
    """
    return sklearn.metrics.root_mean_squared_error(y_true, y_pred)


@EvalMetrics.register("mae")
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates mean absolute error given predictions and ground truth labels.
    :param y_true: Ground truth labels.
    :param y_pred: Predictions.
    :return: Mean absolute error.
    """
    return sklearn.metrics.mean_absolute_error(y_true, y_pred)


@EvalMetrics.register("mape")
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates mean absolute percentage error given predictions and ground truth labels.
    :param y_true: Ground truth labels.
    :param y_pred: Predictions.
    :return: Mean absolute percentage error.
    """
    return sklearn.metrics.mean_absolute_percentage_error(y_true, y_pred)


@EvalMetrics.register("smape")
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates symmetric mean absolute percentage error given predictions and ground truth labels.
    :param y_true: Ground truth labels.
    :param y_pred: Predictions.
    :return: Symmetric mean absolute percentage error.
    """
    epsilon = np.full(y_true.shape, 1e-10)
    return np.mean(2 * np.abs(y_pred - y_true) / (np.maximum(np.abs(y_true) + np.abs(y_pred), epsilon)))


@EvalMetrics.register("r2_score")
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates R^2 (coefficient of determination) regression score function given predictions and ground truth labels.
    :param y_true: Ground truth labels.
    :param y_pred: Predictions.
    :return: R^2 regression score.
    """
    return sklearn.metrics.r2_score(y_true, y_pred)
