"""Functions for model training"""

import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.metrics import make_scorer

from modules.predictor.data.utils import custom_data_split
from modules.predictor.training_and_evaluation.evaluation_metrics import smape

# pylint: disable=invalid-name


def feature_search(model: object, X: pd.DataFrame, y: pd.DataFrame, train_size: float, fixed_features: list) -> list:
    """
    Performs feature selection using the provided list of features (optimization of RMSE).
    :param model: prediction model.
    :param X: dataframe with features.
    :param y: dataframe with target variable.
    :param train_size: size of the training set (proportion).
    :param fixed_features: list with features to keep.
    :return: best score and best features.
    """
    scorer = make_scorer(smape, greater_is_better=False)
    folds = custom_data_split(X, y, train_size=train_size)
    fixed_features_ids = tuple(X.columns.get_loc(f) for f in fixed_features)

    sfs = SequentialFeatureSelector(
        model, k_features=(max(len(fixed_features), 1), len(X.columns)), forward=True, floating=False, scoring=scorer, cv=folds, n_jobs=-1, fixed_features=fixed_features_ids
    )
    sfs.fit(X, y[y.columns[0]])
    print(sfs.k_feature_names_, sfs.k_score_)
    return sfs.k_feature_names_
