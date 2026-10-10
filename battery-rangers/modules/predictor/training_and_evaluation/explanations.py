"""Explanation of the model."""
from typing import Iterable

import pandas as pd
import shap

# pylint: disable=invalid-name


def explain_model(model: object, X_test: pd.DataFrame) -> Iterable:
    """
    Explain the model.
    :param model: prediction model.
    :param X_test: test data.
    :return: shap values.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except ValueError:
        # If the model is not tree-based, use KernelExplainer
        explainer = shap.KernelExplainer(model.predict, X_test)
        shap_values = explainer.shap_values(X_test, nsamples="auto")
    return shap_values
