"""Functions for generating features."""
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

from modules.predictor.features.utils import fingerprint_feature_types


class FeatureFactory:
    """
    Class for generating features.
    """

    _features: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Register a feature function with a given name.

        This method is used as a decorator to register a feature function
        under a specified name. The registered function can later be retrieved
        and used to generate features.

        :param name: The name to register the feature function under.
        :return: A decorator that registers the feature function.
        """

        def decorator(func: Callable) -> Callable:
            cls._features[name] = func
            return func

        return decorator

    def apply(self, names: list, smiles: list, **kwargs) -> tuple:
        """
        Apply the feature function with the given name.
        :param names: names of the feature functions.
        :param smiles: list of SMILES strings.
        :param kwargs: additional arguments for the feature function.
        :return: list of features and feature names.
        """
        features = []
        features_names = []
        feature_types_combo = {}
        for name in names:
            if name not in self._features:
                raise ValueError(f"Feature function '{name}' is not defined.")
            fp = self._features[name](**kwargs[name])
            fps = fp.fit_transform(smiles)
            if name in ["descriptor", "custom_pattern"]:
                fps_names = fp.get_feature_names_out()
                feature_types = fp.get_feature_types()
            else:
                fps_names = [f"{name}_{i}" for i in range(fps.shape[1])]
                feature_types = fingerprint_feature_types(name, fps_names, **kwargs[name])
            features.append(fps)
            features_names.extend(fps_names)
            for f_type, f_names in feature_types.items():
                if f_type not in feature_types_combo:
                    feature_types_combo[f_type] = []
                feature_types_combo[f_type].extend(f_names)
        features = np.concat(features, axis=1)
        return features, features_names, feature_types_combo


def generate_features(df: pd.DataFrame, pre_dfs: dict, smiles_col: str, target_col: str, feature_types: list[str], remove_threshold: float, **kwargs) -> tuple[pd.DataFrame, dict]:
    """
    Generates fingerprints for the given dataset.
    :param df: dataframe with SMILES and target columns.
    :param smiles_col: name of the column with SMILES.
    :param target_col: name of the target column.
    :param feature_types: list of feature types to generate.
    :param remove_threshold: threshold for removing features with low variance.
    :param kwargs: additional arguments for the fingerprint generation.
    :return: dataframe with generated fingerprints.
    """
    smiles_list = df[smiles_col].tolist()
    target_list = df[target_col].tolist()

    if len(pre_dfs) > 0:
        feature_list_parts = []
        features_names_pre = []
        feature_types_combo = {}
        for f_type, (pre_df, f_types) in pre_dfs.items():
            feature_array = pre_df.drop(columns=[smiles_col, target_col]).to_numpy()
            feature_list_parts.append(feature_array)
            features_names_pre.extend(pre_df.drop(columns=[smiles_col, target_col]).columns.tolist())
            for ft, fnames in f_types.items():
                if ft not in feature_types_combo:
                    feature_types_combo[ft] = []
                feature_types_combo[ft].extend(fnames)
        feature_list_pre = np.concatenate(feature_list_parts, axis=1)
        feature_types_pre = feature_types_combo
        feature_types = [f for f in feature_types if f not in pre_dfs.keys()]

    if len(feature_types) == 0:
        feature_list = np.empty((len(smiles_list), 0))
        features_names = []
        feature_types = {}
    else:
        feature_list, features_names, feature_types = FeatureFactory().apply(feature_types, smiles_list, **kwargs["kwargs"])

    if len(pre_dfs) > 0:
        feature_list = np.concatenate([feature_list_pre, feature_list], axis=1)
        features_names = features_names_pre + features_names
        for ft, fnames in feature_types_pre.items():
            if ft not in feature_types:
                feature_types[ft] = []
            feature_types[ft].extend(fnames)

    df_features = pd.DataFrame(feature_list, columns=features_names)

    scaler = MinMaxScaler()
    df_features_scaled = scaler.fit_transform(df_features)
    selector = VarianceThreshold(threshold=1 - remove_threshold)
    selector.fit(df_features_scaled)
    df_features = df_features.iloc[:, selector.get_support(indices=True)]

    df_features[target_col] = target_list
    df_features[smiles_col] = smiles_list

    return df_features, feature_types
