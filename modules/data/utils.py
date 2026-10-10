"""Utils functions for data."""
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_data_for_regressors(df_to_transform: pd.DataFrame, dfs: tuple, numerical_features: list, categorical_features: list) -> pd.DataFrame:
    """
    Prepares data for regressors (standarization, one-hot encoding)
    :param df_to_transform: dataframe with molecules and features to transform.
    :param dfs: dataframes with molecules and features to fit (both to one-hot and standard scaler).
    :param numerical_features: features to standardize (numerical).
    :param categorical_features: features to encode (categorical).
    :return: standardized dataframe
    """
    df, _ = dfs
    if len(categorical_features) > 0:
        enc_features = pd.concat(dfs)[categorical_features]
        c_features = []
        for c in enc_features.columns:
            c_features.append(enc_features[c].unique())
        enc_features_to_transform = df_to_transform[categorical_features]
        enc = OneHotEncoder(drop="first", categories=c_features)
        enc.fit(enc_features)
        enc_features_to_transform = enc.transform(enc_features_to_transform).toarray()
        enc_features_names = enc.get_feature_names_out(categorical_features)
        df_to_transform[enc_features_names] = enc_features_to_transform
        df_to_transform.drop(categorical_features, axis=1, inplace=True)

    for c in categorical_features:
        if c in numerical_features:
            numerical_features.remove(c)

    if len(numerical_features) > 0:
        st_features = df[numerical_features]
        st_features_to_transform = df_to_transform[numerical_features]
        scaler = StandardScaler().fit(st_features.values)
        st_features_to_transform = scaler.transform(st_features_to_transform.values)
        df_to_transform[numerical_features] = st_features_to_transform

    return df_to_transform


def custom_discretization(y: pd.DataFrame, num_bins: int = 4) -> pd.DataFrame:
    """
    Discretization of the continuous target attribute.
    :param y: dataframe with target
    :param num_bins: number of capacity bins to use
    :return: discretized target attribute
    """
    binned_capacity = pd.qcut(y[y.columns[0]], q=num_bins, labels=False)
    return binned_capacity


def custom_data_kfold(X: pd.DataFrame, y: pd.DataFrame, num_splits: int, num_bins: int = 4, random_state: int = 42) -> list:
    """
    Performs custom data split on the provided data
    :param X: dataframe with molecules and generated features
    :param y: dataframe with target
    :param num_splits: number of folds
    :param num_bins: number of capacity bins to use
    :param random_state: random state (default: 42)
    :return: generated splits (indices)
    """
    binned_capacity = custom_discretization(y, num_bins)
    if len(y) < 50:
        splits = 3
        repetitions = int(np.ceil(num_splits / splits))
        skf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repetitions, random_state=random_state)
    else:
        skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    kfolds = list(skf.split(X, binned_capacity))
    return kfolds


def custom_data_split(X: pd.DataFrame, y: pd.DataFrame, train_size: float, num_bins: int = 4, random_state: int = 42) -> list:
    """

    :param X: dataframe with features
    :param y:dataframe with target
    :param train_size: required train size (percentage
    :param num_bins: number of bins for disretization
    :param random_state: random state (default: 42)
    :return: custom splits (indices)
    """
    idx = X.index.tolist()
    binned_capacity = custom_discretization(y, num_bins)
    train_ids, test_ids = train_test_split(idx, train_size=train_size, random_state=random_state, shuffle=True, stratify=binned_capacity)
    return [(train_ids, test_ids)]


def find_nitrile_group(smiles_string):
    """
    Checks for the presence of a nitrile group (-CN) in a molecule
    represented by a SMILES string.

    Args:
      smiles_string: The SMILES string of the molecule.

    Returns:
      A tuple containing a boolean indicating if the group was found,
      and the atom indices of the matches.
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        print("Invalid SMILES string.")
        return False, []

    nitrile_smarts = "[C]#[N]"
    nitrile_pattern = Chem.MolFromSmarts(nitrile_smarts)
    has_nitrile = mol.HasSubstructMatch(nitrile_pattern)
    return has_nitrile


def cft_cof_filter(smiles: pd.Series) -> pd.Series:
    """
    Filters the SMILES strings to recognize CTF molecules.
    :param smiles: pd.Series with SMILES strings.
    :return: pd.Series with boolean values indicating whether each SMILES corresponds to a CTF.
    """
    is_ctf = smiles.apply(find_nitrile_group)
    return is_ctf


def custom_data_kfold_ctf_cof(X: pd.DataFrame, y: pd.DataFrame, cof_types: pd.Series, num_splits: int, num_bins: int = 4, random_state: int = 42) -> list:
    """
    Performs custom data split on the provided data, ensuring stratification by both target bins and CTF type.
    :param X: dataframe with molecules and generated features
    :param y: dataframe with target
    :param cof_types: pd.Series with boolean values indicating CTF type
    :param num_splits: number of folds
    :param num_bins: number of capacity bins to use
    :param random_state: random state (default: 42)
    :return: generated splits (indices)
    """
    cof_types_match = cof_types.loc[X.index]
    ctf_X = X.loc[cof_types_match]
    ctf_y = y.loc[cof_types_match]
    non_ctf_indices = X.loc[~cof_types_match].index.to_numpy()

    splits = 2
    repetitions = int(np.ceil(num_splits / splits))
    skf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repetitions, random_state=random_state)
    binned_capacity = custom_discretization(ctf_y, num_bins)
    kfolds_ctf = list(skf.split(ctf_X, binned_capacity))

    final_splits = []
    for train_idx_local, test_idx_local in kfolds_ctf:
        original_ctf_train_idx = ctf_X.index[train_idx_local].to_numpy()
        original_ctf_test_idx = ctf_X.index[test_idx_local].to_numpy()

        final_train_indices = np.concatenate([original_ctf_train_idx, non_ctf_indices])
        final_test_indices = original_ctf_test_idx

        final_splits.append((final_train_indices, final_test_indices))
    return final_splits
