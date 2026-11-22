"""Build feature tables for the CTF substrate split experiment (approaches 1-3)."""

import argparse
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

import modules.features.fingerprints  # noqa: F401  # registers fingerprints in FeatureFactory
from modules.features.feature_factory import FeatureFactory

SMILES_COLS = {"full": "smiles", "main": "main_chain_smiles", "side": "side_chain_smiles"}
TARGET_COL = "capacity_max"

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="experiments/ctf_substrate_split/config.yaml", help="Path to the experiment config")


def load_data(source_path: str, ctf_only: bool) -> pd.DataFrame:
    """
    Load the substrate split table, optionally keeping only CTF substrates.
    :param source_path: path to substrate_split.csv.
    :param ctf_only: whether to keep only CTF substrates.
    :return: dataframe with full / main / side SMILES and target.
    """
    df = pd.read_csv(source_path)
    if ctf_only:
        df = df[df["is_ctf"]].reset_index(drop=True)
    df["side_chain_smiles"] = df["side_chain_smiles"].fillna("")
    return df


def encode_part(smiles: pd.Series, f_type: str, f_params: dict, prefix: str) -> pd.DataFrame:
    """
    Encode one part of the substrate. Empty SMILES (no side chain) get an all-zero vector.
    :param smiles: SMILES strings of the part.
    :param f_type: fingerprint name registered in FeatureFactory.
    :param f_params: fingerprint parameters.
    :param prefix: prefix for the column names (part name).
    :return: dataframe with features.
    """
    non_empty = smiles != ""
    features, names, _ = FeatureFactory().apply([f_type], smiles[non_empty].tolist(), **{f_type: f_params})
    full = np.zeros((len(smiles), features.shape[1]))
    full[non_empty.to_numpy()] = features
    return pd.DataFrame(full, columns=[f"{prefix}__{n}" for n in names])


def variance_filter(df_features: pd.DataFrame, remove_threshold: float) -> pd.DataFrame:
    """
    Remove low-variance features, same rule as modules.features.feature_factory.generate_features.
    :param df_features: dataframe with features.
    :param remove_threshold: threshold for removing features with low variance.
    :return: filtered dataframe.
    """
    scaled = MinMaxScaler().fit_transform(df_features)
    selector = VarianceThreshold(threshold=1 - remove_threshold).fit(scaled)
    return df_features.iloc[:, selector.get_support(indices=True)]


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config_path, "r", encoding="UTF-8") as f:
        config = yaml.safe_load(f)

    df = load_data(config["source_path"], ctf_only=config["train_scope"] == "ctf")
    print(f"Loaded {len(df)} substrates ({df['is_ctf'].sum()} CTF, {(df['side_chain_smiles'] != '').sum()} with side chains).")

    features_dir = os.path.join(config["work_dir"], "features")
    os.makedirs(features_dir, exist_ok=True)
    df[["idx", "smiles", "main_chain_smiles", "side_chain_smiles", "is_ctf", TARGET_COL]].to_csv(os.path.join(config["work_dir"], "data.csv"), index=False)

    for rep_name, rep in config["representations"].items():
        parts = {part: encode_part(df[col], rep["type"], rep["params"], part) for part, col in SMILES_COLS.items()}
        for approach, used_parts in config["approaches"].items():
            df_features = pd.concat([parts[p] for p in used_parts], axis=1)
            df_features = variance_filter(df_features, config["remove_threshold"])
            df_features[TARGET_COL] = df[TARGET_COL]
            df_features["smiles"] = df["smiles"]
            out_path = os.path.join(features_dir, f"{approach}__{rep_name}.csv")
            df_features.to_csv(out_path, index=False)
            print(f"{approach:14s} {rep_name:20s} -> {df_features.shape[1] - 2} features")
