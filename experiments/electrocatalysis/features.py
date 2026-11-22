"""Build the feature tables: Julia's feature sets (configs/best_feature_generation.yaml) plus MACCS, on the cleaned data."""

import os

import numpy as np
import pandas as pd

import modules.features.custom_descriptor  # noqa: F401  # registers 'descriptor'
import modules.features.fingerprints  # noqa: F401  # registers fingerprints
import modules.features.split_fingerprints  # noqa: F401  # registers main_* / side_* features
from experiments.electrocatalysis.common import load_config, variance_filter
from modules.features.feature_factory import FeatureFactory
from modules.features.split_fingerprints import split_parts

# feature set names in configs/best_feature_generation.yaml -> short names used in the result tables
SET_NAMES = {"ecfp": "ecfp", "rf": "rf_best", "brr": "brr_best"}
APPROACH = {"substrate": "a1_full", "main_side": "a2_main_side", "substrate_main_side": "a3_all"}


def encode(smiles: list, f_type: str, params: dict, prefix: str) -> pd.DataFrame:
    """
    Fingerprint a list of SMILES; empty SMILES (a part that does not exist, e.g. no side chain) get zeros.
    """
    smiles = pd.Series(smiles)
    non_empty = (smiles != "").to_numpy()
    features, names, _ = FeatureFactory().apply([f_type], smiles[non_empty].tolist(), **{f_type: params})
    full = np.zeros((len(smiles), features.shape[1]))
    full[non_empty] = features
    return pd.DataFrame(full, columns=[f"{prefix}_{n}" for n in names])


if __name__ == "__main__":
    config = load_config()
    feature_config = load_config(config["feature_sets_config"])
    work_dir = config["work_dir"]
    features_dir = os.path.join(work_dir, "features")
    os.makedirs(features_dir, exist_ok=True)

    df = pd.read_csv(config["clean_csv"])
    df = df[df["j"].notna()].reset_index(drop=True)
    df["suspect"] = df["suspect"].fillna("")
    # a COF's two monomers in one SMILES, as in scripts/prepare_electrocatalysis.py; split features split each monomer
    df["smiles"] = [s1 if not isinstance(s2, str) or not s2 else f"{s1}.{s2}" for s1, s2 in zip(df["smiles_1"], df["smiles_2"])]
    parts = [split_parts(s) for s in df["smiles"]]
    df["main_smiles"] = [main for main, _ in parts]
    df["side_smiles"] = [".".join(sides) for _, sides in parts]
    df["has_side_chain"] = df["side_smiles"] != ""
    df[["number", "name", "paper", "type", "j", "suspect", "smiles", "main_smiles", "side_smiles", "has_side_chain"]].to_csv(os.path.join(work_dir, "data.csv"), index=False)
    print(f"{len(df)} materials, {df['paper'].nunique()} papers, {df['has_side_chain'].sum()} with side chains")

    tables = {}
    for fs in feature_config["feature_generation"]:
        prefix, variant = fs["name"].split("_", 1)
        variant = variant.replace("best_", "")
        features, names, _ = FeatureFactory().apply(fs["types"], df["smiles"].tolist(), **fs["params"])
        tables[f"{APPROACH[variant]}__{SET_NAMES[prefix]}"] = pd.DataFrame(features, columns=names)
    if config.get("add_maccs"):
        maccs = {part: encode(df[col].tolist(), "maccs", {"count": True}, part) for part, col in [("full", "smiles"), ("main", "main_smiles"), ("side", "side_smiles")]}
        tables["a1_full__maccs"] = maccs["full"]
        tables["a2_main_side__maccs"] = pd.concat([maccs["main"], maccs["side"]], axis=1)
        tables["a3_all__maccs"] = pd.concat([maccs["full"], maccs["main"], maccs["side"]], axis=1)

    for name, X in tables.items():
        X = variance_filter(X.astype(float).fillna(0.0), feature_config["remove_threshold"])
        X["is_ctf"] = (df["type"] == "CTF").astype(float)
        X.to_csv(os.path.join(features_dir, f"{name}.csv"), index=False)
        print(f"{name:28s} {X.shape[1]} features")
