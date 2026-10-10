"""Script to generate molecular features based on specified configurations."""
import argparse
import os

import pandas as pd
import yaml

from modules.predictor.features.custom_descriptor import *
from modules.predictor.features.custom_patterns import *
from modules.predictor.features.feature_factory import *
from modules.predictor.features.fingerprints import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Path to the input data file - raw dataset")
parser.add_argument("--output_dir", type=str, required=True, help="Path to directory to save the processed data")
parser.add_argument("--config", type=str, required=True, help="Path to the feature generation configuration file")

if __name__ == "__main__":
    args = parser.parse_args()
    df = pd.read_csv(args.data_path)

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    combinations = config.get("feature_generation", [])

    print(f"Generating features for {len(combinations)} combinations.")

    rth = config.get("remove_threshold", 1.0)

    for combo in combinations:
        f_types = combo.get("types", [])
        f_params = combo.get("params", {})
        print(f"Generating features for combination: {f_types}")
        print(f"With parameters: {f_params}")

        features_strings = [f"{f_type}_" + "_".join(str(v) for v in f_params.get(f_type, {}).values()) for f_type in f_types]
        output_dir_csv = os.path.join(args.output_dir, f"rth_{rth}", "features")
        output_dir_ftypes = os.path.join(args.output_dir, f"rth_{rth}", "feature_types")
        output_path_csv = os.path.join(f"{output_dir_csv}", f"features_rth_{rth}_" + "_".join(features_strings) + ".csv")
        output_path_ftypes = os.path.join(f"{output_dir_ftypes}", f"feature_types_rth_{rth}_" + "_".join(features_strings) + ".yaml")

        df_created = {}
        for i, feature_string in enumerate(features_strings):
            feature_path = os.path.join(f"{output_dir_csv}", f"features_rth_{rth}_" + feature_string + ".csv")
            types_path = os.path.join(f"{output_dir_ftypes}", f"feature_types_rth_{rth}_" + feature_string + ".yaml")
            if os.path.exists(feature_path):
                df_feat = pd.read_csv(feature_path)
                with open(types_path, "r") as f:
                    dict_types = yaml.load(f, Loader=yaml.FullLoader)
                df_created[f_types[i]] = (df_feat, dict_types)

        if os.path.exists(output_path_csv):
            print(f"Features already exist at {output_path_csv}, skipping generation.")
            print("-" * 50)
            continue

        df_features, dict_feature_types = generate_features(df, df_created, smiles_col="smiles", target_col="capacity_max", feature_types=f_types, remove_threshold=rth, kwargs=f_params)

        os.makedirs(os.path.dirname(output_path_csv), exist_ok=True)
        os.makedirs(os.path.dirname(output_path_ftypes), exist_ok=True)

        df_features.to_csv(output_path_csv, index=False)

        with open(output_path_ftypes, "w") as f:
            yaml.dump(dict_feature_types, f)

        print(f"Features saved to {output_path_csv}")
        print("-" * 50)
    print("Feature generation completed.")
