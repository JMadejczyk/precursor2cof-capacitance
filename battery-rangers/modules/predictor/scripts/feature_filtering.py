"""Script for filtering features based on variance thresholding/ number of unique values."""

import argparse
import os
import shutil

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument("--variance_threshold", type=float, required=True, help="Variance threshold for feature filtering")
parser.add_argument("--percent_threshold", type=float, required=True, help="Percentage threshold for unique values filtering")
parser.add_argument("--input_feature_dir", type=str, required=True, help="Directory with input feature files")

if __name__ == "__main__":
    args = parser.parse_args()
    variance_threshold = args.variance_threshold
    percent_threshold = args.percent_threshold
    input_feature_dir = args.input_feature_dir

    input_feature_dir_features = os.path.join(input_feature_dir, "features")
    input_feature_dir_types = os.path.join(input_feature_dir, "feature_types")

    input_dir = os.path.split(input_feature_dir)[0]
    output_name = f"rth_{variance_threshold}_{percent_threshold}"
    output_dir_features = os.path.join(input_dir, output_name, "features")
    output_dir_types = os.path.join(input_dir, output_name, "feature_types")

    print(output_dir_features, output_dir_types)

    os.makedirs(output_dir_features, exist_ok=True)

    if os.path.exists(output_dir_types):
        shutil.rmtree(output_dir_types)
    shutil.copytree(input_feature_dir_types, output_dir_types)

    feature_files = [f for f in os.listdir(input_feature_dir_features) if f.endswith(".csv")]
    for feature_file in feature_files:
        feature_path = os.path.join(input_feature_dir_features, feature_file)
        df = pd.read_csv(feature_path)

        features_df = df.drop(columns=["smiles", "capacity_max"], errors="ignore")

        scaler = MinMaxScaler()
        features_df_scaled = scaler.fit_transform(features_df)

        selector = VarianceThreshold(threshold=1 - variance_threshold)
        selector.fit(features_df_scaled)
        features_var_filtered = features_df.loc[:, selector.get_support()]

        features_final_filtered = features_var_filtered[[col for col in features_var_filtered.columns if features_var_filtered[col].value_counts(normalize=True).iloc[0] <= percent_threshold]]

        df_filtered = df[["smiles", "capacity_max"]].copy()
        df_filtered = pd.concat([df_filtered, features_final_filtered], axis=1)
        # Save filtered features
        output_path = os.path.join(output_dir_features, feature_file)
        df_filtered.to_csv(output_path, index=False)
        print(f"Processed {feature_file}: {features_df.shape[1]} -> {features_final_filtered.shape[1]} features")
        print(f"Filtered features saved to {output_path}")
        print("-" * 50)
    print("Feature filtering completed.")
