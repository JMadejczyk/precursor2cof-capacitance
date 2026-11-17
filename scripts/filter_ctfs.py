"""Script to keep only the CTF substrates (those with a nitrile group) of a dataset."""

import argparse
import os

import pandas as pd

from modules.data.utils import cft_cof_filter

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Path to the input data file with a smiles column")
parser.add_argument("--output_path", type=str, required=True, help="Path to save the CTF-only data file")

if __name__ == "__main__":
    args = parser.parse_args()
    df = pd.read_csv(args.data_path)
    df_ctfs = df[cft_cof_filter(df["smiles"])].reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df_ctfs.to_csv(args.output_path, index=False)
    print(f"{len(df_ctfs)} of {len(df)} substrates are CTFs, saved to {args.output_path}")
