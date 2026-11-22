"""Script to build the electrocatalysis dataset (target JD_max) from the spreadsheet.

A COF made from two substrates is stored as one SMILES with both substrates joined by '.',
so whole-substrate features cover both and the main/side split is done for each substrate.
"""

import argparse
import os

import pandas as pd
from rdkit import Chem

parser = argparse.ArgumentParser()
parser.add_argument("--xlsx_path", type=str, required=True, help="Path to data_electrocatalysis.xlsx")
parser.add_argument("--output_path", type=str, default="electrocatalysis/data_electrocatalysis.csv", help="Path to save the dataset")


def canon_smiles(smiles: object) -> str | None:
    """
    Canonical SMILES, or None for an empty cell, '-' or an invalid SMILES.
    :param smiles: cell value.
    :return: canonical SMILES or None.
    """
    if not isinstance(smiles, str) or smiles.strip() in ("", "-"):
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


if __name__ == "__main__":
    args = parser.parse_args()
    data = pd.read_excel(args.xlsx_path, sheet_name="data")

    substrate_1 = data["smiles (substrate 1)"].map(canon_smiles)
    substrate_2 = data["smiles (substrate 2)"].map(canon_smiles)
    df = pd.DataFrame(
        {
            "smiles": [s1 if s2 is None else f"{s1}.{s2}" for s1, s2 in zip(substrate_1, substrate_2)],
            "JD_max": data["JD_max"],
            "cof_type": data["COF/CTF"],
            "literature": data["literature"],
            "working_name": data["working name"],
        }
    )
    labelled = df[substrate_1.notna() & df["JD_max"].notna()].reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    labelled.to_csv(args.output_path, index=False)
    print(f"{len(labelled)} of {len(df)} materials have SMILES and JD_max ({(labelled['cof_type'] == 'CTF').sum()} CTF), saved to {args.output_path}")
