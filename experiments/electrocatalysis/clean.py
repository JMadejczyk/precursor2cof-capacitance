"""Clean the electrocatalysis dataset: fix SMILES that can be fixed safely, flag the rest, log every change."""

import argparse
import os

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

DATA_DIR = "experiments/electrocatalysis/data"
COLUMNS = {
    "number": "number",
    "working name": "name",
    "literature": "paper",
    "COF/CTF": "type",
    "smiles (substrate 1)": "smiles_1",
    "smiles (substrate 2)": "smiles_2",
    "JD_max": "j",
}
# Rows whose monomers look wrong but cannot be fixed without the chemists.
SUSPECT = {
    22: "labelled CTF but the monomer has no nitrile group",
    28: "nitrile + aldehyde pair; the other rows of Li2025a use an amine",
    50: "tri-bromide + dialdehyde pair; sibling rows 48 and 49 use amines",
    53: "imine-COF with a carboxylic acid monomer; looks swapped with row 54",
    54: "amide-COF with an aldehyde monomer; looks swapped with row 53",
}

parser = argparse.ArgumentParser()
parser.add_argument("--xlsx", type=str, default=os.path.join(DATA_DIR, "data_electrocatalysis.xlsx"))
parser.add_argument("--out_dir", type=str, default=DATA_DIR)


def skeleton_key(smiles: str) -> str:
    """
    Heavy-atom graph with bond orders, aromaticity, charges and hydrogens removed.
    Two SMILES with the same key describe the same skeleton drawn with different bonds.
    :param smiles: SMILES string.
    :return: canonical SMILES of the generic skeleton.
    """
    mol = Chem.RWMol(Chem.MolFromSmiles(smiles))
    Chem.Kekulize(mol, clearAromaticFlags=True)
    for bond in mol.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
    for atom in mol.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
    return Chem.MolToSmiles(mol, canonical=True)


def n_sp3_ring_carbons(smiles: str) -> int:
    """
    Count sp3 ring carbons; in these conjugated monomers they indicate a broken aromatic drawing.
    :param smiles: SMILES string.
    :return: number of sp3 carbons in rings.
    """
    mol = Chem.MolFromSmiles(smiles)
    return sum(a.GetSymbol() == "C" and a.IsInRing() and a.GetHybridization() == Chem.HybridizationType.SP3 for a in mol.GetAtoms())


def largest_fragment(smiles: str) -> str:
    """
    Drop counter-ions such as [Br-].
    :param smiles: SMILES string.
    :return: SMILES of the largest fragment.
    """
    frags = Chem.GetMolFrags(Chem.MolFromSmiles(smiles), asMols=True)
    return Chem.MolToSmiles(max(frags, key=lambda m: m.GetNumHeavyAtoms()))


if __name__ == "__main__":
    args = parser.parse_args()
    df = pd.read_excel(args.xlsx)[list(COLUMNS)].rename(columns=COLUMNS)
    df["name"] = df["name"].fillna("")
    df["smiles_2"] = df["smiles_2"].replace("-", "")
    log = []

    def change(row: int, column: str, new: str, reason: str, confidence: str):
        old = df.loc[df["number"] == row, column].item()
        df.loc[df["number"] == row, column] = new
        log.append({"number": row, "column": column, "old": old, "new": new, "reason": reason, "confidence": confidence})

    # 1. hydrazide written as O-N-N instead of C(=O)-N-N
    for _, r in df.iterrows():
        for col in ["smiles_1", "smiles_2"]:
            if "ONN" in r[col]:
                change(r["number"], col, r[col].replace("ONN", "C(=O)NN"), "hydrazide written as ONN", "medium")

    # 2. counter-ions
    for _, r in df.iterrows():
        for col in ["smiles_1", "smiles_2"]:
            if r[col] and "." in r[col]:
                change(r["number"], col, largest_fragment(r[col]), "removed counter-ion", "high")

    # 3. broken aromatic drawings: replace with an aromatic SMILES of the same skeleton found elsewhere in the file
    all_smiles = pd.concat([df["smiles_1"], df["smiles_2"]])
    aromatic_by_key = {}
    for s in all_smiles[all_smiles != ""].unique():
        if n_sp3_ring_carbons(s) == 0:
            aromatic_by_key.setdefault(skeleton_key(s), set()).add(Chem.MolToSmiles(Chem.MolFromSmiles(s)))
    for _, r in df.iterrows():
        for col in ["smiles_1", "smiles_2"]:
            s = r[col]
            if s and n_sp3_ring_carbons(s) > 0:
                matches = aromatic_by_key.get(skeleton_key(s), set())
                if len(matches) == 1:
                    change(r["number"], col, next(iter(matches)), "broken aromatic drawing; same skeleton drawn correctly elsewhere in the file", "high")
                else:
                    log.append({"number": r["number"], "column": col, "old": s, "new": s, "reason": f"broken aromatic drawing, no unique correct match ({len(matches)} found)", "confidence": "unfixed"})

    # canonical SMILES for everything
    for col in ["smiles_1", "smiles_2"]:
        df[col] = df[col].apply(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if s else "")

    df["suspect"] = df["number"].map(SUSPECT).fillna("")
    unfixed = {e["number"] for e in log if e["confidence"] == "unfixed"}
    df.loc[df["number"].isin(unfixed) & (df["suspect"] == ""), "suspect"] = "broken aromatic drawing, not fixed"

    # only labelled rows are written: the unlabelled ones are the chemists' unpublished in-house materials
    labelled = df[df["j"].notna()]
    labelled.to_csv(os.path.join(args.out_dir, "electro_clean.csv"), index=False)
    pd.DataFrame(log).to_csv(os.path.join(args.out_dir, "corrections.csv"), index=False)
    print(pd.DataFrame(log)[["number", "column", "reason", "confidence"]].to_string(index=False))
    print(f"\n{len(df)} rows, {len(labelled)} labelled (written), {(labelled['suspect'] != '').sum()} labelled suspect rows")
