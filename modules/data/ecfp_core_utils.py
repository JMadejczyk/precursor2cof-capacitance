"""Utilities to compute a largest common subgraph (MCS), split molecules into
core/side parts and partition ECFP bits into core-only vs other bits.

Functions implemented:
- compute_mcs_smarts(smiles_list, timeout=10)
- get_mcs_matches(smarts, mol)
- split_core_side_by_smarts(smiles, smarts)
- partition_ecfp_bits_for_molecule(smiles, core_atom_idxs, radius=2, n_bits=2048, use_chiral=True, strict=True)
- split_ecfp_featureframe(features_df, core_bits, n_bits)
- aggregate_shap_with_core(shap_expl, features_df, core_bits, sample_index=0)

These are written to match the behaviour described by the user: a bit is
considered "core-only" only when none of its contributing atom environments
reach atoms outside the core (strict rule). If a bit has any instance that
touches side-group atoms, it is classified into the "other" partition.

The code relies on RDKit and SHAP being available in the environment.
"""

from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import shap
from rdkit import Chem
from rdkit.Chem import rdFMCS


def compute_mcs_smarts(smiles_list: List[str], timeout: int = 10) -> Tuple[str, Dict[int, Tuple[int, ...]]]:
    """Compute MCS SMARTS and return SMARTS + first match per molecule.

    Returns
    -------
    smarts: str
        SMARTS string for the MCS (empty string if none found)
    matches: dict
        mapping molecule index -> first tuple of atom indices matching the MCS
        in that molecule. If a molecule has no match, it will map to an empty tuple.
    """
    if not smiles_list:
        raise ValueError("smiles_list must contain at least one SMILES string")

    mols = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            raise ValueError(f"Invalid SMILES: {s}")
        mols.append(m)

    mcs_res = rdFMCS.FindMCS(mols, timeout=timeout)
    smarts = mcs_res.smartsString or ""
    print(smarts)
    # smarts = 'N#Cc(cc1)ccc1-[n](cc1)c(cc2)c1[n]2-c(cc1)ccc1#N'

    matches: Dict[int, Tuple[int, ...]] = {}
    if not smarts:
        for i in range(len(mols)):
            matches[i] = tuple()
        return smarts, matches

    patt = Chem.MolFromSmarts(smarts)
    for i, m in enumerate(mols):
        ms = m.GetSubstructMatches(patt)
        matches[i] = ms[0] if ms else tuple()

    return smarts, matches


def get_mcs_matches(smarts: str, mol: Chem.Mol) -> List[Tuple[int, ...]]:
    """Return all substructure matches for SMARTS in a molecule."""
    if not smarts:
        return []
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return []
    return list(mol.GetSubstructMatches(patt))


def split_core_side_by_smarts(smiles: str, smarts: str) -> Dict[str, object]:
    """Given a SMILES and SMARTS describing the core, return atom index sets
    for core and side, and a canonical SMILES for the core fragment.

    Returns dict with keys: 'mol', 'core_atoms', 'side_atoms', 'core_smiles'
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    if not smarts:
        # No core: everything is side
        all_atoms = set(range(mol.GetNumAtoms()))
        return {"mol": mol, "core_atoms": set(), "side_atoms": all_atoms, "core_smiles": ""}

    patt = Chem.MolFromSmarts(smarts)
    matches = mol.GetSubstructMatches(patt)
    if not matches:
        all_atoms = set(range(mol.GetNumAtoms()))
        return {"mol": mol, "core_atoms": set(), "side_atoms": all_atoms, "core_smiles": ""}

    # take first match
    match = matches[0]
    core_atoms = set(match)
    side_atoms = set(range(mol.GetNumAtoms())) - core_atoms

    # create a fragment mol for the core (keep original atom mapping)
    emol = Chem.RWMol(mol)
    # Remove atoms not in core in reverse order to preserve indices
    for idx in sorted(side_atoms, reverse=True):
        emol.RemoveAtom(idx)
    core_mol = emol.GetMol()
    core_smiles = Chem.MolToSmiles(core_mol)

    return {"mol": mol, "core_atoms": core_atoms, "side_atoms": side_atoms, "core_smiles": core_smiles}


def _instance_atoms_for_bit(mol: Chem.Mol, center_atom: int, radius: int) -> Set[int]:
    """Return atom indices included in a particular Morgan environment
    instance (center atom + all atoms in the radius-neighbourhood bonds).
    """
    if radius == 0:
        return {center_atom}
    env_bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom)
    atoms = {center_atom}
    for bidx in env_bonds:
        bond = mol.GetBondWithIdx(bidx)
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())
    return atoms


def partition_ecfp_bits_for_molecule(smiles: str, core_atom_idxs: Set[int], radius: int = 2, n_bits: int = 2048, use_chiral: bool = True) -> Dict[str, object]:
    """Partition ECFP bit indices for a single molecule into core-only and other.

    Parameters
    ----------
    smiles: SMILES string
    core_atom_idxs: set of atom indices (0-based) that belong to the core
    radius: radius of the Morgan fingerprint
    n_bits: number of bits in the Morgan fingerprint
    use_chiral: whether to use chiral information

    Returns a dict with keys: 'core_bits' (set), 'other_bits' (set), 'bit_info' (rdkit bitInfo)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    bit_info = {}

    core_bits = {}
    other_bits = {}

    for bit, instances in bit_info.items():
        # instances is list of (atom_idx, radius_env)
        instance_atom_sets = []
        for atom_idx, r in instances:
            inst_atoms = _instance_atoms_for_bit(mol, atom_idx, r)
            instance_atom_sets.append(inst_atoms)

        for inst_atoms in instance_atom_sets:
            is_in_core = inst_atoms <= core_atom_idxs
            if is_in_core:
                if bit not in core_bits:
                    core_bits[bit] = 1
                else:
                    core_bits[bit] += 1
            else:
                if bit not in other_bits:
                    other_bits[bit] = 1
                else:
                    other_bits[bit] += 1
        if bit in other_bits:
            other_bits[bit] /= len(instance_atom_sets)
        if bit in core_bits:
            core_bits[bit] /= len(instance_atom_sets)

    # bits that are not present in bit_info are zero bits; classify them as other
    # (they don't correspond to any substructure)
    present_bits = set(bit_info.keys())
    all_bits = set(range(n_bits))
    zero_bits = list(all_bits - present_bits)
    for b in zero_bits:
        other_bits[b] = 1

    return {"core_bits": core_bits, "other_bits": other_bits, "bit_info": bit_info}


def aggregate_shap_with_core(shap_expl: shap.Explanation, features_df: pd.DataFrame, core_bits: dict, other_bits: dict, sample_index: int = 0):
    """Aggregate SHAP values: sum core-bit SHAPs into a single 'core' feature
    and return a SHAP Explanation and Matplotlib figure showing the bar plot.

    Assumes that the order of features in `features_df` matches the order used
    to build `shap_expl` (this is the case when you explained `features_df`).
    """
    # extract raw shap values for the sample
    sample_vals = np.array(shap_expl[0].values)
    # if vals.ndim == 3:
    #     # tree explainer sometimes returns (n_samples, 1, n_features)
    #     vals = vals.squeeze(axis=1)
    # if vals.ndim == 1:
    #     vals = vals.reshape(1, -1)
    #
    # if sample_index >= vals.shape[0]:
    #     raise IndexError("sample_index out of range for provided shap_expl")
    #
    # sample_vals = vals[sample_index]

    # map features_df columns to indices
    feat_cols = list(features_df.columns)
    col_to_idx = {c: i for i, c in enumerate(feat_cols)}

    # compute core importance by summing SHAP values for core ecfp cols that exist
    core_importance = 0
    fully_in_core = []
    for b, frac in core_bits.items():
        ecfp_f = f"ecfp_{b}"
        if ecfp_f in col_to_idx:
            core_idx = col_to_idx[ecfp_f]
            core_importance += frac * sample_vals[core_idx]
            core_bits[b] = frac * sample_vals[core_idx]
        if frac == 1:
            fully_in_core.append(ecfp_f)
    print(f"Core: {core_importance}")
    print(core_bits)
    # core_cols_present = [f'ecfp_{i}' for i in core_bits if f'ecfp_{i}' in col_to_idx]
    # core_idx = [col_to_idx[c] for c in core_cols_present]
    # core_importance = float(np.sum(sample_vals[core_idx])) if core_idx else 0.0

    for b, frac in other_bits.items():
        ecfp_f = f"ecfp_{b}"
        if ecfp_f in col_to_idx:
            core_idx = col_to_idx[ecfp_f]
            sample_vals[core_idx] = frac * sample_vals[core_idx]

    # other columns are all feature columns except the core ecfp columns
    other_cols = [c for c in feat_cols if c not in fully_in_core]
    other_idx = [col_to_idx[c] for c in other_cols]

    other_vals = sample_vals[other_idx]

    # build new arrays: other_vals followed by core_importance as new feature
    new_vals = np.concatenate([other_vals, np.array([core_importance])], axis=0)

    new_feature_names = other_cols + ["core"]

    # prepare data array for Explanation (use original data for other cols, and 0 for core)
    data_row = features_df.iloc[sample_index][other_cols].to_numpy()
    data_row = np.concatenate([data_row, np.array([0.0])], axis=0)

    new_expl = shap.Explanation(values=new_vals, data=data_row, feature_names=new_feature_names)

    # plot using shap.bar
    fig = shap.plots.bar(shap_expl[0], show=False)

    return new_expl, fig


if __name__ == "__main__":
    smiles = pd.read_csv("notebooks/lab_smiles.csv")["smiles"].tolist()
    smarts, matches = compute_mcs_smarts(smiles)
    core_atoms = set(matches.get(0, ()))
    print(core_atoms)
    if core_atoms:
        part = partition_ecfp_bits_for_molecule(smiles[0], core_atoms, radius=2, n_bits=1024, strict=True)
        core_bits = part["core_bits"]
        print(core_bits)
