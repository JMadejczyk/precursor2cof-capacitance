"""Pore size calculation functions."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor


def calculate_repeating_unit_length(smiles: str, random_seed: int = 42) -> float | None:
    """
    Calculates the length of the repeating unit for a given molecule as the largest distance between any two atoms.

    Args:
        smiles: SMILES string of the molecule

    Returns:
        The length of the repeating unit in Angstroms.
    """

    try:
        molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    except ValueError as e:
        raise ValueError("Invalid SMILES string.") from e
    molecule = Chem.AddHs(molecule)
    a = AllChem.EmbedMolecule(molecule, randomSeed=random_seed, maxAttempts=500)
    if a < 0:
        a = AllChem.EmbedMolecule(molecule, randomSeed=random_seed, maxAttempts=500, useRandomCoords=True)
        if a < 0:
            return None
    AllChem.MMFFOptimizeMolecule(molecule)

    conf = molecule.GetConformer()
    max_distance = 0

    for i in range(molecule.GetNumAtoms()):
        for j in range(i + 1, molecule.GetNumAtoms()):
            pos_i = np.array(conf.GetAtomPosition(i))
            pos_j = np.array(conf.GetAtomPosition(j))

            # Calculate the distance between the two points
            distance = np.linalg.norm(pos_i - pos_j)

            max_distance = max(max_distance, distance)

    return max_distance


def calculate_hexagonal_pore_diameter(smiles: str) -> float | None:
    """
    Calculates the diameter of a hexagonal pore given the length of the unit.

    Args:
        smiles: SMILES string of the molecule

    Returns:
        pore_diameter_nm: The diameter of the pore in nanometers.
    """

    unit_length_pm = calculate_repeating_unit_length(smiles)
    if unit_length_pm is None:
        return None
    perimeter = unit_length_pm * 6

    pore_diameter = perimeter / np.pi

    pore_diameter_nm = pore_diameter / 10  # Convert to nanometers

    return pore_diameter_nm


def sanity_check_with_experts() -> None:
    """Whatever the hell is that."""
    smiles_terephtalonitrile = "N#CC1=CC=C(C#N)C=C1"
    smiles_2cnpp = "CC(C)C(C=C1)=CC=C1N2C(C=C(C3=CC=C(C#N)C=C3)N4C5=CC=C(C(C)C)C=C5)=C4C=C2C6=CC=C(C#N)C=C6"
    pore_diameter_terephtalonitrile = calculate_hexagonal_pore_diameter(smiles_terephtalonitrile)
    pore_diameter_2cnpp = calculate_hexagonal_pore_diameter(smiles_2cnpp)
    print(f"Pore diameter for Terephtalonitrile {pore_diameter_terephtalonitrile}")
    print(f"Pore diameter for 2CNPP {pore_diameter_2cnpp}")


# TODO: Delete this function after the target one is implemented.
def get_furthest_atom_id(mol, atom_indices, atom_name=None):
    """Find atom index furthest from the molecular center."""
    conf = mol.GetConformer(0)
    if atom_name:
        atom_ids_with_symbol = [a for a in atom_indices if mol.GetAtomWithIdx(a).GetSymbol() == atom_name]
        positions = np.array([conf.GetAtomPosition(idx) for idx in atom_indices if mol.GetAtomWithIdx(idx).GetSymbol() == atom_name])
        # Compute distances from center and find the atom furthest away
        distances = np.linalg.norm(positions, axis=1)
        max_dist_index = np.argmax(distances)

        return atom_ids_with_symbol[max_dist_index]
    positions = np.array([conf.GetAtomPosition(idx) for idx in atom_indices])
    distances = np.linalg.norm(positions, axis=1)
    max_dist_index = np.argmax(distances)

    return atom_indices[max_dist_index]


# TODO: Delete this function after the target one is implemented.
def estimate_pore_size(sml: str) -> float:
    """Temporary method until a target one is implemented"""
    if isinstance(sml, str):
        molecule = Chem.MolFromSmiles(sml, sanitize=True)
    elif isinstance(sml, Chem.Mol):
        molecule = sml
    else:
        return 1000.0
    molecule = Chem.AddHs(molecule)
    rdDepictor.Compute2DCoords(molecule, sampleSeed=42)

    conf = molecule.GetConformer(0)

    match = list(range(0, molecule.GetNumAtoms()))
    furthest_atom_idx = get_furthest_atom_id(molecule, match)

    pos_j = np.array(conf.GetAtomPosition(furthest_atom_idx))

    # Calculate the distance between the two points
    distance = np.linalg.norm(np.array([0, 0, 0]) - pos_j)
    distance = distance * 6 / np.pi / 10

    return distance


if __name__ == "__main__":
    sanity_check_with_experts()
