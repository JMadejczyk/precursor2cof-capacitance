import itertools
from functools import partial

import numpy as np
import skfp
from predictor.features.csm_runner import CSMRunner
from predictor.features.flatness import get_flatness_mol
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdMolDescriptors, rdmolops
from tqdm import tqdm

from modules.features.feature_factory import FeatureFactory


@FeatureFactory.register("descriptor")
def descriptor(flatness: bool = False, symmetry: bool = False) -> object:
    """
    Generate custom molecular descriptors.
    :return: CustomDescriptor object.
    """
    return CustomDescriptor(flatness=flatness, symmetry=symmetry)


class CustomDescriptor:
    """
    Custom descriptor class for calculating various molecular descriptors.
    Attributes:
        csm_runner (CSMRunner): An instance of the CSMRunner class for symmetry analysis.
        desc (dict): A dictionary mapping descriptor names to their corresponding functions.
    Methods:
        calculate_percentage(mol: Chem.Mol, idx: int) -> float:
            Calculate the percentage of a specific atom in the molecule.
        calculate_symmetry(mol: Chem.Mol, point_group: list) -> float:
            Calculate symmetry descriptor for the molecule.
        fit_transform(smiles: list) -> np.ndarray:
            Custom descriptor for fingerprints.
        get_feature_names_out() -> list:
            Get feature names for custom descriptors.
    """

    def __init__(self, flatness: bool = False, symmetry: bool = False):
        self.csm_runner = CSMRunner()

        self.desc = {
            # "radius": skfp.descriptors.radius,
            "diameter": skfp.descriptors.diameter,
            "diameter_filtered": self.calculate_diameter_endings,
            "num_heteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
            "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds,
            # "num_h_acceptors": rdMolDescriptors.CalcNumLipinskiHBA,
            # "num_h_donors": rdMolDescriptors.CalcNumLipinskiHBD,
            # "tpsa": rdMolDescriptors.CalcTPSA,
            "mol_wt": rdMolDescriptors.CalcExactMolWt,
            "o%": partial(self.calculate_percentage, idx=8),
            "n%": partial(self.calculate_percentage, idx=7),
            "c%": partial(self.calculate_percentage, idx=6),
        }
        if flatness:
            self.desc.update(
                {
                    "flatness": partial(get_flatness_mol, max_attempts=5000),
                }
            )
        if symmetry:
            self.desc.update(
                {
                    "symmetry": partial(self.calculate_symmetry, point_group=["c2", "c3", "c4"]),
                }
            )

    @staticmethod
    def calculate_percentage(mol: Chem.Mol, idx: int) -> float:
        """
        Calculate the percentage of oxygen atoms in the molecule.
        :param idx: atomic number of the atom to calculate percentage for.
        :param mol: RDKit molecule object.
        :return: percentage of oxygen atoms.
        """
        num_oxygen = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == idx)
        num_atoms = mol.GetNumAtoms()
        return num_oxygen / num_atoms

    def calculate_symmetry(self, mol: Chem.Mol, point_group: list) -> float:
        """
        Calculate symmetry descriptor for the molecule.
        :param mol: RDKit molecule object.
        :param point_group: list of point groups to evaluate.
        :return: symmetry score.
        """
        csm_result = self.csm_runner.analyze_molecule(mol, point_groups=point_group, exact=False)
        return csm_result.lowest_csm[1] / mol.GetNumAtoms() if csm_result and csm_result.lowest_csm else 0.0

    def calculate_diameter_endings(self, mol: Chem.Mol) -> int:
        """
        Finds the maximum distance between atoms of the same pattern,
        highlights the path, and returns an SVG image.
        """
        regular_diameter = skfp.descriptors.diameter(mol)

        patterns = [
            "O=*",
            "C#N",
            "Cl-*",
            "Br-*",
            "[NH2]",
            "[HO]",
        ]
        if mol is None:
            return 0

        rdDepictor.Compute2DCoords(mol)
        dist_matrix = rdmolops.GetDistanceMatrix(mol)

        max_dist = -1

        for pattern_str in patterns:
            pattern = Chem.MolFromSmarts(pattern_str)
            matches = mol.GetSubstructMatches(pattern)
            if len(matches) < 2:
                continue

            for m1, m2 in itertools.combinations(matches, 2):
                heavy1 = [i for i in m1 if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
                heavy2 = [i for i in m2 if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
                for i in heavy1:
                    for j in heavy2:
                        d = dist_matrix[i][j]
                        if d > max_dist:
                            max_dist = d

        if max_dist < 0:
            return regular_diameter
        return max_dist

    def fit_transform(self, smiles: list) -> np.ndarray:
        """
        Custom descriptor for fingerprints.
        :param smiles: list of SMILES strings.
        :return: array with custom descriptors.
        """
        descriptors = np.zeros((len(smiles), len(self.desc.keys())))
        for i, s in tqdm(enumerate(smiles)):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            for j, (name, func) in enumerate(self.desc.items()):
                moll = Chem.MolFromSmiles(s)
                descriptors[i, j] = func(moll)
        return descriptors

    def get_feature_names_out(self) -> list:
        """
        Get feature names for custom descriptors.
        :return: list of feature names.
        """
        return list(self.desc.keys())

    def get_feature_types(self) -> dict:
        """
        Get feature types for custom descriptors.
        :return: list of feature types.
        """
        return {"categorical": [], "binary": [], "numerical": list(self.desc.keys())}
