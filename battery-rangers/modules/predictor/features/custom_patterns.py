"""Custom pattern descriptors module."""
import numpy as np
from rdkit import Chem

from modules.predictor.features.feature_factory import FeatureFactory


@FeatureFactory.register("custom_pattern")
def custom_pattern(count: bool = False) -> object:
    """
    Generate custom pattern descriptors.
    :param count: whether to count occurrences or just indicate presence.
    :return: CustomPatterns object.
    """
    return CustomPatterns(count=count)


class CustomPatterns:
    """
    Custom pattern descriptor class for calculating the presence of specific substructure patterns in molecules.
    Attributes:
        patterns (dict): A dictionary mapping pattern names to their corresponding SMARTS patterns.
    Methods:
        fit_transform(smiles: list) -> np.ndarray:
            Custom descriptor for patterns.
        get_feature_names_out() -> list:
            Get the names of the custom pattern features.
    """

    def __init__(self, count: bool = False):
        patterns = ["c2ccc1ccccc1c2", "c2ccc(c1ccccc1)cc2", "c1ccncc1", "c1ncncn1", "N#Cc1ccccc1", "C/C=C/C=C/C", "C=CC(=O)C=C"]
        self.patterns = {f"custom_pattern_{i}": pattern for i, pattern in enumerate(patterns)}
        self.count = count

    def fit_transform(self, smiles: list) -> np.ndarray:
        """
        Custom descriptor for patterns.
        :param smiles: list of SMILES strings.
        :param count: whether to count occurrences or just indicate presence.
        :return: array with custom pattern descriptors.
        """
        descriptors = np.zeros((len(smiles), len(self.patterns)))
        for i, s in enumerate(smiles):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            for j, pattern in enumerate(self.patterns.values()):
                patt_mol = Chem.MolFromSmarts(pattern)
                if mol.HasSubstructMatch(patt_mol):
                    if not self.count:
                        descriptors[i, j] = 1
                    else:
                        matches = mol.GetSubstructMatches(patt_mol)
                        descriptors[i, j] = len(matches)
        return descriptors

    def get_feature_names_out(self) -> list[str]:
        """
        Get the names of the custom pattern features.
        :return: list of feature names.
        """
        return list(self.patterns.keys())

    def get_feature_types(self) -> dict:
        """
        Get feature types for custom patterns.
        :return: list of feature types.
        """
        if self.count:
            return {"categorical": [], "binary": [], "numerical": list(self.patterns.keys())}
        return {"categorical": [], "binary": list(self.patterns.keys()), "numerical": []}
