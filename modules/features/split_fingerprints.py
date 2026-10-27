"""Features computed separately on the main chain and side chains of a substrate."""

from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from skfp.fingerprints import ECFPFingerprint

from modules.features.feature_factory import FeatureFactory
from modules.features.substrate_split import TIER1_END_GROUPS, TIER2_END_GROUPS, Split, split_substrate

END_GROUP_NAMES = [name for name, _ in TIER1_END_GROUPS + TIER2_END_GROUPS]


@lru_cache(maxsize=None)
def cached_split(smiles: str) -> Split:
    """
    Split a substrate once per SMILES (main_ecfp, side_ecfp and split_descriptor all need it).
    :param smiles: SMILES string of the substrate.
    :return: Split with main-chain and side-chain atoms and fragment SMILES.
    """
    return split_substrate(smiles)


@FeatureFactory.register("main_ecfp")
def main_ecfp_fingerprint(size: int = 1024, radius: int = 2, count: bool = False) -> object:
    """
    Generate ECFP fingerprints of the main-chain fragment.
    :param size: size of the fingerprint.
    :param radius: radius of neighbors to consider.
    :param count: whether to include counts or to use binary values.
    :return: SplitECFP object.
    """
    return SplitECFP(part="main", size=size, radius=radius, count=count)


@FeatureFactory.register("side_ecfp")
def side_ecfp_fingerprint(size: int = 1024, radius: int = 2, count: bool = False) -> object:
    """
    Generate ECFP fingerprints of all side-chain fragments together.
    :param size: size of the fingerprint.
    :param radius: radius of neighbors to consider.
    :param count: whether to include counts or to use binary values.
    :return: SplitECFP object.
    """
    return SplitECFP(part="side", size=size, radius=radius, count=count)


@FeatureFactory.register("split_descriptor")
def split_descriptor() -> object:
    """
    Generate size/shape descriptors of the main-chain/side-chain split.
    :return: SplitDescriptor object.
    """
    return SplitDescriptor()


class SplitECFP:
    """
    ECFP of one part of the split. Fragments keep a dummy atom (*) at every cut bond,
    so attachment points are part of the environments. A substrate without side
    chains gets an all-zero side fingerprint.
    """

    def __init__(self, part: str, size: int = 1024, radius: int = 2, count: bool = False):
        if part not in ("main", "side"):
            raise ValueError(f"part must be 'main' or 'side', got '{part}'")
        self.part = part
        self.fp = ECFPFingerprint(fp_size=size, radius=radius, include_chirality=True, count=count)

    def fragment_smiles(self, smiles: str) -> str:
        """
        Get the SMILES of the selected part.
        :param smiles: SMILES string of the substrate.
        :return: main-chain SMILES, or side chains joined by '.' (empty if none).
        """
        split = cached_split(smiles)
        if self.part == "main":
            return split.main_chain_smiles
        return ".".join(split.side_chain_smiles)

    def fit_transform(self, smiles: list) -> np.ndarray:
        """
        ECFP of the selected part for every substrate.
        :param smiles: list of SMILES strings.
        :return: array with fingerprints.
        """
        mols = [Chem.MolFromSmiles(self.fragment_smiles(s)) for s in smiles]
        return self.fp.transform(mols)


class SplitDescriptor:
    """
    Descriptors of the split: part sizes, number of side chains, main-chain length
    (longest graph distance inside the main chain, which links the reactive ends and
    sets the pore size) and which end groups were found.
    """

    def __init__(self):
        self.numerical = [
            "split_n_main_atoms",
            "split_n_side_atoms",
            "split_side_fraction",
            "split_n_side_chains",
            "split_largest_side_chain",
            "split_n_ends",
            "split_main_chain_length",
        ]
        self.binary = [f"split_end_{name}" for name in END_GROUP_NAMES] + ["split_flagged"]

    @staticmethod
    def calculate(smiles: str) -> list:
        """
        Calculate split descriptors for one substrate.
        :param smiles: SMILES string of the substrate.
        :return: numerical descriptors followed by binary descriptors.
        """
        split = cached_split(smiles)
        mol = Chem.MolFromSmiles(smiles)
        main = sorted(split.main_atoms)
        dist_matrix = rdmolops.GetDistanceMatrix(mol)
        n_main, n_side = len(split.main_atoms), len(split.side_atoms)
        numerical = [
            n_main,
            n_side,
            n_side / (n_main + n_side),
            len(split.side_chains),
            max((len(c) for c in split.side_chains), default=0),
            split.n_ends,
            dist_matrix[np.ix_(main, main)].max(),
        ]
        flagged = bool({"large_side_chain", "no_ends_fallback_diameter"} & set(split.flags))
        binary = [int(name in split.end_types) for name in END_GROUP_NAMES] + [int(flagged)]
        return numerical + binary

    def fit_transform(self, smiles: list) -> np.ndarray:
        """
        Split descriptors for every substrate.
        :param smiles: list of SMILES strings.
        :return: array with descriptors.
        """
        return np.array([self.calculate(s) for s in smiles], dtype=float)

    def get_feature_names_out(self) -> list:
        """
        Get feature names for split descriptors.
        :return: list of feature names.
        """
        return self.numerical + self.binary

    def get_feature_types(self) -> dict:
        """
        Get feature types for split descriptors.
        :return: dictionary with feature types.
        """
        return {"categorical": [], "binary": self.binary, "numerical": self.numerical}
