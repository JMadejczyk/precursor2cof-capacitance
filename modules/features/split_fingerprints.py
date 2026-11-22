"""Features computed separately on the main chain and side chains of a substrate."""

from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from skfp.fingerprints import ECFPFingerprint

from modules.features.custom_descriptor import descriptor
from modules.features.feature_factory import FeatureFactory
from modules.features.fingerprints import estate_fingerprint, functional_groups_fingerprint
from modules.features.substrate_split import TIER1_END_GROUPS, TIER2_END_GROUPS, Split, split_substrate
from modules.features.utils import fingerprint_feature_types

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


@FeatureFactory.register("main_estate")
def main_estate_fingerprint(variant: str = "sum") -> object:
    """
    Generate estate fingerprints of the main chain.
    :param variant: type of estate fingerprint.
    :return: SplitFeatures object.
    """
    return SplitFeatures("main", "estate", estate_fingerprint(variant), variant=variant)


@FeatureFactory.register("side_estate")
def side_estate_fingerprint(variant: str = "sum") -> object:
    """
    Generate estate fingerprints of all side chains together.
    :param variant: type of estate fingerprint.
    :return: SplitFeatures object.
    """
    return SplitFeatures("side", "estate", estate_fingerprint(variant), variant=variant)


@FeatureFactory.register("main_functional_groups")
def main_functional_groups_fingerprint(count: bool = False) -> object:
    """
    Generate functional groups fingerprints of the main chain.
    :param count: whether to include counts or to use binary values.
    :return: SplitFeatures object.
    """
    return SplitFeatures("main", "functional_groups", functional_groups_fingerprint(count), count=count)


@FeatureFactory.register("side_functional_groups")
def side_functional_groups_fingerprint(count: bool = False) -> object:
    """
    Generate functional groups fingerprints of all side chains together.
    :param count: whether to include counts or to use binary values.
    :return: SplitFeatures object.
    """
    return SplitFeatures("side", "functional_groups", functional_groups_fingerprint(count), count=count)


@FeatureFactory.register("main_descriptor")
def main_descriptor(flatness: bool = False, symmetry: bool = False) -> object:
    """
    Generate custom molecular descriptors of the main chain.
    :return: SplitFeatures object.
    """
    return SplitFeatures("main", "descriptor", part_descriptor(flatness, symmetry))


@FeatureFactory.register("side_descriptor")
def side_descriptor(flatness: bool = False, symmetry: bool = False) -> object:
    """
    Generate custom molecular descriptors of all side chains together.
    :return: SplitFeatures object.
    """
    return SplitFeatures("side", "descriptor", part_descriptor(flatness, symmetry))


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


def part_descriptor(flatness: bool = False, symmetry: bool = False) -> object:
    """
    Custom descriptors for a part that can consist of several separate side chains.
    Distances between separate pieces are undefined, so diameters are the largest within one piece.
    :return: CustomDescriptor object.
    """
    custom_descriptor = descriptor(flatness, symmetry)
    for name in ("diameter", "diameter_filtered"):
        diameter = custom_descriptor.desc[name]
        custom_descriptor.desc[name] = lambda mol, diameter=diameter: max(diameter(piece) for piece in Chem.GetMolFrags(mol, asMols=True))
    return custom_descriptor


def capped_fragment_smiles(smiles: str, part: str) -> str:
    """
    Get one part of the split as a molecule of its own: every cut point (*) is removed and the
    free valences are filled with hydrogens, also for cut double bonds (e.g. a ring C=O in the side chains).
    :param smiles: SMILES string of the substrate.
    :param part: 'main' or 'side'.
    :return: SMILES of the part (side chains joined by '.', empty if there are none).
    """
    split = cached_split(smiles)
    fragment = split.main_chain_smiles if part == "main" else ".".join(split.side_chain_smiles)
    if not fragment:
        return ""
    mol = Chem.MolFromSmiles(fragment)
    # in the Kekule form a removed cut point leaves plain free valences instead of a broken aromatic ring
    Chem.Kekulize(mol, clearAromaticFlags=True)
    mol = Chem.RWMol(mol)
    cut_points = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    for atom in sorted(cut_points, key=lambda a: a.GetIdx(), reverse=True):
        for neighbor in atom.GetNeighbors():
            neighbor.SetNoImplicit(False)
        mol.RemoveAtom(atom.GetIdx())
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)


class SplitFeatures:
    """
    Any whole-molecule feature computed on one part of the split instead. Cut points are
    capped with hydrogens, so descriptors and atom-type features see an ordinary molecule.
    A substrate without side chains gets an all-zero side row.
    """

    def __init__(self, part: str, name: str, features: object, **kwargs):
        if part not in ("main", "side"):
            raise ValueError(f"part must be 'main' or 'side', got '{part}'")
        self.part = part
        self.name = name
        self.features = features
        self.kwargs = kwargs
        self.num_features = 0

    def fit_transform(self, smiles: list) -> np.ndarray:
        """
        Features of the selected part for every substrate.
        :param smiles: list of SMILES strings.
        :return: array with features.
        """
        fragments = [capped_fragment_smiles(s, self.part) for s in smiles]
        present = [i for i, fragment in enumerate(fragments) if fragment]
        values = np.asarray(self.features.fit_transform([fragments[i] for i in present]), dtype=float)
        self.num_features = values.shape[1]
        features = np.zeros((len(smiles), self.num_features))
        features[present] = values
        return features

    def get_feature_names_out(self) -> list:
        """
        Get feature names: the whole-molecule names prefixed with the part.
        :return: list of feature names.
        """
        if hasattr(self.features, "get_feature_types"):
            names = self.features.get_feature_names_out()
        else:
            names = [f"{self.name}_{i}" for i in range(self.num_features)]
        return [f"{self.part}_{name}" for name in names]

    def get_feature_types(self) -> dict:
        """
        Get feature types, the same as for the whole-molecule feature.
        :return: dictionary with feature types.
        """
        if hasattr(self.features, "get_feature_types"):
            return {f_type: [f"{self.part}_{name}" for name in names] for f_type, names in self.features.get_feature_types().items()}
        return fingerprint_feature_types(self.name, self.get_feature_names_out(), **self.kwargs)


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
