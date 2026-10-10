"""Module for generating various molecular fingerprints using skfp."""
from typing import Literal

import skfp
from skfp.fingerprints import (
    BCUT2DFingerprint,
    ECFPFingerprint,
    EStateFingerprint,
    FunctionalGroupsFingerprint,
    MACCSFingerprint,
    PhysiochemicalPropertiesFingerprint,
)

from modules.predictor.features.feature_factory import FeatureFactory


# connectivity
@FeatureFactory.register("ecfp")
def ecfp_fingerprint(size: int = 1024, radius: int = 2, count: bool = False) -> skfp.bases.BaseFingerprintTransformer:
    """
    Generate ECFP fingerprints.
    :param size: size of the fingerprint.
    :param radius: radius of neighbors to consider.
    :param count: whether to include counts or to use binary values.
    :return: ECFP fingerprint.
    """
    return ECFPFingerprint(fp_size=size, radius=radius, include_chirality=True, count=count)


@FeatureFactory.register("maccs")
def maccs_fingerprint(count: bool = False) -> skfp.bases.BaseFingerprintTransformer:
    """
    Generate MACCS fingerprints.
    :param count: whether to include counts or to use binary values.
    :return: Maccs fingerprint.
    """
    return MACCSFingerprint(count=count)


@FeatureFactory.register("functional_groups")
def functional_groups_fingerprint(count: bool = False) -> skfp.bases.BaseFingerprintTransformer:
    """
    Generate functional groups fingerprints.
    :param count: whether to include counts or to use binary values.
    :return: functional groups fingerprint.
    """
    return FunctionalGroupsFingerprint(count=count)


@FeatureFactory.register("atom_pair")
def atom_pair_fingerprint(size: int = 1024, count: bool = False) -> skfp.bases.BaseFingerprintTransformer:
    """
    Generate atom pair fingerprints.
    :param size: size of the fingerprint.
    :param count: whether to include counts or to use binary values.
    :return: atom pair fingerprint.
    """
    return skfp.fingerprints.AtomPairFingerprint(fp_size=size, count=count, include_chirality=True)


# properties
@FeatureFactory.register("estate")
def estate_fingerprint(variant: Literal["bit", "count", "sum"] = "sum") -> skfp.bases.BaseFingerprintTransformer:
    """
    Generate estate fingerprints.
    :param variant: type of estate fingerprint.
    :return: estate fingerprint.
    """
    return EStateFingerprint(variant=variant)


@FeatureFactory.register("physiochemical")
def physiochemical_fingerprint(size: int = 1024, count: bool = False) -> skfp.bases.BaseFingerprintTransformer:
    """
    Generate physiochemical fingerprints.
    :param size: size of the fingerprint.
    :param count: whether to include counts or to use binary values.
    :return: physiochemical fingerprint.
    """
    return PhysiochemicalPropertiesFingerprint(fp_size=size, count=count)


@FeatureFactory.register("bcut")
def bcut_fingerprint() -> skfp.bases.BaseFingerprintTransformer:
    """
    Generate BCUT2D fingerprints.
    :return: BCUT fingerprint.
    """
    return BCUT2DFingerprint()
