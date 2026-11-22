"""Fit one configuration on all papers of a subset and save everything needed to explain it.

Output in results/final_model/<subset>__<method>__<model>__<table>/:
- model.joblib, scaler.joblib   fitted model and feature scaler (model input = scaler.transform(features))
- training_data.csv              one row per material: number, name, paper, j, the corrected target, and the unscaled features
- shap_values.csv                SHAP value of every feature for every material (in units of the corrected target)
- importance.csv                 mean |SHAP|, mean SHAP and direction per feature, sorted
- ecfp_bits.csv                  for ECFP features: which substructures set the bit, in how many materials
                                 ([r0] = one atom type, [r1]/[r2] = atom with neighbours up to 1/2 bonds away)

Sign convention: J is negative and a more negative J is a better catalyst. A positive SHAP value pushes the
prediction toward a higher (less negative) J, i.e. a worse catalyst.
"""

import argparse
import os
import re
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from skfp.fingerprints import ECFPFingerprint
from sklearn.preprocessing import StandardScaler

from experiments.electrocatalysis.common import SUBSETS, get_model, load_config, load_data, shap_values
from experiments.electrocatalysis.corrections import CORRECTIONS, fit_mixed
from modules.features.split_fingerprints import split_parts

ECFP_NAME = re.compile(r"^(?:(main|side)_)?ecfp_(\d+)$")

parser = argparse.ArgumentParser()
parser.add_argument("--subset", type=str, default=None)
parser.add_argument("--method", type=str, default=None)
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--table", type=str, default=None, help="Feature table, e.g. a1_full__ecfp")


def part_smiles(smiles: str, part: str) -> str:
    """Whole material, its main chains, or its side chains (as used by the ECFP features)."""
    if part == "full":
        return smiles
    main, sides = split_parts(smiles)
    return main if part == "main" else ".".join(sides)


def ecfp_environments(smiles_list: list, part: str, size: int = 1024, radius: int = 2) -> dict:
    """
    For every ECFP bit, count the atom environments (as SMILES fragments) that set it across the materials.
    Uses the same RDKit Morgan generator settings as skfp's ECFPFingerprint, and checks that the counts match.
    :return: {bit: Counter(environment SMILES -> number of materials)}.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=size, includeChirality=True)
    reference = ECFPFingerprint(fp_size=size, radius=radius, include_chirality=True, count=True)
    envs = {}
    for smiles in smiles_list:
        fragment = part_smiles(smiles, part)
        if not fragment:
            continue
        mol = Chem.MolFromSmiles(fragment)
        output = rdFingerprintGenerator.AdditionalOutput()
        output.AllocateBitInfoMap()
        counts = generator.GetCountFingerprintAsNumPy(mol, additionalOutput=output)
        assert np.array_equal(counts, reference.transform([mol])[0]), "RDKit and skfp ECFP bits differ"
        for bit, hits in output.GetBitInfoMap().items():
            seen = set()
            for atom, r in hits:
                if r == 0:
                    env_smiles = mol.GetAtomWithIdx(atom).GetSymbol()
                else:
                    bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom)
                    atoms = {a for b in bonds for a in (mol.GetBondWithIdx(b).GetBeginAtomIdx(), mol.GetBondWithIdx(b).GetEndAtomIdx())}
                    env_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms), bondsToUse=list(bonds), rootedAtAtom=atom)
                # radius 0 = a single atom type (element, degree, H count, charge, ring), not just the element
                seen.add(f"{env_smiles} [r{r}]")
            envs.setdefault(bit, Counter()).update(seen)
    return envs


if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config()
    cfg = config["final_model"] | {k: v for k, v in vars(args).items() if v is not None}
    work_dir = config["work_dir"]
    data = load_data(work_dir)
    rows = np.asarray(SUBSETS[cfg["subset"]](data))
    data = data.loc[rows].reset_index(drop=True)
    X = pd.read_csv(os.path.join(work_dir, "features", f"{cfg['table']}.csv")).loc[rows].reset_index(drop=True)
    j, papers = data["j"].to_numpy(), data["paper"].to_numpy()

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    model, shap_method = get_model(cfg["model"])
    if cfg["method"] == "mixed":
        model = fit_mixed(model, X_scaled, j, papers)
        target, keep = j - model.paper_intercepts_.reindex(papers).to_numpy(), np.ones(len(j), dtype=bool)
    else:
        target, keep = CORRECTIONS[cfg["method"]]().fit_transform(j, papers)
        model.fit(X_scaled[keep], target[keep])

    out_dir = os.path.join(work_dir, "final_model", f"{cfg['subset']}__{cfg['method']}__{cfg['model']}__{cfg['table']}")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    info = data[["number", "name", "paper", "type", "j", "smiles"]].assign(target=target, used_for_training=keep)
    pd.concat([info, X], axis=1).to_csv(os.path.join(out_dir, "training_data.csv"), index=False)

    values = shap_values(model, shap_method, X_scaled[keep], X_scaled[keep])
    if values is None:
        raise SystemExit(f"No SHAP explainer for model {cfg['model']}; use bayes, random_forest or xgboost")
    shap_df = pd.DataFrame(values, columns=X.columns)
    pd.concat([info[keep].reset_index(drop=True)[["number", "name", "paper"]], shap_df], axis=1).to_csv(os.path.join(out_dir, "shap_values.csv"), index=False)

    X_used = X[keep].reset_index(drop=True)
    direction = [np.corrcoef(X_used[c], shap_df[c])[0, 1] if X_used[c].std() > 0 and shap_df[c].std() > 0 else np.nan for c in X.columns]
    importance = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(values).mean(axis=0),
        "mean_shap": values.mean(axis=0),
        # > 0: more of this feature -> higher (less negative, worse) J; < 0: more of it -> better catalyst
        "direction": direction,
        "n_materials_with_feature": (X_used > 0).sum(axis=0).to_numpy(),
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(os.path.join(out_dir, "importance.csv"), index=False)

    bit_rows = []
    smiles_used = data.loc[keep, "smiles"].tolist()
    envs = {part: None for part in ["full", "main", "side"]}
    for feature in importance["feature"]:
        match = ECFP_NAME.match(feature)
        if not match:
            continue
        part, bit = match.group(1) or "full", int(match.group(2))
        if envs[part] is None:
            envs[part] = ecfp_environments(smiles_used, part)
        # most frequent first, ties alphabetical so the file is identical across runs
        top = sorted(envs[part].get(bit, Counter()).items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        bit_rows.append({"feature": feature, "part": part, "bit": bit, "substructures": " | ".join(f"{s} ({n})" for s, n in top)})
    if bit_rows:
        pd.DataFrame(bit_rows).to_csv(os.path.join(out_dir, "ecfp_bits.csv"), index=False)

    print(f"Saved to {out_dir}")
    print(importance.head(10).merge(pd.DataFrame(bit_rows or [{"feature": None}]), on="feature", how="left")[["feature", "mean_abs_shap", "direction", "substructures"]].to_string(index=False))
