"""Split a COF substrate into a main chain and side chains.

Idea (from meeting 1): a substrate has a main chain that links its reactive ends
(and sets the pore size) and side chains hanging off it.

Algorithm:
1. Find reactive end groups. Tier-1 linkage groups (nitrile, aldehyde, primary amine,
   boronic acid, terminal alkyne, stannane) are used first. Tier-2 groups (halides,
   phenol OH) are added only when tier 1 gives fewer than 2 ends, because Br/F/OH
   are usually substituents.
2. Main chain = union of shortest graph paths between every pair of end anchors,
   plus the end-group atoms themselves.
3. Ring expansion: every ring that shares >= 2 atoms with the main chain is added
   whole; repeat until nothing changes, so fused ring systems on the path are kept.
4. Optionally add terminal atoms double-bonded to the main chain (e.g. ring C=O),
   since they are part of the conjugated core rather than a pendant group.
5. Side chains = connected components of the remaining atoms.

If fewer than 2 ends are found, the graph diameter path is used instead (flagged).
Splits with a side chain larger than half the main chain are flagged "large_side_chain"
for manual review, because that often means an end group was not recognised.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import rdmolops

# (name, SMARTS); the first atom of each match is the anchor bonded to the rest of the molecule.
TIER1_END_GROUPS = [
    ("nitrile", "[CX2]#[NX1]"),
    ("aldehyde", "[CX3H1;$(C[#6])]=[OX1]"),
    ("amine", "[NX3;H2;!$(NC=[O,S,N])]"),
    ("boronic_acid", "[BX3]([OX2H])[OX2H]"),
    ("alkyne", "[CX2]#[CX2H1]"),
    ("stannane", "[Sn](-[CH3])(-[CH3])-[CH3]"),
]
TIER2_END_GROUPS = [
    ("bromide", "[Br]"),
    ("chloride", "[Cl]"),
    ("iodide", "[I]"),
    ("hydroxyl", "[OX2H][#6]"),
]
HALIDES = {"bromide", "chloride", "iodide"}
# Cross-coupling groups: their partner end is a halide, so halides count as ends too.
COUPLING_GROUPS = {"boronic_acid", "alkyne", "stannane"}
_PATTERNS = {
    name: Chem.MolFromSmarts(s) for name, s in TIER1_END_GROUPS + TIER2_END_GROUPS
}


@dataclass
class Split:
    smiles: str
    end_types: list[str]
    n_ends: int
    main_atoms: set[int]
    end_atoms: set[int]
    side_chains: list[set[int]]
    main_chain_smiles: str
    side_chain_smiles: list[str]
    flags: list[str] = field(default_factory=list)

    @property
    def side_atoms(self) -> set[int]:
        return set().union(*self.side_chains) if self.side_chains else set()


def find_end_groups(
    mol: Chem.Mol,
) -> tuple[list[tuple[str, tuple[int, ...]]], list[str]]:
    """Return [(group_name, match_atoms)] and flags.

    Tier 1 alone if it gives >= 2 ends (plus halides when a cross-coupling group is
    present); otherwise tier 1 + tier 2 together (e.g. one NH2 + two Br in a
    Buchwald monomer); otherwise no ends.
    """
    tier1 = [
        (name, m)
        for name, _ in TIER1_END_GROUPS
        for m in mol.GetSubstructMatches(_PATTERNS[name])
    ]
    ends, flags = tier1, []
    if {name for name, _ in tier1} & COUPLING_GROUPS:
        ends = tier1 + [
            (name, m)
            for name, _ in TIER2_END_GROUPS
            if name in HALIDES
            for m in mol.GetSubstructMatches(_PATTERNS[name])
        ]
    if len(tier1) < 2:
        ends = tier1 + [
            (name, m)
            for name, _ in TIER2_END_GROUPS
            for m in mol.GetSubstructMatches(_PATTERNS[name])
        ]
        flags.append("tier2_ends")
    if len(ends) < 2:
        return [], ["no_ends_fallback_diameter"]
    if len({name for name, _ in ends}) > 1:
        flags.append("mixed_end_types")
    return ends, flags


def _diameter_endpoints(mol: Chem.Mol) -> tuple[int, int]:
    dm = rdmolops.GetDistanceMatrix(mol)
    i, j = divmod(int(dm.argmax()), dm.shape[0])
    return i, j


def _expand_rings(mol: Chem.Mol, atoms: set[int]) -> set[int]:
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    atoms = set(atoms)
    changed = True
    while changed:
        changed = False
        for ring in rings:
            if not ring <= atoms and len(ring & atoms) >= 2:
                atoms |= ring
                changed = True
    return atoms


def _add_exocyclic_double_bonded(mol: Chem.Mol, atoms: set[int]) -> set[int]:
    extra = set()
    for idx in atoms:
        for bond in mol.GetAtomWithIdx(idx).GetBonds():
            nbr = bond.GetOtherAtom(mol.GetAtomWithIdx(idx))
            if (
                nbr.GetIdx() not in atoms
                and nbr.GetDegree() == 1
                and bond.GetBondType() == Chem.BondType.DOUBLE
            ):
                extra.add(nbr.GetIdx())
    return atoms | extra


def _components(mol: Chem.Mol, atoms: set[int]) -> list[set[int]]:
    remaining, comps = set(atoms), []
    while remaining:
        stack, comp = [remaining.pop()], set()
        while stack:
            a = stack.pop()
            comp.add(a)
            for nbr in mol.GetAtomWithIdx(a).GetNeighbors():
                if nbr.GetIdx() in remaining:
                    remaining.discard(nbr.GetIdx())
                    stack.append(nbr.GetIdx())
        comps.append(comp)
    return sorted(comps, key=min)


def _fragment_smiles(
    mol: Chem.Mol, main_atoms: set[int], side_chains: list[set[int]]
) -> tuple[str, list[str]]:
    """Cut main/side bonds, put a '*' at each cut, return main-chain and side-chain SMILES."""
    cut = [
        b.GetIdx()
        for b in mol.GetBonds()
        if (b.GetBeginAtomIdx() in main_atoms) != (b.GetEndAtomIdx() in main_atoms)
    ]
    if not cut:
        return Chem.MolToSmiles(mol), []
    frag = Chem.FragmentOnBonds(
        mol, cut, addDummies=True, dummyLabels=[(0, 0)] * len(cut)
    )
    frag_atom_sets = Chem.GetMolFrags(frag, asMols=False, sanitizeFrags=False)
    frag_mols = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)
    main_smi, side_smis = "", {}
    for atom_ids, fmol in zip(frag_atom_sets, frag_mols):
        original = {a for a in atom_ids if a < mol.GetNumAtoms()}
        smi = Chem.MolToSmiles(fmol)
        if original & main_atoms:
            main_smi = smi
        else:
            side_smis[min(original)] = smi
    return main_smi, [side_smis[k] for k in sorted(side_smis)]


def split_substrate(smiles: str, keep_exocyclic_double_bonds: bool = True) -> Split:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    ends, flags = find_end_groups(mol)
    end_atoms = {a for _, m in ends for a in m}
    if ends:
        anchors = [m[0] for _, m in ends]
    else:
        anchors = list(_diameter_endpoints(mol))

    main = set(end_atoms) | set(anchors)
    for a, b in itertools.combinations(anchors, 2):
        if a != b:
            main |= set(rdmolops.GetShortestPath(mol, a, b))
    main = _expand_rings(mol, main)
    if keep_exocyclic_double_bonds:
        main = _add_exocyclic_double_bonded(mol, main)

    side_chains = _components(mol, set(range(mol.GetNumAtoms())) - main)
    main_smi, side_smis = _fragment_smiles(mol, main, side_chains)
    if not side_chains:
        flags.append("no_side_chains")
    elif max(len(c) for c in side_chains) > len(main) / 2:
        flags.append("large_side_chain")

    return Split(
        smiles=smiles,
        end_types=sorted({name for name, _ in ends}),
        n_ends=len(ends),
        main_atoms=main,
        end_atoms=end_atoms,
        side_chains=side_chains,
        main_chain_smiles=main_smi,
        side_chain_smiles=side_smis,
        flags=flags,
    )


if __name__ == "__main__":
    for s in [
        "N#Cc1ccc(C#N)cc1",
        "CC(C)c1ccc(-n2c(-c3ccc(C#N)cc3)cc3c2cc(-c2ccc(C#N)cc2)n3-c2ccc(C(C)C)cc2)cc1",
        "N#Cc1c(F)c(F)c(C#N)c(F)c1F",
    ]:
        r = split_substrate(s)
        print(
            r.end_types,
            r.n_ends,
            "| main:",
            r.main_chain_smiles,
            "| side:",
            r.side_chain_smiles,
            r.flags,
        )
