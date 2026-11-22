"""Split a COF substrate into a main chain and side chains.

Idea (from meeting 1): a substrate has a main chain that links its reactive ends
(and sets the pore size) and side chains hanging off it.

Convention: the main chain is every arm that ends in a reactive group; side chains are
pendant groups that never become network bonds. That includes non-reacting =O, =S and OH
(ring C=O, imide C=O, sulfone S=O, Tp-type OH), in line with IUPAC's "pendant group".
Atoms that react (aldehyde O, B(OH)2, catechol OH, 1,2-dione O, ...) are end groups.

Algorithm:
1. Find reactive end groups. Tier-1 linkage groups (nitrile, aldehyde, primary amine,
   boronic acid, terminal alkyne, stannane, carboxylic acid, imide N-H, formamide,
   active CH2, SH next to NH2, Zincke N-(2,4-dinitrophenyl)pyridinium, 1,2-dione) are
   used first. Tier-2 groups (halides, phenol OH) are added only when
   tier 1 gives fewer than 2 ends, because Br/F/OH are usually substituents. Exceptions:
   phenol OH counts in benzoxazine monomers, aryl Br counts when all amines sit on one
   ring, and manual_ends.py overrides the choice for hand-checked substrates. A manual
   entry can also name extra anchors: atoms the main chain must reach although no end
   group marks them.
2. Main chain = union of shortest graph paths between every pair of end anchors,
   plus the end-group atoms themselves.
3. Ring expansion: every ring that shares >= 2 atoms with the main chain is added
   whole; repeat until nothing changes, so fused ring systems on the path are kept.
4. Only with keep_exocyclic_double_bonds=True (off by default): add terminal atoms
   double-bonded to the main chain (e.g. ring C=O).
5. If all ends sit on one ring and the leftover part is larger than half the main chain,
   extend the chain to the atoms of that part farthest from it (ties included), then
   repeat steps 3-4. Flagged "ends_on_one_ring".
6. Side chains = connected components of the remaining atoms.

If fewer than 2 ends are found, the anchors are the aryl CH3 groups (the experts' cap for a
linkage position) plus every endpoint of a longest graph path (flagged).
Splits with a side chain larger than half the main chain are flagged "large_side_chain"
for manual review, because that often means an end group was not recognised.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import rdmolops

from manual_ends import MANUAL_ENDS

# (name, SMARTS); the first atom of each match is the anchor bonded to the rest of the molecule.
TIER1_END_GROUPS = [
    ("nitrile", "[CX2]#[NX1]"),
    ("aldehyde", "[CX3H1;$(C[#6])]=[OX1]"),
    ("amine", "[NX3;H2;!$(NC=[O,S,N])]"),
    ("boronic_acid", "[BX3]([OX2H])[OX2H]"),
    ("alkyne", "[CX2]#[CX2H1]"),
    ("stannane", "[Sn](-[CH3])(-[CH3])-[CH3]"),
    ("carboxylic_acid", "[CX3](=O)[OX2H]"),
    ("imide_nh", "[#7;R;D2;$([#7]([#6]=[OX1])[#6]=[OX1])]"),
    ("formamide", "[CX3H1;$(C[#7])]=[OX1]"),
    ("active_ch2", "[CH2;$(C(C=O)C=O)]"),
    ("amino_thiol", "[SH;$(Sc:c[NX3H2])]"),
    # Zincke salt: an amine replaces the N-(2,4-dinitrophenyl) group, so the whole group is the end.
    ("zincke_pyridinium", "[n+]-c1ccc(cc1[N+](=O)[O-])[N+](=O)[O-]"),
    ("o_dione", "[#6;R](=O)[#6;R]=O"),
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
_BENZOXAZINE = Chem.MolFromSmarts("N1COc2ccccc2C1")
_ARYL_BROMIDE = Chem.MolFromSmarts("[Br]c")
_ARYL_METHYL = Chem.MolFromSmarts("[CH3]c")
_MANUAL = {
    Chem.MolToSmiles(Chem.MolFromSmiles(e["smiles"])): e for e in MANUAL_ENDS
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


def _on_one_ring(mol: Chem.Mol, anchors: list[int]) -> bool:
    """True if every anchor is in, or directly bonded to, the same ring."""
    for ring in mol.GetRingInfo().AtomRings():
        near = set(ring)
        for a in ring:
            near |= {n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors()}
        if set(anchors) <= near:
            return True
    return False


def _add_ends(
    ends: list[tuple[str, tuple[int, ...]]],
    name: str,
    matches: tuple[tuple[int, ...], ...],
) -> list[tuple[str, tuple[int, ...]]]:
    seen = {m for _, m in ends}
    return ends + [(name, m) for m in matches if m not in seen]


def _manual_entry(mol: Chem.Mol) -> dict | None:
    return _MANUAL.get(Chem.MolToSmiles(mol))


def _manual_atoms(mol: Chem.Mol, entry: dict, key: str) -> list[int]:
    """Atom indices of `mol` for the entry's `key` (indices given in the entry's SMILES order)."""
    match = mol.GetSubstructMatch(Chem.MolFromSmiles(entry["smiles"]))
    return [match[i] for i in entry.get(key, ())]


def _keep_manual_ends(
    mol: Chem.Mol, ends: list[tuple[str, tuple[int, ...]]], entry: dict
) -> list[tuple[str, tuple[int, ...]]]:
    """Ends listed in the entry's "anchors"; every end if the entry has no "anchors"."""
    if "anchors" not in entry:
        return ends
    anchors = set(_manual_atoms(mol, entry, "anchors"))
    return [e for e in ends if e[1][0] in anchors]


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
    if mol.HasSubstructMatch(_BENZOXAZINE):
        ends = _add_ends(
            ends, "hydroxyl", mol.GetSubstructMatches(_PATTERNS["hydroxyl"])
        )
    if (
        ends
        and {name for name, _ in ends} == {"amine"}
        and mol.HasSubstructMatch(_ARYL_BROMIDE)
        and _on_one_ring(mol, [m[0] for _, m in ends])
    ):
        ends = _add_ends(
            ends, "bromide", mol.GetSubstructMatches(_PATTERNS["bromide"])
        )
    entry = _manual_entry(mol)
    if entry:
        ends = _keep_manual_ends(mol, ends, entry)
        if not ends:
            flags = []
        flags.append("manual_ends")
        # Extra anchors count towards the two points a chain needs.
        if len(ends) + len(entry.get("extra_anchors", ())) < 2:
            raise ValueError(f"Manual entry leaves fewer than 2 anchors: {entry['smiles']}")
    elif len(ends) < 2:
        return [], ["no_ends_fallback_diameter"]
    if len({name for name, _ in ends}) > 1:
        flags.append("mixed_end_types")
    return ends, flags


def _fallback_anchors(mol: Chem.Mol) -> list[int]:
    """Aryl CH3 (the experts' cap for a linkage position) plus every atom that ends a longest path.

    Taking all tied endpoints keeps symmetry-equivalent arms in the same class.
    """
    dm = rdmolops.GetDistanceMatrix(mol)
    ends = {int(a) for a in (dm == dm.max()).nonzero()[0]}
    ends |= {m[0] for m in mol.GetSubstructMatches(_ARYL_METHYL)}
    return sorted(ends)


def _farthest_tied(dm, from_atoms: set[int], candidates: set[int]) -> list[int]:
    """Atoms of `candidates` tied for the largest distance to the nearest atom of `from_atoms`."""
    dist = {a: min(dm[a][x] for x in from_atoms) for a in candidates}
    far = max(dist.values())
    return sorted(a for a, d in dist.items() if d == far)


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


def split_substrate(smiles: str, keep_exocyclic_double_bonds: bool = False) -> Split:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    ends, flags = find_end_groups(mol)
    end_atoms = {a for _, m in ends for a in m}
    entry = _manual_entry(mol)
    extra_anchors = _manual_atoms(mol, entry, "extra_anchors") if entry else []
    if ends or extra_anchors:
        anchors = [m[0] for _, m in ends] + extra_anchors
    else:
        anchors = _fallback_anchors(mol)

    def close(atoms: set[int]) -> set[int]:
        atoms = _expand_rings(mol, atoms)
        if keep_exocyclic_double_bonds:
            atoms = _add_exocyclic_double_bonded(mol, atoms)
        return atoms

    all_atoms = set(range(mol.GetNumAtoms()))
    main = set(end_atoms) | set(anchors)
    for a, b in itertools.combinations(anchors, 2):
        if a != b:
            main |= set(rdmolops.GetShortestPath(mol, a, b))
    main = close(main)

    # All ends on one ring: the chain cannot leave that ring, so a large leftover part is
    # most likely the rest of the backbone (e.g. the arms cut off in a fragment). Extend
    # the chain to the far end(s) of it.
    if ends and _on_one_ring(mol, anchors):
        big = max(_components(mol, all_atoms - main), key=len, default=set())
        if len(big) > len(main) / 2:
            dm = rdmolops.GetDistanceMatrix(mol)
            for far in _farthest_tied(dm, main, big):
                src = min(main, key=lambda a: dm[a][far])
                main |= set(rdmolops.GetShortestPath(mol, src, far))
            main = close(main)
            flags.append("ends_on_one_ring")

    side_chains = _components(mol, all_atoms - main)
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
