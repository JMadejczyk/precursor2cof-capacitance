# substrate_split

Splits each COF substrate into a **main chain** and **side chains**
The splitter lives in `modules/features/substrate_split.py` and the review script in `scripts/run_split.py`;
this folder keeps the documentation and the review output. Fingerprints of the split are in
`modules/features/split_fingerprints.py` (`main_ecfp`, `side_ecfp`, `split_descriptor`).

## Run
From the repository root, with the main environment (`uv sync`):
```bash
PYTHONPATH=. uv run python scripts/run_split.py     # reads cof_matched/data_substrate_cofs.csv
```
Outputs in `substrate_split/output/`:
- `substrate_split.csv`: one row per substrate: `end_types`, `n_ends`, `main_chain_smiles`,
  `side_chain_smiles` (joined by `.`, `*` = attachment point), atom counts, `flags`.
- `review.html`: every substrate drawn with end groups (green), main chain (blue), side chains (orange).
  Has filters for "CTF only" and "flagged only".
- `substrate_split.xlsx`: workbook for the chemists: substrate, main chain and side chains as SMILES and drawings.

## How the split works (`modules/features/substrate_split.py`)
The module docstring has the full rules. In short:
1. **End groups.** Tier 1 linkage groups (nitrile, aldehyde, primary amine, boronic acid, terminal alkyne,
   SnMe3, COOH, imide N–H, …). Tier 2 (halides, phenol OH) is added only if tier 1 finds < 2 ends. Halides are
   also added when a cross-coupling group (boronic acid / alkyne / stannane) is present.
2. **Main chain** = shortest paths between every pair of ends, then every ring sharing ≥ 2 atoms with it
   (repeated, so fused systems stay whole). Non-reacting `=O`/`=S`/OH stay side chains.
3. **Side chains** = whatever is left, as connected pieces.

Hand-checked substrates the rules get wrong are overridden in `modules/features/manual_ends.py`.

## Status / known issues
- The 29 CTF (nitrile) substrates all split without flags and look correct by eye.
- 47 imine substrates are amine + aldehyde monomers already joined by `N=C`. Both end types are used, and the main chain runs through the imine.
- `large_side_chain`: the side chain is more than half the size of the main chain, which usually means an end group
  wasn't recognised. No substrate has this flag at the moment.
- `no_ends_fallback_diameter` (2 substrates): no reactive group found, so the graph's longest path is used.
- `manual_ends` (5 substrates): split set by hand in `manual_ends.py`.
