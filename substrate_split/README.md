# substrate_split

Splits each COF substrate into a **main chain** and **side chains** 
It's separate from `precursor2cof-capacitance/` and doesn't import or modify it.

## Run
```bash
uv sync
uv run python run_split.py            # reads ../cof_matched/data_substrate_cofs.csv
```
Outputs in `output/`:
- `substrate_split.csv`: one row per substrate: `end_types`, `n_ends`, `main_chain_smiles`,
  `side_chain_smiles` (joined by `.`, `*` = attachment point), atom counts, `flags`.
- `review.html`: every substrate drawn with end groups (green), main chain (blue), side chains (red).
  Has filters for "CTF only" and "flagged only".

## How the split works (`splitter.py`)
1. **End groups.** Tier 1: nitrile, aldehyde, primary amine, boronic acid, terminal alkyne, SnMe3.
   Tier 2 (halides, phenol OH) is added only if tier 1 finds < 2 ends. Halides are also added when a
   cross-coupling group (boronic acid / alkyne / stannane) is present.
2. **Main chain** = shortest paths between every pair of ends, then every ring sharing ≥ 2 atoms with it
   (repeated, so fused systems stay whole), then terminal `=O`/`=S` atoms on it.
3. **Side chains** = whatever is left, as connected pieces.

## Status / known issues
- The 29 CTF (nitrile) substrates all split without flags and look correct by eye.
- 47 imine substrates are amine + aldehyde monomers already joined by `N=C`. Both end types are used, and the main chain runs through the imine.
- `large_side_chain` (16 substrates): the side chain is more than half the size of the main chain. These usually mean an end
  group wasn't recognised (COOH, imide N–H, o-diamine, active CH2, …). Needs a chemist's check.
- `no_ends_fallback_diameter` (3 substrates): no reactive group found, so the graph's longest path is used.
- Open questions for the chemists: should ring `C=O` count as main chain (currently yes)? Is OH on an aldehyde
  ring a side chain (currently yes)?
