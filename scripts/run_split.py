"""Split every substrate in the dataset and write a CSV plus an HTML review page.

Usage: PYTHONPATH=. uv run python scripts/run_split.py [--data cof_matched/data_substrate_cofs.csv] [--out substrate_split/output]
"""

import argparse
import html
import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from modules.features.substrate_split import split_substrate

MAIN_COLOR = (0.55, 0.75, 1.0)  # blue: main chain
END_COLOR = (0.3, 0.85, 0.4)  # green: reactive end groups
SIDE_COLOR = (1.0, 0.6, 0.55)  # red: side chains


def draw_split(split, size=(420, 320)) -> str:
    mol = Chem.MolFromSmiles(split.smiles)
    colors = {a: MAIN_COLOR for a in split.main_atoms}
    colors.update({a: END_COLOR for a in split.end_atoms})
    colors.update({a: SIDE_COLOR for a in split.side_atoms})
    bond_colors = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in colors and j in colors and colors[i] == colors[j]:
            bond_colors[b.GetIdx()] = colors[i]
    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=list(colors),
        highlightAtomColors=colors,
        highlightBonds=list(bond_colors),
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")


def write_review_html(df: pd.DataFrame, svgs: list[str], path: str) -> None:
    cards = []
    for (_, row), svg in zip(df.iterrows(), svgs):
        flags = f"<div class='flags'>{html.escape(row['flags'])}</div>" if row["flags"] else ""
        cards.append(
            f"<div class='card' data-ctf='{int(row.is_ctf)}' data-flag='{int(bool(row['flags']))}'>"
            f"<div class='hd'>#{row.idx} · capacity {row.capacity_max:g} · ends: {html.escape(row.end_types)} ({row.n_ends})</div>{svg}"
            f"<div class='smi'><b>main</b> {html.escape(row.main_chain_smiles)}</div>"
            f"<div class='smi'><b>side</b> {html.escape(row.side_chain_smiles) or '—'}</div>{flags}</div>"
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Substrate split review</title>
<style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px}}
.card svg{{width:100%;height:auto}} .hd{{font-weight:600;font-size:13px}}
.smi{{font-family:monospace;font-size:11px;word-break:break-all;margin-top:4px}}
.flags{{color:#b00;font-size:12px;margin-top:4px}} .legend span{{padding:2px 8px;border-radius:4px;margin-right:8px}}
</style></head><body>
<h1>Substrate main/side chain split ({len(df)} substrates)</h1>
<p class="legend"><span style="background:rgb(76,217,102)">end groups</span><span style="background:rgb(140,191,255)">main chain</span><span style="background:rgb(255,153,140)">side chains</span></p>
<p><label><input type="checkbox" id="ctf"> CTF (nitrile) only</label> <label><input type="checkbox" id="flag"> flagged only</label></p>
<div class="grid">{"".join(cards)}</div>
<script>
const f=()=>document.querySelectorAll('.card').forEach(c=>{{c.style.display=((!ctf.checked||c.dataset.ctf==='1')&&(!flag.checked||c.dataset.flag==='1'))?'':'none'}});
ctf.onchange=f; flag.onchange=f;
</script></body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(here, "..", "cof_matched", "data_substrate_cofs.csv"),
    )
    parser.add_argument("--out", default=os.path.join(here, "..", "substrate_split", "output"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = pd.read_csv(args.data)
    rows, svgs = [], []
    for idx, (smi, cap) in enumerate(zip(data.smiles, data.capacity_max)):
        s = split_substrate(smi)
        rows.append(
            {
                "idx": idx,
                "smiles": smi,
                "capacity_max": cap,
                "is_ctf": "nitrile" in s.end_types,
                "end_types": ";".join(s.end_types),
                "n_ends": s.n_ends,
                "main_chain_smiles": s.main_chain_smiles,
                "side_chain_smiles": ".".join(s.side_chain_smiles),
                "n_side_chains": len(s.side_chains),
                "n_main_atoms": len(s.main_atoms),
                "n_side_atoms": len(s.side_atoms),
                "flags": ";".join(s.flags),
            }
        )
        svgs.append(draw_split(s))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "substrate_split.csv"), index=False)
    write_review_html(df, svgs, os.path.join(args.out, "review.html"))
    print(f"{len(df)} substrates, {df.is_ctf.sum()} CTF, {(df.n_side_chains > 0).sum()} with side chains")
    print(df["flags"].str.split(";").explode().value_counts().drop("", errors="ignore").to_string())


if __name__ == "__main__":
    main()
