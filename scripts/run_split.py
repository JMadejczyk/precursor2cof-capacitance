"""Split every substrate in the dataset and write a CSV plus an HTML review page.

Usage: PYTHONPATH=. uv run python scripts/run_split.py [--data cof_matched/data_substrate_cofs.csv] [--out substrate_split/output]
"""

import argparse
import html
import io
import math
import os

import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XlImage
from openpyxl.drawing.spreadsheet_drawing import AnchorMarker, TwoCellAnchor
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage, rdMolDraw2D

from modules.features.substrate_split import split_substrate

MAIN_COLOR = (0.55, 0.75, 1.0)  # blue: main chain
END_COLOR = (0.3, 0.85, 0.4)  # green: reactive end groups
SIDE_COLOR = (1.0, 0.72, 0.25)  # orange: side chains (red clashes with RDKit's O atoms)
IMAGE_SIZE = (360, 270)


def draw_split(split, size=(420, 320), png=False):
    mol = Chem.MolFromSmiles(split.smiles)
    colors = {a: MAIN_COLOR for a in split.main_atoms}
    colors.update({a: END_COLOR for a in split.end_atoms})
    colors.update({a: SIDE_COLOR for a in split.side_atoms})
    bond_colors = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in colors and j in colors and colors[i] == colors[j]:
            bond_colors[b.GetIdx()] = colors[i]
    drawer = (rdMolDraw2D.MolDraw2DCairo if png else rdMolDraw2D.MolDraw2DSVG)(*size)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=list(colors),
        highlightAtomColors=colors,
        highlightBonds=list(bond_colors),
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    if png:
        return drawer.GetDrawingText()
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
<p class="legend"><span style="background:rgb(76,217,102)">end groups</span><span style="background:rgb(140,191,255)">main chain</span><span style="background:rgb(255,184,64)">side chains</span></p>
<p><label><input type="checkbox" id="ctf"> CTF (nitrile) only</label> <label><input type="checkbox" id="flag"> flagged only</label></p>
<div class="grid">{"".join(cards)}</div>
<script>
const f=()=>document.querySelectorAll('.card').forEach(c=>{{c.style.display=((!ctf.checked||c.dataset.ctf==='1')&&(!flag.checked||c.dataset.flag==='1'))?'':'none'}});
ctf.onchange=f; flag.onchange=f;
</script></body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)


def draw_fragment(smiles: str) -> bytes:
    drawer = rdMolDraw2D.MolDraw2DCairo(*IMAGE_SIZE)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, Chem.MolFromSmiles(smiles))
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def draw_side_chains(smiles_list: list[str]) -> bytes:
    cols = min(len(smiles_list), 3)
    rows = math.ceil(len(smiles_list) / cols)
    img = MolsToGridImage(
        [Chem.MolFromSmiles(s) for s in smiles_list],
        molsPerRow=cols,
        subImgSize=(IMAGE_SIZE[0] // cols, IMAGE_SIZE[1] // rows),
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def write_review_xlsx(df: pd.DataFrame, splits: list, path: str) -> None:
    headers = [
        "smiles (substrate)",
        "substrate",
        "smiles (main chain)",
        "main chain",
        "smiles (side chains)",
        "side chains",
    ]
    widths = [34, 52, 34, 52, 34, 52]
    wb = Workbook()
    ws = wb.active
    ws.title = "splits"
    ws.append(headers)
    for i, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = width
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(df) + 1}"

    def add_image(data: bytes, column: int, r: int) -> None:
        """Pin the image to one cell (0-based column) so it follows filters and sorting."""
        img = XlImage(io.BytesIO(data))
        img.anchor = TwoCellAnchor(
            editAs="twoCell",
            _from=AnchorMarker(col=column, row=r - 1),
            to=AnchorMarker(col=column + 1, row=r),
        )
        ws.add_image(img)

    for r, ((_, row), split) in enumerate(zip(df.iterrows(), splits), start=2):
        ws.append(
            [
                row.smiles,
                None,
                row.main_chain_smiles,
                None,
                " . ".join(split.side_chain_smiles),
                None,
            ]
        )
        ws.row_dimensions[r].height = IMAGE_SIZE[1] * 0.75 + 4
        for cell in ws[r]:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        add_image(draw_split(split, IMAGE_SIZE, png=True), 1, r)
        add_image(draw_fragment(row.main_chain_smiles), 3, r)
        if split.side_chain_smiles:
            add_image(draw_side_chains(split.side_chain_smiles), 5, r)

    wb.save(path)


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
    rows, svgs, splits = [], [], []
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
        splits.append(s)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "substrate_split.csv"), index=False)
    write_review_html(df, svgs, os.path.join(args.out, "review.html"))
    write_review_xlsx(df, splits, os.path.join(args.out, "substrate_split.xlsx"))
    print(f"{len(df)} substrates, {df.is_ctf.sum()} CTF, {(df.n_side_chains > 0).sum()} with side chains")
    print(df["flags"].str.split(";").explode().value_counts().drop("", errors="ignore").to_string())


if __name__ == "__main__":
    main()
