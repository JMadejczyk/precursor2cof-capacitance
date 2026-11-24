"""Figures for the explainability experiment. Reads results/ (run cv.py, summarize.py, explain.py, feature_checks.py
and electro_shap.py first) and writes PNGs to results/plots/.

1_models.png         which models beat a shuffled target and the size baseline, and how much they track size
2_size.png           capacitance against heavy-atom count, CTF and non-CTF
3_shap_capacity.png  SHAP beeswarms of the best capacitance models (how they predict)
4_beyond_size.png    SHAP importance of models trained on size-corrected capacitance
5_feature_checks.png model-free check: size-corrected capacitance with vs without each substructure
6_pred_vs_true.png   held-out predictions vs truth: size only vs the best model
7_shap_electro.png   SHAP beeswarm of the J model that ranks within papers

Run from the repo root:  PYTHONPATH=. .venv/bin/python experiments/explainability/plots.py
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rdkit import Chem
from scipy.stats import rankdata, spearmanr

from experiments.electrocatalysis.common import SUBSETS, load_config
from experiments.electrocatalysis.common import load_data as load_electro
from experiments.explainability.common import TARGET, WORK_DIR, load_data, load_table, size_residual
from experiments.explainability.feature_checks import GROUPS

OUT = os.path.join(WORK_DIR, "plots")

# palette (reference instance of the dataviz skill, light mode)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2: CTF / non-CTF, and two-series panels
UP, DOWN, NEUTRAL = "#2a78d6", "#e34948", "#c3c2b7"  # diverging poles: raises / lowers the prediction
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

# ECFP bits decoded in this experiment (atom environments checked with RDKit bit info)
ECFP_LABELS = {
    849: "aromatic CH",
    356: "aromatic C, no H",
    726: "aromatic CH + neighbours",
    875: "aromatic CH (c(c)c)",
    136: "biaryl C (Ar–Ar)",
    807: "hydroxyl O",
    675: "aromatic S (thiophene)",
    378: "aromatic N, no H",
    116: "imine N (Ar–N=C)",
    62: "aromatic C next to ring N/S",
    830: "aryl C–N",
    1004: "carbonyl O=C",
    525: "imine / aryl ketone",
    361: "triazine C–aryl",
    486: "amide C(=O)N",
    946: "aromatic C–OH",
    963: "aryl C=O",
    593: "aryl aldehyde",
    896: "biaryl CH",
}
ESTATE_LABELS = {
    "aaCH": "aromatic CH",
    "aasC": "substituted aromatic C",
    "aaaC": "fused aromatic C",
    "aaN": "aromatic N",
    "aaS": "aromatic S",
    "sOH": "OH",
    "sNH2": "NH2",
    "dO": "=O",
    "dsN": "imine N (=N–)",
    "dsCH": "=CH–",
    "tN": "nitrile N",
    "dssC": "C= (e.g. C=O)",
}
FR_LABELS = {"fr_benzene": "benzene rings", "fr_bicyclic": "bicyclic", "fr_NH2": "NH2 groups", "fr_Imine": "imines", "fr_nitrile": "nitriles", "fr_aldehyde": "aldehydes"}


def style() -> None:
    """Recessive axes and grid, system sans, light surface."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "DejaVu Sans"],
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


def label(feature: str, name: str, substructures: str | float = np.nan) -> str:
    """Short chemist-readable label for a feature."""
    if feature.startswith("ecfp_"):
        bit = int(feature.split("_")[1])
        if bit in ECFP_LABELS:
            return f"{ECFP_LABELS[bit]}  [ECFP {bit}]"
        first = str(substructures).split(" | ", maxsplit=1)[0].rsplit(" (", 1)[0] if isinstance(substructures, str) else ""
        return f"ECFP {bit}: {first}"
    if name.startswith("EState "):
        code = name.split(" ", 1)[1]
        return f"{ESTATE_LABELS.get(code, code)}  [EState {code}]"
    if name.startswith("fr_"):
        return f"{FR_LABELS.get(name, name[3:])}  [{name}]"
    return {"mol_wt": "molecular weight", "n%": "N %", "o%": "O %", "c%": "C %", "diameter_filtered": "diameter (filtered)"}.get(name, name)


def save(fig: plt.Figure, name: str) -> None:
    """Write a PNG at 200 dpi."""
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


def swarm_offsets(x: np.ndarray, width: float = 0.38, n_bins: int = 40) -> np.ndarray:
    """Vertical offsets that spread points of similar x apart (a simple beeswarm)."""
    if np.ptp(x) == 0:
        return np.zeros_like(x)
    bins = np.digitize(x, np.linspace(x.min(), x.max(), n_bins))
    offsets = np.zeros_like(x, dtype=float)
    max_count = max(np.bincount(bins))
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        k = np.arange(len(idx))
        offsets[idx] = ((k + 1) // 2) * np.where(k % 2 == 0, 1, -1) * width / max(max_count / 2, 1)
    return offsets


def beeswarm(ax: plt.Axes, shap_df: pd.DataFrame, X: pd.DataFrame, importance: pd.DataFrame, n: int, size_rho: bool = True) -> None:
    """SHAP beeswarm of the top-n features; colour = feature value rank within the feature."""
    top = importance.head(n).iloc[::-1]
    for row, (_, f) in enumerate(top.iterrows()):
        s, v = shap_df[f["feature"]].to_numpy(), X[f["feature"]].to_numpy(dtype=float)
        colour = (rankdata(v) - 1) / max(len(v) - 1, 1) if np.ptp(v) > 0 else np.full(len(v), 0.5)
        ax.scatter(s, row + swarm_offsets(s), c=colour, cmap=SEQ, vmin=0, vmax=1, s=9, linewidths=0.3, edgecolors=SURFACE, zorder=3)
    labels = []
    for _, f in top.iterrows():
        text = label(f["feature"], f.get("name", f["feature"]), f.get("substructures", np.nan))
        if size_rho and abs(f.get("rho_heavy_atoms", 0)) >= 0.7:
            text += "  · size"
        labels.append(text)
    ax.set_yticks(range(len(top)), labels)
    ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(top) - 0.3)


def fig_models(summary: pd.DataFrame, data: pd.DataFrame) -> None:
    """Panel A: R2 of the top configurations vs shuffled target and size. Panel B: how much they track size."""
    cap = summary[summary["target"] == "capacity"].copy()
    cap["config"] = cap["model"] + " · " + cap["table"].str.replace("_", " ")
    size_row = cap[(cap["table"] == "size") & (cap["model"] == "BRR")].iloc[0]
    shown = pd.concat([cap[cap["model"] != "Median"].head(14), cap[(cap["table"] == "size") & (cap["model"] == "BRR")], cap[(cap["model"] == "Median")].head(1)])
    shown = shown.drop_duplicates("config").sort_values("r2")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.35, 1]}, layout="constrained")
    for row, (_, r) in enumerate(shown.iterrows()):
        d = os.path.join(WORK_DIR, "cv", "capacity", r["table"], {v: k for k, v in MODEL_KEYS.items()}[r["model"]])
        reps, null = pd.read_csv(os.path.join(d, "metrics.csv"))["r2"], pd.read_csv(os.path.join(d, "null.csv"))["r2"]
        colour = BLUE if r["win_vs_size"] == 1 and r["table"] != "size" else MUTED
        ax.plot([reps.min(), reps.max()], [row, row], color=colour, lw=2, solid_capstyle="round", alpha=0.45, zorder=2)
        ax.scatter(r["r2"], row, s=42, color=colour, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.scatter(null.quantile(0.95), row, marker="|", s=90, color=INK2, linewidths=1.2, zorder=3)
    ax.axvline(size_row["r2"], color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(size_row["r2"], len(shown) - 0.4, " size only (BRR)", color=INK2, fontsize=8, va="bottom")
    ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
    ax.set_yticks(range(len(shown)), ["Median (baseline)" if c.startswith("Median") else c for c in shown["config"]])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("R² of held-out capacitance (dot = mean of 10 CV repeats, bar = range)")
    ax.set_title("A. Which models predict capacitance?", pad=34)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=BLUE, label="beats size in all 10 repeats"),
            Line2D([], [], marker="o", ls="", color=MUTED, label="does not"),
            Line2D([], [], marker="|", ls="", color=INK2, markersize=9, label="95th pct. of shuffled target"),
        ],
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=3,
    )

    # panel B: correlation of held-out predictions with size, and with capacitance once size is removed
    y, heavy = data[TARGET].to_numpy(dtype=float), data["heavy_atoms"].to_numpy(dtype=float)
    best = cap[(cap["model"] != "Median") & (cap["table"] != "size")].head(6).iloc[::-1]
    for row, (_, r) in enumerate(best.iterrows()):
        oof = np.load(os.path.join(WORK_DIR, "cv", "capacity", r["table"], {v: k for k, v in MODEL_KEYS.items()}[r["model"]], "oof.npy")).mean(axis=0)
        a, b = spearmanr(oof, heavy).statistic, partial_spearman(oof, y, heavy)
        bx.plot([a, b], [row, row], color=GRID, lw=1.5, zorder=1)
        bx.scatter(a, row, s=42, color=ORANGE, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        bx.scatter(b, row, s=42, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    bx.axvline(0, color=AXIS, lw=0.8)
    bx.set_yticks(range(len(best)), best["config"])
    bx.grid(axis="y", visible=False)
    bx.set_xlim(-1, 1)
    bx.set_xlabel("Spearman correlation")
    bx.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=ORANGE, label="prediction vs heavy-atom count"),
            Line2D([], [], marker="o", ls="", color=BLUE, label="prediction vs capacitance,\nsize partialled out"),
        ],
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=2,
    )
    bx.set_title("B. How much of it is size?", pad=34)
    save(fig, "1_models.png")


MODEL_KEYS = {"median": "Median", "bayes": "BRR", "lasso": "Lasso", "random_forest": "RF", "xgboost": "XGBoost", "knn": "KNN", "tabpfn": "TabPFN"}


def partial_spearman(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Spearman correlation of a and b after removing a linear fit on c (on ranks)."""
    a, b, c = rankdata(a), rankdata(b), rankdata(c)
    ra, rb = a - np.polyval(np.polyfit(c, a, 1), c), b - np.polyval(np.polyfit(c, b, 1), c)
    return float(np.corrcoef(ra, rb)[0, 1])


def fig_size(data: pd.DataFrame) -> None:
    """Capacitance vs heavy-atom count, CTF and non-CTF, with a straight-line fit for each."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for is_ctf, colour, name in [(False, ORANGE, "other COFs"), (True, BLUE, "CTFs")]:
        d = data[data["is_ctf"] == is_ctf]
        rho = spearmanr(d["heavy_atoms"], d[TARGET]).statistic
        ax.scatter(d["heavy_atoms"], d[TARGET], s=26, color=colour, edgecolors=SURFACE, linewidths=1, alpha=0.9, zorder=3, label=f"{name} (n={len(d)}, Spearman {rho:+.2f})")
        k, b = np.polyfit(d["heavy_atoms"], d[TARGET], 1)
        xs = np.linspace(d["heavy_atoms"].min(), d["heavy_atoms"].max(), 50)
        ax.plot(xs, k * xs + b, color=colour, lw=2, zorder=2)
    rho_all = spearmanr(data["heavy_atoms"], data[TARGET]).statistic
    ax.set_xlabel("heavy atoms in the substrate")
    ax.set_ylabel("capacity_max")
    ax.set_title(f"Bigger monomers, lower capacitance (all: Spearman {rho_all:+.2f})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "2_size.png")


def load_explain(name: str) -> tuple:
    """:return: (shap dataframe, feature dataframe, importance) of one explain.py output."""
    d = os.path.join(WORK_DIR, "explain", name)
    shap_df, importance = pd.read_csv(os.path.join(d, "shap_oof.csv")), pd.read_csv(os.path.join(d, "importance.csv"))
    table = name.split("__")[1]
    return shap_df, load_table(table), importance


def fig_shap_capacity() -> None:
    """Beeswarms of the best RF on EState + FG and on ECFP."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), layout="constrained")
    for ax, (name, title) in zip(axes, [("capacity__rf_best__random_forest", "RF · EState + functional groups"), ("capacity__ecfp__random_forest", "RF · ECFP")]):
        shap_df, X, importance = load_explain(name)
        beeswarm(ax, shap_df, X, importance, 12)
        ax.set_title(title)
        ax.set_xlabel("SHAP value (change in predicted capacity_max)")
    sm = mpl.cm.ScalarMappable(cmap=SEQ, norm=mpl.colors.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01, ticks=[0, 1])
    cbar.ax.set_yticklabels(["low", "high"])
    cbar.set_label("feature value", color=INK2)
    cbar.outline.set_visible(False)
    fig.suptitle("How the capacitance models predict (held-out SHAP; '· size' = Spearman ≥ 0.7 with heavy-atom count)", x=0.01, ha="left", fontsize=10, fontweight="semibold")
    save(fig, "3_shap_capacity.png")


def fig_beyond_size() -> None:
    """Importance of models trained on capacitance minus the size trend, coloured by direction."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for ax, (name, title) in zip(axes, [("size_resid__estate__random_forest", "RF · EState"), ("size_resid__ecfp__random_forest", "RF · ECFP")]):
        imp = pd.read_csv(os.path.join(WORK_DIR, "explain", name, "importance.csv")).head(10).iloc[::-1]
        colours = [UP if d >= 0.3 else DOWN if d <= -0.3 else NEUTRAL for d in imp["direction"]]
        ax.barh(range(len(imp)), imp["mean_abs_shap"], color=colours, height=0.62, edgecolor=SURFACE, linewidth=1)
        ax.set_yticks(range(len(imp)), [label(f, n, s) for f, n, s in zip(imp["feature"], imp["name"], imp.get("substructures", [np.nan] * len(imp)))])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("mean |SHAP| (size-corrected capacity_max)")
        ax.set_title(title)
    axes[0].legend(
        handles=[Patch(color=UP, label="more of it → higher capacitance"), Patch(color=DOWN, label="more of it → lower"), Patch(color=NEUTRAL, label="mixed (|direction| < 0.3)")],
        loc="lower right",
        framealpha=1,
        frameon=True,
        edgecolor=SURFACE,
    )
    fig.suptitle("What is left once size is removed (suggestive: q ≈ 0.06)", x=0.01, ha="left", fontsize=10, fontweight="semibold")
    save(fig, "4_beyond_size.png")


def fig_feature_checks(data: pd.DataFrame) -> None:
    """Median difference in size-corrected capacitance, with vs without each substructure, with a bootstrap 95% CI."""
    y, heavy = data[TARGET].to_numpy(dtype=float), data["heavy_atoms"].to_numpy(dtype=float)
    resid = size_residual(y, heavy)
    mols = [Chem.MolFromSmiles(s) for s in data["smiles"]]
    rng = np.random.default_rng(0)
    rows = []
    for name, smarts in GROUPS.items():
        has = np.array([m.HasSubstructMatch(Chem.MolFromSmarts(smarts)) for m in mols])
        if not 3 <= has.sum() <= len(has) - 3 or name == "nitrile (C#N)":
            continue
        a, b = resid[has], resid[~has]
        boots = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b))) for _ in range(2000)]
        rows.append({"group": name, "n": int(has.sum()), "d": np.median(a) - np.median(b), "lo": np.quantile(boots, 0.025), "hi": np.quantile(boots, 0.975)})
    df = pd.DataFrame(rows).sort_values("d")
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for row, (_, r) in enumerate(df.iterrows()):
        colour = UP if r["lo"] > 0 else DOWN if r["hi"] < 0 else MUTED
        ax.plot([r["lo"], r["hi"]], [row, row], color=colour, lw=2, solid_capstyle="round", alpha=0.5)
        ax.scatter(r["d"], row, s=42, color=colour, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(range(len(df)), [f"{g}  (n={n})" for g, n in zip(df["group"], df["n"])])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("size-corrected capacity_max, with − without\n(difference of medians, bootstrap 95% CI)")
    ax.set_title("Model-free check: only thiophene S stands out")
    fig.text(0.01, -0.02, "Groups were chosen after looking at SHAP, so the intervals are descriptive, not confirmatory.", color=MUTED, fontsize=8)
    fig.tight_layout()
    save(fig, "5_feature_checks.png")


def fig_pred_vs_true(data: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Held-out predictions (mean over 10 repeats) against the truth for size only and for the best model."""
    y, ctf = data[TARGET].to_numpy(dtype=float), data["is_ctf"].to_numpy()
    cap = summary[summary["target"] == "capacity"]
    best = cap[(cap["model"] != "Median") & (cap["table"] != "size")].iloc[0]
    configs = [("size", "bayes", "Size only · BRR"), (best["table"], {v: k for k, v in MODEL_KEYS.items()}[best["model"]], f"Best · {best['model']} · {best['table'].replace('_', ' ')}")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    lim = (min(y.min(), 0) - 20, y.max() + 30)
    for ax, (table, model, title) in zip(axes, configs):
        oof = np.load(os.path.join(WORK_DIR, "cv", "capacity", table, model, "oof.npy")).mean(axis=0)
        r = cap[(cap["table"] == table) & (cap["model"] == MODEL_KEYS[model])].iloc[0]
        ax.plot(lim, lim, color=AXIS, lw=1, ls=(0, (4, 3)), zorder=1)
        for mask, colour, name in [(~ctf, ORANGE, "other COFs"), (ctf, BLUE, "CTFs")]:
            ax.scatter(y[mask], oof[mask], s=24, color=colour, edgecolors=SURFACE, linewidths=1, alpha=0.9, zorder=3, label=name)
        ax.set_title(f"{title}\nR² {r['r2']:.2f}, Spearman {r['spearman']:.2f}")
        ax.set_xlabel("measured capacity_max")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    axes[0].set_ylabel("held-out prediction (mean of 10 repeats)")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "6_pred_vs_true.png")


def fig_shap_electro() -> None:
    """Beeswarm of the J model (RF, ECFP, clean rows, z-scored J, leave-one-paper-out)."""
    d = os.path.join(WORK_DIR, "electro", "clean__zscore__random_forest__a1_full__ecfp")
    if not os.path.exists(d):
        print("skip 7_shap_electro.png: run electro_shap.py first")
        return
    config = load_config()
    data = load_electro(config["work_dir"])
    rows = np.asarray(SUBSETS["clean"](data))
    X = pd.read_csv(os.path.join(config["work_dir"], "features", "a1_full__ecfp.csv")).loc[rows].reset_index(drop=True)
    shap_df, importance = pd.read_csv(os.path.join(d, "shap_oof.csv")), pd.read_csv(os.path.join(d, "importance.csv"))
    importance["name"] = importance["feature"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
    beeswarm(ax, shap_df, X, importance, 10, size_rho=False)
    ax.set_xlabel("SHAP value (within-paper SDs of J)   ← better catalyst · worse catalyst →")
    ax.set_title("How the J model ranks materials within a paper (RF · ECFP, held-out paper)")
    sm = mpl.cm.ScalarMappable(cmap=SEQ, norm=mpl.colors.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, ticks=[0, 1])
    cbar.ax.set_yticklabels(["low", "high"])
    cbar.set_label("feature value", color=INK2)
    cbar.outline.set_visible(False)
    save(fig, "7_shap_electro.png")


if __name__ == "__main__":
    style()
    os.makedirs(OUT, exist_ok=True)
    data_cap = load_data()
    summary_df = pd.read_csv(os.path.join(WORK_DIR, "summary.csv"))
    fig_models(summary_df, data_cap)
    fig_size(data_cap)
    fig_shap_capacity()
    fig_beyond_size()
    fig_feature_checks(data_cap)
    fig_pred_vs_true(data_cap, summary_df)
    fig_shap_electro()
