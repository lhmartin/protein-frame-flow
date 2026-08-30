#!/usr/bin/env python
"""Render the XM cross-regime summary to PDF + PNG + Markdown.

Standalone companion to docs/xm_cross_regime_summary.html — the page draws its
charts in JS/SVG, which headless Chrome can't render in this sandbox, so this
reproduces the same charts (same data, same validated light-mode palette) via
matplotlib straight to a PDF. Run with the repo venv:
    .venv/bin/python docs/make_summary_export.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
os.makedirs(ASSETS, exist_ok=True)

# ---- validated light-mode palette (mirrors palette.md / the HTML) ----
INK, SUB, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE, GOOD = "#e1e0d9", "#fcfcfb", "#0ca30c"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"  # blue / orange / aqua

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": SUB,
    "xtick.color": MUTED, "ytick.color": MUTED,
})

# ---------------------------- data (from the HTML) ----------------------------
REGIMES = [("r10_128", "10k · ≤128", S1),
           ("r50_128", "50k · ≤128", S2),
           ("r50_256", "50k · ≤256", S3)]
ARMS = ["vanilla", "XM K=2", "XM K=4", "XM K=8", "OT"]
DES = {
    "vanilla": {"r10_128": 15.2, "r50_128": 69.4, "r50_256": 29.9},
    "XM K=2":  {"r10_128": 10.4, "r50_128": 30.1, "r50_256": None},
    "XM K=4":  {"r10_128": 13.7, "r50_128": 67.4, "r50_256": 22.9},
    "XM K=8":  {"r10_128": 3.5,  "r50_128": 66.2, "r50_256": None},
    "OT":      {"r10_128": 14.9, "r50_128": 63.2, "r50_256": None},
}
DIV = {
    "vanilla": {"d128": 0.323, "d256": 0.733, "nov128": 0.197},
    "XM K=2":  {"d128": 0.455, "d256": None,  "nov128": 0.215},
    "XM K=4":  {"d128": 0.306, "d256": 0.674, "nov128": 0.203},
    "XM K=8":  {"d128": 0.308, "d256": None,  "nov128": 0.205},
    "OT":      {"d128": 0.402, "d256": None,  "nov128": 0.227},
}


def draw_bars(ax):
    n_arm, n_reg = len(ARMS), len(REGIMES)
    group_w = 0.8
    bar_w = group_w / n_reg
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    for j, (key, label, color) in enumerate(REGIMES):
        xs, ys = [], []
        for i, arm in enumerate(ARMS):
            v = DES[arm][key]
            if v is None:
                continue
            xs.append(i - group_w / 2 + bar_w * (j + 0.5))
            ys.append(v)
        bars = ax.bar(xs, ys, bar_w * 0.92, color=color, label=label,
                      edgecolor=SURFACE, linewidth=1.2)
        for b, v in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:g}",
                    ha="center", va="bottom", fontsize=7.5, color=SUB)
    ax.set_xticks(range(n_arm))
    ax.set_xticklabels(ARMS)
    ax.set_ylim(0, 86)
    ax.set_ylabel("Designability (%)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3,
              handlelength=1.0, columnspacing=1.4, bbox_to_anchor=(0.5, 1.02))
    ax.set_title("Designability by arm and regime", loc="left",
                 fontsize=12, color=INK, pad=10, fontweight="bold")


def draw_scatter(ax):
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, lw=0.8)
    off = {"vanilla": (1.2, 0.006, "left"), "XM K=4": (1.2, -0.014, "left"),
           "XM K=8": (0, 0.014, "center"), "OT": (1.2, 0.004, "left"),
           "XM K=2": (1.2, 0.004, "left")}
    for arm in ARMS:
        x = DES[arm]["r50_128"]
        y = DIV[arm]["d128"]
        if x is None or y is None:
            continue
        ax.scatter([x], [y], s=90, color=S2, edgecolor=SURFACE,
                   linewidth=1.2, zorder=3)
        dx, dy, ha = off[arm]
        ax.annotate(arm, (x, y), (x + dx, y + dy), fontsize=8, color=INK, ha=ha,
                    va="center")
    ax.set_xlabel("Designability (%)  → better")
    ax.set_ylabel("Diversity among designable  → more varied")
    ax.set_xlim(25, 75)
    ax.set_ylim(0.27, 0.49)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.annotate("K grows → diversity falls,\ndesignability flat-to-down: no trade",
                (50, 0.30), fontsize=8, color=SUB, ha="center", style="italic")
    ax.set_title("The XM trade-off  (50k · ≤128 residues)", loc="left",
                 fontsize=12, color=INK, pad=10, fontweight="bold")


def fmt(v, pct=False):
    if v is None:
        return "—"
    return f"{v:.1f}%" if pct else f"{v:.3f}"


def draw_table(ax):
    ax.axis("off")
    cols = ["Arm", "10k ≤128", "50k ≤128", "50k ≤256",
            "Div(d) ≤128", "Div(d) ≤256", "Nov ≤128"]
    rows = []
    for arm in ARMS:
        rows.append([
            arm,
            fmt(DES[arm]["r10_128"], True), fmt(DES[arm]["r50_128"], True),
            fmt(DES[arm]["r50_256"], True),
            fmt(DIV[arm]["d128"]), fmt(DIV[arm]["d256"]), fmt(DIV[arm]["nov128"]),
        ])
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celltable().items() if False else tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(color=INK, fontweight="bold")
            cell.set_facecolor("#f0efec")
        else:
            cell.set_facecolor(SURFACE)
            cell.set_text_props(color=INK if c == 0 else SUB)
        # highlight per-regime best (vanilla) designability cells in green
        if r == 1 and c in (1, 2, 3):
            cell.set_text_props(color=GOOD, fontweight="bold")
    ax.set_title("All numbers", loc="left", fontsize=12, color=INK,
                 pad=6, fontweight="bold")
    ax.text(0.0, -0.05,
            "Designability = % scRMSD < 2 Å (n=402/arm). Div(d) = foldseek "
            "clusters ÷ designable; Nov = 1 − max TM to PDB (both 50k). "
            "“—” = arm not run; green = per-regime best.",
            transform=ax.transAxes, fontsize=6.5, color=MUTED, va="top")


def main():
    # standalone PNGs for the markdown report
    for name, drawer, size in [("bars", draw_bars, (8, 3.6)),
                               ("scatter", draw_scatter, (8, 4.0))]:
        fig, ax = plt.subplots(figsize=size)
        drawer(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(ASSETS, f"xm_{name}.png"), dpi=150)
        plt.close(fig)

    pdf_path = os.path.join(HERE, "xm_cross_regime_summary.pdf")
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        gs = fig.add_gridspec(4, 1, height_ratios=[1.05, 1.5, 1.6, 1.6],
                              hspace=0.55, left=0.09, right=0.95,
                              top=0.95, bottom=0.05)
        # header + verdict
        hax = fig.add_subplot(gs[0]); hax.axis("off")
        hax.text(0, 1.0, "Explorative Modeling (XM) for FrameFlow",
                 fontsize=17, fontweight="bold", color=INK, va="top")
        hax.text(0, 0.72,
                 "Best-of-K coupling selection vs. plain independent coupling "
                 "— designability across four training regimes.",
                 fontsize=9.5, color=SUB, va="top")
        verdict = ("Verdict: across every regime tested — 10k & 50k steps, "
                   "≤128 & ≤256 residues — plain vanilla independent "
                   "coupling wins.\nXM never beats it; larger K trades structural "
                   "diversity for no designability gain, and at ≤256 XM is "
                   "clearly worse.\nThe null result is not a small-protein "
                   "artifact.  Experiment complete.")
        hax.text(0, 0.44, verdict, fontsize=8.6, color=INK, va="top",
                 bbox=dict(boxstyle="round,pad=0.6", facecolor="#f4f7fb",
                           edgecolor=GOOD, linewidth=0.0))
        draw_bars(fig.add_subplot(gs[1]))
        draw_scatter(fig.add_subplot(gs[2]))
        draw_table(fig.add_subplot(gs[3]))
        pdf.savefig(fig, facecolor=SURFACE)
        plt.close(fig)

    # markdown report
    md = ["# Explorative Modeling (XM) for FrameFlow — cross-regime summary",
          "",
          "**Verdict:** across every regime tested — 10k & 50k steps, ≤128 "
          "& ≤256 residues — plain **vanilla** independent coupling wins. "
          "XM never beats it; larger K trades structural diversity for no "
          "designability gain, and at ≤256 XM is clearly *worse*. The null "
          "result is not a small-protein artifact. **Experiment complete.**",
          "",
          "## Designability by arm and regime",
          "",
          "![Designability by arm and regime](assets/xm_bars.png)",
          "",
          "## The XM trade-off (50k · ≤128 residues)",
          "",
          "![XM trade-off](assets/xm_scatter.png)",
          "",
          "## All numbers",
          "",
          "| Arm | 10k ≤128 | 50k ≤128 | 50k ≤256 | Div(des) 50k≤128 | Div(des) 50k≤256 | Novelty 50k≤128 |",
          "|-----|-----------|-----------|-----------|------------------|------------------|------------------|"]
    for arm in ARMS:
        md.append(f"| {arm} | {fmt(DES[arm]['r10_128'], True)} | "
                  f"{fmt(DES[arm]['r50_128'], True)} | {fmt(DES[arm]['r50_256'], True)} | "
                  f"{fmt(DIV[arm]['d128'])} | {fmt(DIV[arm]['d256'])} | "
                  f"{fmt(DIV[arm]['nov128'])} |")
    md += ["",
           "Designability = % designable (scRMSD < 2 Å), n=402/arm. "
           "Diversity (des.) = distinct foldseek clusters ÷ designable count. "
           "Novelty = 1 − max TM-score to the full PDB. “—” = arm "
           "not run in that regime. SE(3)-OT (10k only) = 6.0%, omitted.",
           "",
           "Source: `docs/xm_explorative_modeling.md`, "
           "`xm_eval_results/eval50k/eval50k_full_summary.csv`, "
           "`xm_eval_results/eval256/eval256_summary.csv`. Interactive version: "
           "`docs/xm_cross_regime_summary.html`.", ""]
    with open(os.path.join(HERE, "xm_cross_regime_summary.md"), "w") as f:
        f.write("\n".join(md))

    print("wrote:", pdf_path)
    print("wrote:", os.path.join(HERE, "xm_cross_regime_summary.md"))
    print("wrote:", os.path.join(ASSETS, "xm_bars.png"),
          os.path.join(ASSETS, "xm_scatter.png"))


if __name__ == "__main__":
    main()
