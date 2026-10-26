"""EP-branded charts for the README. Every mark carries a text label, so no
information depends on color alone (palette checked with a CVD validator:
adjacent-pair separation passes; gold and gray sit below 3:1 contrast on the
off-white surface, so every segment is direct-labeled)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
NAVY, GOLD, GRAY, BG, INK, MUTED = "#0B1B33", "#D4A53A", "#8A94A3", "#F5F3EC", "#0B1B33", "#4A5363"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})


def outcomes():
    s = json.loads((OUT / "baseline_summary.json").read_text(encoding="utf-8"))
    c = s["status_counts"]
    n = s["drops"]
    parts = [("Regained half or more", c["recovered"], GOLD, NAVY),
             ("Did not regain half", c["not_recovered"], NAVY, "white"),
             ("No comparable MLB season next year", c["no_mlb_data"], GRAY, NAVY)]
    fig, ax = plt.subplots(figsize=(11, 3.6), facecolor=BG)
    ax.set_facecolor(BG)
    left = 0
    for label, k, col, txt in parts:
        ax.barh(0, k, left=left, color=col, height=0.5, edgecolor=BG, linewidth=2)
        ax.text(left + k / 2, 0, f"{k}\n{100 * k / n:.0f}%", ha="center", va="center",
                color=txt, fontsize=13, fontweight="bold")
        ax.text(left + k / 2, -0.42, label, ha="center", va="top", color=INK, fontsize=10.5)
        left += k
    h = s["headline_test"]
    ax.set_title("Fastball velocity drops of 1+ mph rarely come back the next season",
                 loc="left", color=INK, fontsize=14, fontweight="bold", pad=34)
    ax.text(0, 0.44, f"{n} MLB drops (2018, 2022-2025). Of the {h['n']} with a comparable next season, "
            f"{100 * h['rate']:.0f}% regained half the loss (95% CI {100 * h['wilson_ci'][0]:.0f}-"
            f"{100 * h['wilson_ci'][1]:.0f}%).\nBar shows counts and % of all {n} drops.", color=MUTED, fontsize=10.5)
    ax.set_xlim(0, n); ax.set_ylim(-0.75, 0.55); ax.axis("off")
    fig.text(0.01, 0.02, "Data: Baseball Savant (regular season, primary fastball, 300+ fastballs both years). "
             "Analysis: Emerson Jimenez, Emerson Performance (EP)", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "fig1_natural_recovery.png", dpi=180, facecolor=BG)


def board():
    b = pd.read_csv(OUT / "board_2026.csv")
    order = ["PHYSICAL PLAN", "MEDICAL FIRST", "ROLE CHANGE"]
    names = {"PHYSICAL PLAN": "Physical plan (no other cause)",
             "MEDICAL FIRST": "Medical first (absence or injury)",
             "ROLE CHANGE": "Role change (pen to rotation)"}
    rows, y, ticks, labels = [], 0, [], []
    for lane in order:
        sub = b[b["lane"] == lane].sort_values("drop_mph", ascending=False)
        y -= 1.2
        rows.append(("header", y, names[lane]))
        for _, r in sub.iterrows():
            y -= 1
            rows.append(("row", y, r))
    fig, ax = plt.subplots(figsize=(10, 10.5), facecolor=BG)
    ax.set_facecolor(BG)
    for kind, yy, r in rows:
        if kind == "header":
            ax.text(-0.02, yy, r, fontsize=11.5, fontweight="bold", color=INK, va="center", ha="right",
                    transform=ax.get_yaxis_transform())
            continue
        last, first = r.player_name.split(", ")
        ax.plot([r.velo_base, r.velo_drop_year], [yy, yy], color=GRAY, lw=2, zorder=1)
        ax.scatter(r.velo_base, yy, s=70, color=NAVY, zorder=2, edgecolor=BG, linewidth=2)
        ax.scatter(r.velo_drop_year, yy, s=70, color=GOLD, zorder=2, edgecolor=BG, linewidth=2)
        ax.text(-0.02, yy, f"{first} {last}", va="center", ha="right", fontsize=10, color=INK,
                transform=ax.get_yaxis_transform())
        ax.text(max(r.velo_base, r.velo_drop_year) + 0.3, yy, f"{r.drop_mph:+.1f}", va="center",
                fontsize=9.5, color=MUTED)
    ax.set_xlim(87.5, 101.5); ax.set_ylim(y - 1, 0.5)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_xticks(range(88, 102, 2))
    ax.grid(axis="x", color="#DAD6CB", lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("Primary fastball, season average (mph)")
    ax.scatter([], [], color=NAVY, s=60, label="2025"); ax.scatter([], [], color=GOLD, s=60, label="2026")
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.set_title("2026 offseason board: 21 pitchers who lost 1+ mph\n"
                 "Base rate: about 1 in 6 regains half on their own", loc="left",
                 color=INK, fontsize=13, fontweight="bold")
    fig.text(0.01, 0.01, "Data: Baseball Savant. Lanes use public injury reports (data/injury_exclusions.csv). "
             "Emerson Jimenez, EP", color=MUTED, fontsize=8.5)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.92, bottom=0.07)
    fig.savefig(FIG / "fig2_board_2026.png", dpi=180, facecolor=BG)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    outcomes(); board()
    print("saved")
