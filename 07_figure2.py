#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = "/content/drive/MyDrive/Wasko_JPIM"
DEFAULT_IN = f"{BASE}/output/results/figure2_margins.csv"
DEFAULT_OUT = f"{BASE}/output/figures"

LIGHT = "#BFBFBF"
DARK = "#404040"
BARW = 0.32

PANELS = [
    {"moderator": "framing", "title": "Normative Framing",
     "xlabels": ["Normative framing absent", "Normative framing present"]},
    {"moderator": "apology", "title": "Apology",
     "xlabels": ["Apology absent", "Apology present"]},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_IN)
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    ap.add_argument("--name", default="Figure2_interaction_effects")
    ap.add_argument("--ymax", type=float, default=0.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    m = pd.read_csv(args.input)
    print(f"loaded {args.input}")
    print(m.to_string(index=False))

    ymax = args.ymax
    if ymax <= 0:
        ymax = float(np.ceil((m["pred_backlash_pct"].max() + 4) / 5.0) * 5)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.linewidth": 0.8,
    })

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3), dpi=400)

    for ax, panel in zip(axes, PANELS):
        sub = m[m["moderator"] == panel["moderator"]]
        absent = [float(sub[(sub.moderator_level == lv) &
                            (sub.focal_level == 0)]["pred_backlash_pct"].iloc[0])
                  for lv in (0, 1)]
        present = [float(sub[(sub.moderator_level == lv) &
                             (sub.focal_level == 1)]["pred_backlash_pct"].iloc[0])
                   for lv in (0, 1)]

        x = np.arange(2)
        b1 = ax.bar(x - BARW / 2, absent, BARW, color=LIGHT,
                    edgecolor="black", linewidth=0.5, label="Disclaimer absent")
        b2 = ax.bar(x + BARW / 2, present, BARW, color=DARK,
                    edgecolor="black", linewidth=0.5, label="Disclaimer present")

        for bars in (b1, b2):
            for bar in bars:
                ax.annotate(f"{bar.get_height():.1f}%",
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

        ax.set_title(panel["title"], fontsize=10, fontweight="bold", pad=10)
        ax.set_ylim(0, ymax)
        ticks = np.arange(0, ymax + 0.1, 5)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{v:.1f}%" for v in ticks], fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(panel["xlabels"], fontsize=8)
        ax.set_ylabel("Backlash (in %)", fontsize=8, style="italic", labelpad=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", length=3, width=0.8)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#D9D9D9", linewidth=0.6)
        ax.xaxis.grid(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    stem = os.path.join(args.outdir, args.name)
    for ext in ("png", "pdf", "tiff"):
        fig.savefig(f"{stem}.{ext}", dpi=400, bbox_inches="tight", facecolor="white")
        print(f"wrote {stem}.{ext}")


if __name__ == "__main__":
    main()
