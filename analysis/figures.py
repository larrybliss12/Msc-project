#!/usr/bin/env python3
"""Generate the paper's figures from the scored utterances.

Produces two publication-quality figures the Results section currently lacks:

  Figure 1  Forest plot of adjusted disparity rate ratios with 95% confidence
            intervals, both systems, reference line at parity.
  Figure 2  Word error rate by accent group and system, with speaker-level
            bootstrap intervals.

Both are computed on the included set (the paper's primary analysis), applying
the same inclusion criteria as the pipeline.

Run from the project root:

    python analysis\\figures.py

Reads outputs/scored_utterances.csv. Writes outputs/figure1_forest.png and
outputs/figure2_wer.png at 300 dpi.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config, REFERENCE_GROUP
from src.report import apply_inclusion_criteria
from src.scoring import bootstrap_wer_by_speaker, token_weighted_wer
from src.stats import fit_disparity_model

LABELS = {"us_baseline": "US English", "indian": "Indian",
          "scottish": "Scottish", "nigerian": "Nigerian", "jamaican": "Jamaican"}
ORDER = ["indian", "scottish", "nigerian", "jamaican"]
SYS = {"whisper": ("whisper_err", "whisper_reflen", "Whisper Large V3"),
       "wav2vec2": ("wav2vec2_err", "wav2vec2_reflen", "wav2vec 2.0")}
COLOURS = {"whisper": "#1F4E79", "wav2vec2": "#C55A11"}


def load():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "outputs", "scored_utterances.csv")
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Copy scored_utterances.csv into outputs/.")
    df = pd.read_csv(path)
    return here, apply_inclusion_criteria(df, Config())


def forest(df, here):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    systems = list(SYS)
    offsets = {"whisper": -0.16, "wav2vec2": 0.16}
    yticks, ylabels = [], []

    for gi, group in enumerate(ORDER):
        y0 = len(ORDER) - gi
        yticks.append(y0)
        ylabels.append(LABELS[group])
        for system in systems:
            err, reflen, _ = SYS[system]
            fit = fit_disparity_model(df, err, reflen, reference_group=REFERENCE_GROUP)
            idx = [i for i in fit.table.index if f"T.{group}]" in i]
            if not idx:
                continue
            r = fit.table.loc[idx[0]]
            rr, lo, hi = r["rate_ratio"], r["rr_ci_low"], r["rr_ci_high"]
            y = y0 + offsets[system]
            ax.plot([lo, hi], [y, y], color=COLOURS[system], lw=1.8, zorder=2)
            ax.plot(rr, y, "o", color=COLOURS[system], ms=7, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.8)

    ax.axvline(1.0, color="#555", ls="--", lw=1, zorder=1)
    ax.text(1.0, len(ORDER) + 0.6, "parity", ha="center", va="bottom",
            fontsize=8, color="#555")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylim(0.4, len(ORDER) + 0.8)
    ax.set_xlabel("Adjusted disparity rate ratio vs US English (95% CI)")
    ax.set_title("Accent disparity in ASR, adjusted for utterance length\n"
                 "and speaker clustering", fontsize=11)
    handles = [plt.Line2D([], [], color=COLOURS[s], marker="o", lw=1.8,
                          label=SYS[s][2]) for s in systems]
    ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(here, "outputs", "figure1_forest.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("written:", out)


def wer_bars(df, here):
    groups = [REFERENCE_GROUP] + ORDER
    cfg = Config()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    width = 0.38
    x = np.arange(len(groups))

    for si, system in enumerate(SYS):
        err, reflen, name = SYS[system]
        wers, los, his = [], [], []
        for g in groups:
            sub = df[df.accent_group == g]
            p, lo, hi = bootstrap_wer_by_speaker(
                sub.speaker_id.values, sub[err].values, sub[reflen].values,
                n_iterations=cfg.bootstrap_iterations, seed=cfg.seed)
            wers.append(p)
            los.append(p - lo if np.isfinite(lo) else 0)
            his.append(hi - p if np.isfinite(hi) else 0)
        ax.bar(x + (si - 0.5) * width, wers, width, label=name,
               color=COLOURS[system], yerr=[los, his], capsize=3,
               error_kw={"lw": 1, "ecolor": "#333"})

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[g] for g in groups])
    ax.set_ylabel("Word error rate")
    ax.set_title("Word error rate by accent group and system\n"
                 "(included set, 95% speaker-level bootstrap CI)", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e5e5e5", lw=0.8)
    fig.tight_layout()
    out = os.path.join(here, "outputs", "figure2_wer.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("written:", out)


def main() -> int:
    here, df = load()
    print(f"figures computed on {len(df)} included utterances\n")
    forest(df, here)
    wer_bars(df, here)
    print("\nInsert Figure 1 after the rate-ratio table (IV-C) and Figure 2 "
          "after the WER table (IV-B).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
