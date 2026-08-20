"""End-to-end validation of the analysis stage on synthetic data.

Purpose: exercise the full path from manifest and hypotheses through scoring,
modelling and table emission, WITHOUT corpus access. A known disparity is
injected so the pipeline's recovered rate ratio can be checked against ground
truth. This validates the analysis chain independently of the data source.

Run:  python validate_synthetic.py
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config  # noqa: E402
from src import report as R  # noqa: E402
from src.scoring import Normaliser  # noqa: E402

WORDS = ("the quick brown fox jumps over a lazy dog while birds sing in bright "
         "morning air near quiet rivers and open fields").split()

# Injected ground truth: error rate multiplier relative to the US baseline.
TRUE_DISPARITY = {
    "us_baseline": 1.0,
    "scottish": 1.4,
    "indian": 2.0,
    "nigerian": 2.4,
    "jamaican": 3.0,
}
BASE_ERROR_RATE = 0.08


def corrupt(tokens, rate, rng):
    """Introduce substitutions, deletions and insertions at a target rate."""
    out = []
    for tok in tokens:
        roll = rng.random()
        if roll < rate * 0.6:                      # substitution
            out.append(rng.choice(WORDS))
        elif roll < rate * 0.85:                   # deletion
            continue
        elif roll < rate:                          # insertion
            out.extend([tok, rng.choice(WORDS)])
        else:
            out.append(tok)
    return out


def build(seed=17, speakers_per_group=12, utts_per_speaker=25):
    rng = np.random.default_rng(seed)
    manifest, whisper, w2v = [], [], []

    for group, multiplier in TRUE_DISPARITY.items():
        for s in range(speakers_per_group):
            speaker = f"{group}_spk{s:02d}"
            for u in range(utts_per_speaker):
                uid = f"{speaker}_{u}"
                length = int(rng.integers(8, 26))
                ref = list(rng.choice(WORDS, size=length))
                manifest.append({
                    "uid": uid, "split": "validation", "row_index": len(manifest),
                    "accent_group": group, "speaker_id": speaker,
                    "age": rng.choice(["18-30", "31-50", "51+"]),
                    "gender": rng.choice(["male", "female"]),
                    "text": " ".join(ref),
                    "duration_s": float(length) * 0.38,
                })
                rate = BASE_ERROR_RATE * multiplier
                whisper.append({"uid": uid,
                                "whisper_hyp": " ".join(corrupt(ref, rate, rng))})
                # Second system: uniformly worse, same ordering.
                w2v.append({"uid": uid,
                            "w2v_hyp": " ".join(corrupt(ref, rate * 1.6, rng))})

    return (pd.DataFrame(manifest), pd.DataFrame(whisper), pd.DataFrame(w2v))


def main() -> int:
    out = "outputs_synthetic"
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)

    cfg = Config(output_dir=out, bootstrap_iterations=400,
                 min_utterances_for_inference=50, min_speakers_for_inference=3)
    manifest, whisper, w2v = build()
    print(f"synthetic manifest: {len(manifest)} utterances, "
          f"{manifest.speaker_id.nunique()} speakers, "
          f"{manifest.accent_group.nunique()} groups\n")

    scored = R.score_all(manifest, {"whisper": whisper, "wav2vec2": w2v},
                         Normaliser())

    tables = {
        "Table V-1 Corpus composition": R.table_composition(manifest, cfg),
        "Table V-2 WER by accent group": R.table_wer(scored, cfg),
        "Table V-3 Adjusted disparity rate ratios": R.table_rate_ratios(scored, cfg),
        "Table V-4 Intersectional analysis": R.table_intersectional(scored, cfg),
        "Table V-5 Error composition": R.table_error_composition(scored, cfg),
    }
    for name, frame in tables.items():
        print(f"\n=== {name} ===")
        print(frame.to_string(index=False) if not frame.empty else "(empty)")

    # ---- validate against injected ground truth -------------------------
    print("\n=== VALIDATION: recovered vs injected disparity (Whisper) ===")
    rr = tables["Table V-3 Adjusted disparity rate ratios"]
    rr = rr[rr.system == "whisper"]
    ok = True
    print(f"{'group':<14}{'injected':>10}{'recovered':>12}{'status':>10}")
    for group, truth in TRUE_DISPARITY.items():
        if group == "us_baseline":
            continue
        rows = rr[rr.term.str.contains(group, regex=False)]
        if rows.empty:
            print(f"{group:<14}{truth:>10.2f}{'MISSING':>12}{'FAIL':>10}")
            ok = False
            continue
        got = float(rows.iloc[0]["rate_ratio"])
        within = abs(got - truth) / truth < 0.25
        ok &= within
        print(f"{group:<14}{truth:>10.2f}{got:>12.2f}{'ok' if within else 'FAIL':>10}")

    ordering = rr.sort_values("rate_ratio")["term"].tolist()
    print("\nrecovered ordering (least to most disadvantaged):")
    for t in ordering:
        print("   ", t)

    R.to_markdown(tables, os.path.join(out, "paper_tables.md"))
    print(f"\ntables written to {out}/")
    print("\nRESULT:", "PASS, analysis chain recovers injected disparities"
          if ok else "FAIL, recovered ratios diverge from ground truth")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
