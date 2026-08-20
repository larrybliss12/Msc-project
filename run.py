#!/usr/bin/env python3
"""EdAcc accent-fairness measurement study, staged pipeline runner.

Usage (VSCode terminal, or any shell):

    python run.py audit                  # stage 1: schema + accent enumeration
    python run.py manifest               # stage 2: build utterance manifest
    python run.py infer --system whisper # stage 3a
    python run.py infer --system wav2vec2# stage 3b
    python run.py analyse                # stage 4: score, model, emit tables
    python run.py all --smoke 200        # end to end on a small subset

Every stage writes its artefacts to --output-dir and can be re-run
independently. Inference resumes from partial output.
"""
from __future__ import annotations

import os as _os
# Must precede any torch import: reduces allocator fragmentation, which is a
# common cause of out-of-memory on mid-sized GPUs during long inference runs.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import logging
import os
import sys

import pandas as pd

from src.config import Config, STUDY_GROUPS
from src.data import (
    load_splits, probe_schema, resolve_columns, audit_accents,
    build_manifest, EmptyStageError,
)
from src.mapping import audit_dictionary
from src import report as R
from src.scoring import Normaliser


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _paths(cfg: Config) -> dict:
    os.makedirs(cfg.output_dir, exist_ok=True)
    d = cfg.output_dir
    return {
        "schema": os.path.join(d, "schema.json"),
        "raw_labels": os.path.join(d, "raw_accent_labels.csv"),
        "group_audit": os.path.join(d, "group_audit.csv"),
        "manifest": os.path.join(d, "manifest.csv"),
        "hyp_whisper": os.path.join(d, "hyp_whisper.csv"),
        "hyp_wav2vec2": os.path.join(d, "hyp_wav2vec2.csv"),
        "scored": os.path.join(d, "scored_utterances.csv"),
        "config": os.path.join(d, "run_config.json"),
        "tables_md": os.path.join(d, "paper_tables.md"),
    }


# --------------------------------------------------------------------------

def stage_audit(cfg: Config) -> None:
    p = _paths(cfg)
    collisions = audit_dictionary()
    if collisions:
        logging.warning("dictionary overlaps (resolved by longest-token rule):")
        for a, b, detail in collisions:
            logging.warning("   %s vs %s : %s", a, b, detail)

    splits = load_splits(cfg)
    first = splits[next(iter(splits))]
    schema = probe_schema(first)
    print("\nCOLUMNS:", schema["columns"])
    print("\nSAMPLE ROW:")
    for k, v in schema["sample"].items():
        print(f"  {k}: {v}")

    cfg = resolve_columns(cfg, first)
    with open(p["schema"], "w") as fh:
        json.dump({"schema": schema, "resolved": {
            r: getattr(cfg, r) for r in vars(cfg) if r.startswith("col_")}},
            fh, indent=2)

    raw_df, summary_df = audit_accents(cfg, splits)
    raw_df.to_csv(p["raw_labels"], index=False)
    summary_df.to_csv(p["group_audit"], index=False)
    cfg.to_json(p["config"])

    print("\n=== GROUP AUDIT ===")
    print(summary_df.to_string(index=False))
    print(f"\ndistinct raw label strings: {raw_df['value'].nunique()}")
    print(f"artefacts written to {cfg.output_dir}/")

    excluded = summary_df[(~summary_df.admitted_to_modelling)
                          & (summary_df.accent_group != "(labelled but unmapped)")]
    for _, row in excluded.iterrows():
        print(f"  ! {row.accent_group}: {row.utterances} utterances, "
              f"{row.speakers} speakers, below viability threshold")


def stage_manifest(cfg: Config) -> None:
    p = _paths(cfg)
    splits = load_splits(cfg)
    cfg = resolve_columns(cfg, splits[next(iter(splits))])

    admitted = None
    if os.path.exists(p["group_audit"]):
        audit = pd.read_csv(p["group_audit"])
        admitted = audit.loc[audit.admitted_to_modelling == True,  # noqa: E712
                             "accent_group"].tolist()
        logging.info("restricting manifest to admitted groups: %s", admitted)

    manifest = build_manifest(cfg, splits, admitted=admitted)
    manifest.to_csv(p["manifest"], index=False)
    print(R.table_composition(manifest, cfg).to_string(index=False))
    print(f"\nmanifest written: {p['manifest']} ({len(manifest)} rows)")


def stage_infer(cfg: Config, system: str) -> None:
    from src.inference import transcribe
    p = _paths(cfg)
    if not os.path.exists(p["manifest"]):
        raise SystemExit("Run `python run.py manifest` first.")

    manifest = pd.read_csv(p["manifest"])
    splits = load_splits(cfg)
    cfg = resolve_columns(cfg, splits[next(iter(splits))])

    spec = {
        "whisper": (cfg.whisper_id, p["hyp_whisper"], "whisper_hyp"),
        "wav2vec2": (cfg.w2v_id, p["hyp_wav2vec2"], "w2v_hyp"),
    }[system]
    transcribe(cfg, splits, manifest, *spec)
    print(f"hypotheses written: {spec[1]}")


def stage_analyse(cfg: Config) -> None:
    p = _paths(cfg)
    manifest = pd.read_csv(p["manifest"])

    hypotheses = {}
    for name, path, col in [("whisper", p["hyp_whisper"], "whisper_hyp"),
                            ("wav2vec2", p["hyp_wav2vec2"], "w2v_hyp")]:
        if os.path.exists(path):
            hypotheses[name] = pd.read_csv(path)[["uid", col]]
        else:
            logging.warning("missing hypotheses for %s (%s)", name, path)
    if not hypotheses:
        raise SystemExit("No hypothesis files found. Run the inference stages.")

    normaliser = Normaliser()
    scored = R.score_all(manifest, hypotheses, normaliser, cfg)
    scored.to_csv(p["scored"], index=False)

    tables = {
        "Table V-1 Corpus composition": R.table_composition(manifest, cfg),
        "Table V-2 WER by accent group": R.table_wer(scored, cfg),
        "Table V-3 Adjusted disparity rate ratios": R.table_rate_ratios(scored, cfg),
        "Table V-4 Intersectional analysis": R.table_intersectional(scored, cfg),
        "Table V-5 Error composition": R.table_error_composition(scored, cfg),
    }
    for name, frame in tables.items():
        slug = name.split()[1].replace("-", "_").lower()
        frame.to_csv(os.path.join(cfg.output_dir, f"table_{slug}.csv"), index=False)
        print(f"\n=== {name} ===")
        print(frame.to_string(index=False) if not frame.empty else "(empty)")

    R.to_markdown(tables, p["tables_md"])
    with open(os.path.join(cfg.output_dir, "normaliser.json"), "w") as fh:
        json.dump(normaliser.describe(), fh, indent=2)
    print(f"\nall tables written to {cfg.output_dir}/ (markdown: {p['tables_md']})")


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage",
                        choices=["audit", "manifest", "infer", "analyse", "all"])
    parser.add_argument("--system", choices=["whisper", "wav2vec2"], default="whisper")
    parser.add_argument("--smoke", type=int, default=None,
                        help="cap utterances for a fast end-to-end check")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="inference batch size; halves automatically on OOM")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    cfg = Config(output_dir=args.output_dir)
    if args.dataset_id:
        cfg.dataset_id = args.dataset_id
    if args.seed is not None:
        cfg.seed = args.seed
    if args.batch_size is not None:
        cfg.batch_size = max(1, args.batch_size)
        logging.info("batch size set to %d", cfg.batch_size)
    if args.smoke:
        cfg.max_utterances = args.smoke
        logging.info("SMOKE MODE: capped at %d utterances", args.smoke)

    try:
        if args.stage == "audit":
            stage_audit(cfg)
        elif args.stage == "manifest":
            stage_manifest(cfg)
        elif args.stage == "infer":
            stage_infer(cfg, args.system)
        elif args.stage == "analyse":
            stage_analyse(cfg)
        elif args.stage == "all":
            stage_audit(cfg)
            stage_manifest(cfg)
            stage_infer(cfg, "whisper")
            stage_infer(cfg, "wav2vec2")
            stage_analyse(cfg)
    except EmptyStageError as exc:
        logging.error("STAGE PRODUCED NO OUTPUT\n%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
