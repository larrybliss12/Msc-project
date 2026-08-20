"""Produce the tables the paper reports, in the form the paper reports them.

Each function corresponds to one numbered table in Section V, so that writing
the results section is transcription rather than reinterpretation.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config, REFERENCE_GROUP
from .scoring import (
    Normaliser, score_pair, token_weighted_wer,
    bootstrap_wer_by_speaker, error_composition,
)
from .stats import fit_disparity_model, fit_intersectional

log = logging.getLogger(__name__)

SYSTEMS = {"whisper": "whisper_hyp", "wav2vec2": "w2v_hyp"}

_TAG = re.compile(r"<[^>]*>")


def apply_inclusion_criteria(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Apply the stated utterance-level inclusion criteria, reporting exclusions.

    Returns the retained rows. The counts are logged and belong in the paper's
    inclusion criteria rather than being applied silently.
    """
    n0 = len(df)
    text = df["text"].astype(str)

    tagged = text.str.contains(_TAG) if cfg.exclude_nonspeech_tags else pd.Series(
        False, index=df.index)
    cleaned = text.str.replace(_TAG, " ", regex=True)
    short = cleaned.str.split().str.len() < cfg.min_reference_words

    keep = ~(tagged | short)
    log.info("inclusion: %d of %d utterances retained (%.1f%%)",
             int(keep.sum()), n0, 100 * keep.sum() / max(n0, 1))
    log.info("  excluded %d shorter than %d reference words",
             int(short.sum()), cfg.min_reference_words)
    if cfg.exclude_nonspeech_tags:
        log.info("  excluded %d containing non-speech annotations",
                 int(tagged.sum()))
    return df[keep].copy()


def score_all(manifest: pd.DataFrame, hypotheses: Dict[str, pd.DataFrame],
              normaliser: Optional[Normaliser] = None,
              cfg: Optional[Config] = None) -> pd.DataFrame:
    """Merge hypotheses onto the manifest and score every utterance.

    One normaliser instance is used for every reference and every hypothesis of
    every system, which is the control that makes cross-system comparison valid.
    """
    normaliser = normaliser or Normaliser()
    df = manifest.copy()
    df = apply_inclusion_criteria(df, cfg) if cfg is not None else df
    for system, frame in hypotheses.items():
        df = df.merge(frame, on="uid", how="inner")

    for system, column in SYSTEMS.items():
        if column not in df.columns:
            continue
        subs, dels, ins, reflen, errs = [], [], [], [], []
        for _, row in df.iterrows():
            counts = score_pair(row["text"], row.get(column, ""), normaliser)
            if counts is None:
                subs.append(np.nan); dels.append(np.nan); ins.append(np.nan)
                reflen.append(np.nan); errs.append(np.nan)
            else:
                subs.append(counts.substitutions); dels.append(counts.deletions)
                ins.append(counts.insertions); reflen.append(counts.ref_len)
                errs.append(counts.errors)
        df[f"{system}_sub"] = subs
        df[f"{system}_del"] = dels
        df[f"{system}_ins"] = ins
        df[f"{system}_reflen"] = reflen
        df[f"{system}_err"] = errs

    scored = df.dropna(subset=[f"{s}_err" for s in SYSTEMS if f"{s}_err" in df])
    log.info("scored %d utterances (%d excluded for empty reference)",
             len(scored), len(df) - len(scored))
    return df


def table_composition(manifest: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Table V-1: realised corpus composition after mapping."""
    rows = []
    for group in cfg.group_names:
        sub = manifest[manifest.accent_group == group]
        minutes = sub["duration_s"].dropna().sum() / 60 if "duration_s" in sub else np.nan
        rows.append({
            "accent_group": group,
            "utterances": len(sub),
            "speakers": sub["speaker_id"].nunique(),
            "minutes": round(minutes, 1) if np.isfinite(minutes) else np.nan,
            "admitted_to_modelling": (
                len(sub) >= cfg.min_utterances_for_inference
                and sub["speaker_id"].nunique() >= cfg.min_speakers_for_inference),
        })
    return pd.DataFrame(rows)


def table_wer(scored: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Table V-2: WER per group per system, with speaker-level intervals."""
    rows = []
    for system in SYSTEMS:
        err_col, len_col = f"{system}_err", f"{system}_reflen"
        if err_col not in scored.columns:
            continue
        base = scored[scored.accent_group == REFERENCE_GROUP]
        base_wer = token_weighted_wer(base[err_col], base[len_col])

        for group in cfg.group_names:
            sub = scored[scored.accent_group == group]
            if sub.empty:
                continue
            point, lo, hi = bootstrap_wer_by_speaker(
                sub["speaker_id"].values, sub[err_col].values, sub[len_col].values,
                n_iterations=cfg.bootstrap_iterations, seed=cfg.seed)
            rows.append({
                "system": system,
                "accent_group": group,
                "utterances": int(sub[err_col].notna().sum()),
                "speakers": sub["speaker_id"].nunique(),
                "WER": round(point, 4),
                "CI_low": round(lo, 4) if np.isfinite(lo) else np.nan,
                "CI_high": round(hi, 4) if np.isfinite(hi) else np.nan,
                "disparity_vs_baseline": (
                    round(point / base_wer, 2)
                    if base_wer and np.isfinite(base_wer) and base_wer > 0 else np.nan),
            })
    return pd.DataFrame(rows)


def table_rate_ratios(scored: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Table V-3: adjusted disparity rate ratios from the fitted model."""
    frames = []
    for system in SYSTEMS:
        err_col, len_col = f"{system}_err", f"{system}_reflen"
        if err_col not in scored.columns:
            continue
        try:
            fit = fit_disparity_model(
                scored, err_col, len_col,
                reference_group=REFERENCE_GROUP,
                overdispersion_threshold=cfg.overdispersion_threshold)
        except ValueError as exc:
            log.warning("model failed for %s: %s", system, exc)
            continue
        table = fit.table.reset_index().rename(columns={"index": "term"})
        table.insert(0, "system", system)
        table["family"] = fit.family
        table["pearson_chi2_dof"] = round(fit.pearson_chi2_dof, 3)
        table["n_observations"] = fit.n_observations
        table["n_speakers"] = fit.n_speakers
        frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def table_intersectional(scored: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Table V-4: accent-by-demographic interactions, where cells permit."""
    frames = []
    for system in SYSTEMS:
        err_col, len_col = f"{system}_err", f"{system}_reflen"
        if err_col not in scored.columns:
            continue
        for covariate in ("age", "gender"):
            if covariate not in scored.columns:
                continue
            fit = fit_intersectional(scored, err_col, len_col, covariate,
                                     reference_group=REFERENCE_GROUP)
            if fit is None:
                frames.append(pd.DataFrame([{
                    "system": system, "covariate": covariate, "term": "(not fitted)",
                    "note": "cells too sparse for stable interaction estimates",
                }]))
                continue
            table = fit.table.reset_index().rename(columns={"index": "term"})
            table.insert(0, "covariate", covariate)
            table.insert(0, "system", system)
            table["note"] = ""
            frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def table_error_composition(scored: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Table V-5: proportion of substitutions, deletions and insertions."""
    rows = []
    for system in SYSTEMS:
        if f"{system}_sub" not in scored.columns:
            continue
        for group in cfg.group_names:
            sub = scored[scored.accent_group == group]
            if sub.empty:
                continue
            comp = error_composition(sub[f"{system}_sub"], sub[f"{system}_del"],
                                     sub[f"{system}_ins"])
            rows.append({"system": system, "accent_group": group,
                         **{k: round(v, 3) for k, v in comp.items()}})
    return pd.DataFrame(rows)


def to_markdown(tables: Dict[str, pd.DataFrame], path: str) -> None:
    """Write every table as markdown for direct paste into the paper."""
    with open(path, "w") as fh:
        for name, frame in tables.items():
            fh.write(f"\n\n## {name}\n\n")
            if frame is None or frame.empty:
                fh.write("_(empty)_\n")
            else:
                fh.write(frame.to_markdown(index=False))
