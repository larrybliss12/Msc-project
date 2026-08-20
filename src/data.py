"""EdAcc loading, schema discovery, accent audit and manifest construction.

Design rule throughout: fail loudly. Every stage asserts that it produced
non-empty output before returning. The predecessor pipeline completed
successfully while producing nothing, which is the failure mode these guards
exist to prevent.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .audio_compat import get_duration
from .config import Config, STUDY_GROUPS
from .mapping import match_group, audit_dictionary

log = logging.getLogger(__name__)


class EmptyStageError(RuntimeError):
    """Raised when a stage produces no output. Never suppressed."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_splits(cfg: Config):
    """Load the configured splits, trying documented alias names.

    EdAcc's released partition names have varied. Rather than assume one, each
    candidate is attempted and the successful name recorded.
    """
    from datasets import load_dataset

    loaded, errors = {}, {}
    for want in cfg.splits:
        for candidate in cfg.split_aliases.get(want, [want]):
            try:
                loaded[want] = load_dataset(cfg.dataset_id, split=candidate)
                log.info("loaded split %r as %r (%d rows)",
                         candidate, want, len(loaded[want]))
                break
            except Exception as exc:  # noqa: BLE001
                errors[f"{want}/{candidate}"] = str(exc)[:200]

    if not loaded:
        raise EmptyStageError(
            "No splits loaded. Confirm the dataset id, that you are logged in "
            "to Hugging Face, and that the licence terms have been accepted.\n"
            + "\n".join(f"  {k}: {v}" for k, v in errors.items())
        )
    return loaded


def probe_schema(dataset) -> Dict[str, object]:
    """Report the actual column names and a sample row.

    Column roles are discovered rather than assumed, because a wrong column
    name would silently yield empty accent labels and an empty study.
    """
    row = dataset[0]
    sample = {}
    for key, value in row.items():
        if isinstance(value, dict) and "sampling_rate" in value:
            sample[key] = f"<audio sr={value.get('sampling_rate')}>"
        else:
            sample[key] = repr(value)[:120]
    return {"columns": list(row.keys()), "sample": sample}


def resolve_columns(cfg: Config, dataset) -> Config:
    """Match configured column roles against the real schema, with fallbacks."""
    available = set(dataset[0].keys())
    candidates = {
        "col_text": ["text", "sentence", "transcript", "transcription"],
        "col_speaker": ["speaker", "speaker_id", "client_id", "spk_id"],
        "col_accent": ["accent", "accents", "self_reported_accent"],
        "col_l1": ["l1", "first_language", "native_language", "mother_tongue"],
        "col_age": ["age", "age_group"],
        "col_gender": ["gender", "sex"],
        "col_audio": ["audio", "wav", "speech"],
    }
    unresolved = []
    for role, options in candidates.items():
        current = getattr(cfg, role)
        if current in available:
            continue
        found = next((o for o in options if o in available), None)
        if found:
            setattr(cfg, role, found)
            log.info("resolved %s -> %r", role, found)
        else:
            unresolved.append(role)

    for role in unresolved:
        setattr(cfg, role, "")   # absent columns read as empty, never as a default name

    if cfg.col_text not in available or cfg.col_speaker not in available:
        raise EmptyStageError(
            f"Required columns missing. Available: {sorted(available)}. "
            f"Unresolved roles: {unresolved}"
        )
    if unresolved:
        log.warning("optional columns unresolved: %s", unresolved)
    return cfg


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------

def _field(row: dict, name: str) -> str:
    value = row.get(name)
    return "" if value is None else str(value)


def audit_accents(cfg: Config, splits: Dict[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Enumerate every raw accent and first-language string, then map to groups.

    Returns (raw_frequencies, group_summary). The raw enumeration is published
    as an appendix so that the grouping is reproducible and contestable.
    """
    collisions = audit_dictionary()
    if collisions:
        log.warning("dictionary token overlaps detected (resolved by "
                    "specificity, longest token wins): %s", collisions)

    raw_accent, raw_l1 = Counter(), Counter()
    group_utts = Counter()
    group_speakers = defaultdict(set)
    group_examples = defaultdict(Counter)
    unmapped_examples = Counter()
    total, labelled, unmapped = 0, 0, 0

    for split_name, dataset in splits.items():
        for row in dataset:
            total += 1
            accent = _field(row, cfg.col_accent)
            l1 = _field(row, cfg.col_l1)
            speaker = _field(row, cfg.col_speaker)
            if accent:
                raw_accent[accent] += 1
            if l1:
                raw_l1[l1] += 1
            if accent or l1:
                labelled += 1

            group = match_group(accent, l1)
            if group:
                group_utts[group] += 1
                group_speakers[group].add(speaker)
                group_examples[group][f"{accent} | {l1}"] += 1
            elif accent or l1:
                unmapped += 1
                unmapped_examples[f"{accent} | {l1}"] += 1

    if total == 0:
        raise EmptyStageError("Audit read zero rows.")
    if labelled == 0:
        raise EmptyStageError(
            "No utterance carried an accent or first-language label. The "
            "column mapping is almost certainly wrong."
        )

    raw_rows = [{"field": "accent", "value": v, "count": c}
                for v, c in raw_accent.most_common()]
    raw_rows += [{"field": "l1", "value": v, "count": c}
                 for v, c in raw_l1.most_common()]
    raw_rows += [{"field": "unmapped", "value": v, "count": c}
                 for v, c in unmapped_examples.most_common()]
    raw_df = pd.DataFrame(raw_rows)

    summary = []
    for group in cfg.group_names:
        n_utt = group_utts.get(group, 0)
        n_spk = len(group_speakers.get(group, ()))
        viable = (n_utt >= cfg.min_utterances_for_inference
                  and n_spk >= cfg.min_speakers_for_inference)
        summary.append({
            "accent_group": group,
            "utterances": n_utt,
            "speakers": n_spk,
            "admitted_to_modelling": viable,
            "example_labels": "; ".join(
                k for k, _ in group_examples[group].most_common(3)),
        })
    summary.append({
        "accent_group": "(labelled but unmapped)",
        "utterances": unmapped, "speakers": 0,
        "admitted_to_modelling": False, "example_labels": "",
    })
    summary_df = pd.DataFrame(summary)

    if summary_df.loc[summary_df.accent_group != "(labelled but unmapped)",
                      "utterances"].sum() == 0:
        raise EmptyStageError(
            "No utterance mapped to any study group. Inspect the raw label "
            "enumeration and revise STUDY_GROUPS before proceeding."
        )
    return raw_df, summary_df


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

def build_manifest(cfg: Config, splits: Dict[str, object],
                   admitted: Optional[List[str]] = None) -> pd.DataFrame:
    """One row per mapped utterance, with metadata and duration.

    Audio arrays are not retained; only the index needed to fetch them during
    inference, so the manifest stays small enough to inspect by hand.
    """
    rows = []
    for split_name, dataset in splits.items():
        for i, row in enumerate(dataset):
            group = match_group(_field(row, cfg.col_accent), _field(row, cfg.col_l1))
            if not group or (admitted is not None and group not in admitted):
                continue
            duration = get_duration(row.get(cfg.col_audio))
            rows.append({
                "uid": f"{split_name}:{i}",
                "split": split_name,
                "row_index": i,
                "accent_group": group,
                "speaker_id": _field(row, cfg.col_speaker),
                "age": _field(row, cfg.col_age),
                "gender": _field(row, cfg.col_gender),
                "text": _field(row, cfg.col_text),
                "duration_s": duration,
            })

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise EmptyStageError("Manifest is empty; no utterances were mapped.")
    if manifest["text"].str.strip().eq("").all():
        raise EmptyStageError("Every reference transcript is empty; wrong text column.")
    return manifest
