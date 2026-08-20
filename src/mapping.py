"""Accent label mapping for EdAcc.

EdAcc curates speaker accent into a closed set in the `accent` column. Mapping
is therefore EXACT: a label either is one of the study's declared accent labels
or it is not. There is no fuzzy matching, no substring containment and no
tokenisation.

This replaces an earlier substring-matching implementation written on the
assumption that labels were free text. That assumption was wrong for this
corpus and the consequences were severe and silent, documented in config.py:
Latin American speakers were absorbed into the US reference group while genuine
US utterances went unmapped, invalidating every disparity ratio computed
against that baseline.

The lesson generalises. Fuzzy matching on a categorical field trades a loud
failure (label not recognised) for a silent one (label recognised as the wrong
thing). Where the field is closed, exact matching is the correct choice.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .config import ACCENT_TO_GROUP


def canonical(raw: Optional[str]) -> str:
    """Whitespace-and-case canonical form used for exact comparison.

    Deliberately minimal. It does NOT strip words, split compounds or perform
    any semantic transformation, because every such transformation is an
    opportunity for one label to be silently read as another.

    >>> canonical("  Nigerian   English ")
    'nigerian english'
    >>> canonical(None)
    ''
    """
    if raw is None:
        return ""
    text = " ".join(str(raw).split()).strip().casefold()
    return "" if text in {"nan", "none", "n/a", "unknown", "-", ""} else text


_LOOKUP: Dict[str, str] = {canonical(k): v for k, v in ACCENT_TO_GROUP.items()}


def match_group(
    accent: Optional[str],
    l1: Optional[str] = None,
    groups: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Assign an utterance to a study group by exact accent label, else None.

    The first-language field is accepted for signature compatibility but is NOT
    consulted. EdAcc's `accent` column already encodes the English variety; L1
    encodes the speaker's first language, which is a different construct. A
    speaker whose accent is curated as Mainstream US English belongs in the US
    group whatever their first language, because this study measures accent.

    Returns None for any label outside the declared mapping, so unmapped
    utterances are counted and reported rather than absorbed by a near match.
    """
    lookup = ({canonical(k): v for k, v in groups.items()}
              if groups is not None else _LOOKUP)
    return lookup.get(canonical(accent))


def declared_labels() -> List[str]:
    """The accent labels this study claims, for the appendix."""
    return sorted(ACCENT_TO_GROUP)


def audit_dictionary(groups: Optional[Dict[str, str]] = None) -> List[tuple]:
    """Check the declared mapping is well formed.

    With exact matching the only possible defect is two labels differing solely
    by case or whitespace, which would collide after canonicalisation. Returns
    an empty list when the mapping is sound.
    """
    mapping = groups if groups is not None else ACCENT_TO_GROUP
    seen: Dict[str, str] = {}
    collisions = []
    for label, group in mapping.items():
        key = canonical(label)
        if not key:
            collisions.append((label, group, "canonicalises to empty string"))
            continue
        if key in seen and seen[key] != label:
            collisions.append((seen[key], label, f"both canonicalise to {key!r}"))
        seen[key] = label
    return collisions


def unmapped_report(observed: Dict[str, int], top: int = 20) -> List[tuple]:
    """Accent labels present in the data but outside the declared mapping.

    Reviewed before every run. A large or unexpected entry here means the study
    is excluding data it may intend to include, which is a decision to be made
    explicitly rather than by omission.
    """
    rows = [(label, count) for label, count in observed.items()
            if canonical(label) not in _LOOKUP]
    return sorted(rows, key=lambda r: -r[1])[:top]
