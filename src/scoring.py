"""Word error rate computation and uncertainty estimation.

Two methodological commitments are enforced here rather than left to the caller:

1. A single shared text normaliser is applied to every reference and every
   hypothesis, for every system. Scoring two systems under different
   normalisation regimes would make any comparison an artefact of text
   processing rather than a measurement of recognition quality.

2. Bootstrap resampling is performed at the SPEAKER level, not the utterance
   level. Utterances from one speaker are correlated; resampling utterances
   independently would yield intervals that are too narrow and would overstate
   the precision of every reported estimate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Text normalisation
# --------------------------------------------------------------------------

_CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'ve": " have", "'ll": " will",
    "'d": " would", "'m": " am",
}
_FILLERS = {"uh", "um", "erm", "eh", "mm", "hmm", "mhm", "uhhuh", "ah"}
_PUNCT = re.compile(r"[^\w\s']")
_WS = re.compile(r"\s+")


class Normaliser:
    """Deterministic English text normaliser.

    Implemented explicitly rather than imported so that the exact transformation
    applied is auditable and reportable in the paper, and so that results do not
    silently change when an upstream library revises its normaliser.
    """

    def __init__(self, remove_fillers: bool = True):
        self.remove_fillers = remove_fillers

    def __call__(self, text: Optional[str]) -> str:
        if text is None:
            return ""
        s = str(text).lower().strip()
        for src, dst in _CONTRACTIONS.items():
            s = s.replace(src, dst)
        s = _PUNCT.sub(" ", s)
        s = _WS.sub(" ", s).strip()
        if self.remove_fillers and s:
            s = " ".join(t for t in s.split() if t not in _FILLERS)
        return s

    def describe(self) -> Dict[str, object]:
        """Machine-readable description for the reproducibility appendix."""
        return {
            "lowercase": True,
            "contraction_expansion": sorted(_CONTRACTIONS),
            "punctuation_removed": True,
            "filler_words_removed": sorted(_FILLERS) if self.remove_fillers else [],
        }


# --------------------------------------------------------------------------
# Edit distance
# --------------------------------------------------------------------------

@dataclass
class ErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    ref_len: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        return float("nan") if self.ref_len == 0 else self.errors / self.ref_len


def align(ref_tokens: Sequence[str], hyp_tokens: Sequence[str]) -> ErrorCounts:
    """Levenshtein alignment at word level returning the error breakdown.

    The breakdown matters beyond the aggregate rate: a substitution-dominant
    profile indicates acoustic-phonetic mismatch, whereas a deletion-dominant
    profile suggests segmentation or decoding failure, and the two imply
    different remedies.
    """
    n, m = len(ref_tokens), len(hyp_tokens)
    if n == 0:
        return ErrorCounts(0, 0, m, 0)

    # dp[i][j] = (cost, subs, dels, ins)
    dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            c_sub, s, d, ins = dp[i - 1][j - 1]
            sub = (c_sub + 1, s + 1, d, ins)
            c_del, s2, d2, i2 = dp[i - 1][j]
            dele = (c_del + 1, s2, d2 + 1, i2)
            c_ins, s3, d3, i3 = dp[i][j - 1]
            inse = (c_ins + 1, s3, d3, i3 + 1)
            dp[i][j] = min(sub, dele, inse, key=lambda t: t[0])

    _, subs, dels, ins = dp[n][m]
    return ErrorCounts(subs, dels, ins, n)


def score_pair(reference: str, hypothesis: str, normaliser: Normaliser) -> Optional[ErrorCounts]:
    """Score one utterance. Returns None if the reference normalises to empty.

    WER is undefined with an empty reference, so such utterances are excluded
    and counted rather than scored as zero or as total error.
    """
    ref = normaliser(reference).split()
    hyp = normaliser(hypothesis).split()
    if not ref:
        return None
    return align(ref, hyp)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def token_weighted_wer(errors: Sequence[float], ref_lens: Sequence[float]) -> float:
    """Total errors over total reference tokens.

    Preferred over the mean of per-utterance rates, which over-weights short
    utterances where a single error produces an extreme rate.
    """
    e = np.asarray(errors, dtype=float)
    n = np.asarray(ref_lens, dtype=float)
    mask = np.isfinite(e) & np.isfinite(n) & (n > 0)
    if not mask.any():
        return float("nan")
    return float(e[mask].sum() / n[mask].sum())


def bootstrap_wer_by_speaker(
    speakers: Sequence[str],
    errors: Sequence[float],
    ref_lens: Sequence[float],
    n_iterations: int = 2000,
    seed: int = 17,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Point estimate and percentile interval, resampling speakers with replacement.

    Speakers are the independent sampling unit. Resampling utterances instead
    would treat correlated observations as independent and produce intervals
    that are too narrow.
    """
    speakers = np.asarray(speakers)
    e = np.asarray(errors, dtype=float)
    n = np.asarray(ref_lens, dtype=float)

    mask = np.isfinite(e) & np.isfinite(n) & (n > 0)
    speakers, e, n = speakers[mask], e[mask], n[mask]
    if len(e) == 0:
        return float("nan"), float("nan"), float("nan")

    point = float(e.sum() / n.sum())

    unique = np.unique(speakers)
    if len(unique) < 2:
        # A single speaker gives no between-speaker variance to resample.
        return point, float("nan"), float("nan")

    index_by_speaker = {s: np.flatnonzero(speakers == s) for s in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_iterations, dtype=float)
    for k in range(n_iterations):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_speaker[s] for s in picked])
        denom = n[idx].sum()
        draws[k] = e[idx].sum() / denom if denom > 0 else np.nan

    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def error_composition(
    subs: Sequence[float], dels: Sequence[float], ins: Sequence[float]
) -> Dict[str, float]:
    """Proportion of each error type, diagnostic of failure mode."""
    s, d, i = (float(np.nansum(x)) for x in (subs, dels, ins))
    total = s + d + i
    if total == 0:
        return {"substitution": float("nan"), "deletion": float("nan"),
                "insertion": float("nan")}
    return {"substitution": s / total, "deletion": d / total, "insertion": i / total}
