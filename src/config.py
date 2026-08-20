"""Central configuration for the EdAcc accent-fairness measurement study.

Every tunable lives here so that a run is fully described by this file plus a
random seed. Nothing downstream hard-codes a parameter.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import json


# --------------------------------------------------------------------------
# Accent group definitions.
#
# EdAcc curates speaker accent into a CLOSED set of roughly 32 labels in the
# `accent` column, distinct from the free-text `raw_accent` column that records
# what the speaker actually typed. Because `accent` is curated, exact matching
# is both correct and safe, and substring matching is actively harmful.
#
# Substring matching was used initially, on the assumption that labels were
# free text as in community-donated corpora. It produced two silent failures on
# real data:
#
#   "america"  is a substring of  "Latin American"
#       -> 172 Latin American speakers (L1 Spanish) were assigned to the US
#          reference group.
#   "Mainstream US English" matched NOTHING, because normalisation stripped the
#       token "english", leaving "mainstream us", which no declared token
#       covered.
#
# The combined effect was a reference group composed entirely of L2 Spanish
# speakers while 1,983 genuine US utterances went unmapped. Every disparity
# ratio computed against that baseline was invalid. Exact matching on the
# curated column eliminates this entire class of defect.
ACCENT_TO_GROUP: Dict[str, str] = {
    "Mainstream US English": "us_baseline",
    "Indian English": "indian",
    "Scottish English": "scottish",
    "Nigerian English": "nigerian",
    "Jamaican English": "jamaican",
}

# Additional L1 English varieties present in EdAcc, available if the design is
# widened. Southern British English and Irish English are inner-circle native
# varieties and would strengthen the nativeness versus distributional-distance
# contrast by giving it more than one native comparison point.
AVAILABLE_ACCENTS_NOT_USED = {
    "Southern British English": 1371,
    "Irish English": 1317,
    "Kenyan English": 1157,
    "Ghanain English": 357,
    "Indonesian English": 324,
    "South African English": 246,
}

REFERENCE_GROUP = "us_baseline"

# Derived: group name -> the curated accent labels feeding it. Reported in the
# paper appendix as the declared mapping.
STUDY_GROUPS: Dict[str, List[str]] = {}
for _label, _group in ACCENT_TO_GROUP.items():
    STUDY_GROUPS.setdefault(_group, []).append(_label)


@dataclass
class Config:
    # ---- corpus -----------------------------------------------------------
    dataset_id: str = "edinburghcstr/edacc"
    splits: tuple = ("validation", "test")
    # Fallback names tried if the primary split name is rejected by the loader.
    split_aliases: Dict[str, List[str]] = field(default_factory=lambda: {
        "validation": ["validation", "dev", "valid"],
        "test": ["test"],
    })

    # ---- column roles -----------------------------------------------------
    # EdAcc's exact column names are discovered at runtime by audit.probe_schema
    # and written back here. These are defaults, not assumptions.
    col_text: str = "text"
    col_speaker: str = "speaker"
    col_accent: str = "accent"
    col_l1: str = "l1"
    col_age: str = "age"          # absent from EdAcc; intersectional is gender-only
    col_gender: str = "gender"
    col_audio: str = "audio"

    # ---- models -----------------------------------------------------------
    whisper_id: str = "openai/whisper-large-v3"
    w2v_id: str = "facebook/wav2vec2-large-960h-lv60-self"
    batch_size: int = 8
    target_sample_rate: int = 16_000

    # ---- experiment scope -------------------------------------------------
    # None = full corpus. Set to an integer for a smoke run.
    max_utterances: Optional[int] = None

    # ---- utterance-level inclusion ----------------------------------------
    # Very short references make word error rate unstable: a single error on a
    # one-word reference yields a WER of 1.0, and encoder-decoder models are
    # prone to hallucinatory repetition loops on sub-second audio, producing
    # hundreds of insertions against a one-token reference. Both effects are
    # properties of the utterance rather than of the accent, and both fall
    # unevenly across groups, so they are excluded by a stated criterion.
    min_reference_words: int = 5
    # EdAcc marks non-speech events such as <LAUGH> and <DTMF> inline. These are
    # annotations, not transcription targets, and cannot be meaningfully scored.
    exclude_nonspeech_tags: bool = True

    # ---- viability --------------------------------------------------------
    # A group below either threshold is reported descriptively but excluded
    # from inferential modelling, and the exclusion is stated in the paper.
    min_utterances_for_inference: int = 100
    min_speakers_for_inference: int = 3

    # ---- statistics -------------------------------------------------------
    seed: int = 17
    bootstrap_iterations: int = 2000
    # Pearson chi2/df above this triggers the negative binomial fallback.
    overdispersion_threshold: float = 1.5

    # ---- paths ------------------------------------------------------------
    output_dir: str = "outputs"

    def to_json(self, path: str) -> None:
        payload = asdict(self)
        payload["accent_to_group"] = ACCENT_TO_GROUP
        payload["study_groups"] = STUDY_GROUPS
        payload["reference_group"] = REFERENCE_GROUP
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)

    @property
    def group_names(self) -> List[str]:
        """Reference group first, so it is the model's baseline category."""
        return [REFERENCE_GROUP] + [g for g in STUDY_GROUPS if g != REFERENCE_GROUP]
