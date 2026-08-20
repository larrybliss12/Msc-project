"""Verification suite.

These tests exist because the predecessor pipeline failed silently: an
exact-string filter against a free-text field returned an empty selection and
the run completed without error, producing a 22-byte artefact. Every test below
targets a failure mode that would otherwise be invisible in the output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mapping import canonical, match_group, audit_dictionary, unmapped_report, declared_labels
from src.scoring import (
    Normaliser, align, score_pair, token_weighted_wer,
    bootstrap_wer_by_speaker, error_composition,
)
from src.stats import fit_disparity_model, fit_intersectional


# =========================================================================
# Label normalisation
# =========================================================================

class TestCanonical:
    def test_collapses_whitespace_and_case(self):
        assert canonical("  Nigerian   English ") == "nigerian english"
        assert canonical("JAMAICAN ENGLISH") == "jamaican english"

    def test_placeholders_become_empty(self):
        for v in (None, "", "  ", "nan", "N/A", "unknown", "-"):
            assert canonical(v) == ""

    def test_does_not_strip_words(self):
        """Stripping 'english' is what caused US labels to match nothing."""
        assert "english" in canonical("Mainstream US English")


class TestMatchGroup:
    @pytest.mark.parametrize("label,expected", [
        ("Mainstream US English", "us_baseline"),
        ("Indian English", "indian"),
        ("Scottish English", "scottish"),
        ("Nigerian English", "nigerian"),
        ("Jamaican English", "jamaican"),
    ])
    def test_declared_labels_map(self, label, expected):
        assert match_group(label) == expected

    def test_latin_american_is_not_us_baseline(self):
        """Regression: 'america' is a substring of 'Latin American'.

        Under substring matching this placed 172 L1-Spanish utterances into the
        US reference group, contaminating the baseline against which every
        disparity ratio is computed.
        """
        assert match_group("Latin American") is None
        assert match_group("Latin American", "Spanish") is None

    def test_mainstream_us_english_is_recognised(self):
        """Regression: stripping 'english' left 'mainstream us', matching nothing.

        1,983 genuine US utterances went unmapped while the reference group was
        filled with Latin American speakers.
        """
        assert match_group("Mainstream US English") == "us_baseline"

    def test_l1_is_not_consulted(self):
        """Accent, not first language, determines the group."""
        assert match_group("Mainstream US English", "Urdu") == "us_baseline"
        assert match_group(None, "Yoruba") is None

    def test_undeclared_varieties_are_unmapped(self):
        for label in ("Southern British English", "Irish English",
                      "Kenyan English", "Vietnamese", "Hausa", "West Indian"):
            assert match_group(label) is None

    def test_case_and_whitespace_tolerated(self):
        assert match_group("  nigerian  english ") == "nigerian"

    def test_declared_mapping_has_no_collisions(self):
        assert audit_dictionary() == []

    def test_collision_is_detected_when_present(self):
        bad = {"Indian English": "a", "indian  english": "b"}
        assert audit_dictionary(bad) != []

    def test_unmapped_report_ranks_by_frequency(self):
        observed = {"Southern British English": 1371, "Irish English": 1317,
                    "Nigerian English": 1356}
        rows = unmapped_report(observed)
        assert rows[0][0] == "Southern British English"
        assert all(label != "Nigerian English" for label, _ in rows)

    def test_declared_labels_listed_for_appendix(self):
        assert len(declared_labels()) == 5


# =========================================================================
# Text normaliser used for scoring
# =========================================================================

class TestScoringNormaliser:
    def setup_method(self):
        self.norm = Normaliser()

    def test_removes_punctuation_and_lowercases(self):
        assert self.norm("Hello, World!") == "hello world"

    def test_expands_contractions(self):
        assert self.norm("can't") == "cannot"
        assert "not" in self.norm("don't")

    def test_removes_fillers(self):
        assert self.norm("um hello uh there") == "hello there"

    def test_is_idempotent(self):
        once = self.norm("Hello, World!")
        assert self.norm(once) == once

    def test_none_and_empty_are_safe(self):
        assert self.norm(None) == ""
        assert self.norm("") == ""

    def test_description_is_serialisable(self):
        d = self.norm.describe()
        assert d["lowercase"] is True
        assert "punctuation_removed" in d


# =========================================================================
# Edit distance
# =========================================================================

class TestAlign:
    def test_identical_sequences_have_no_errors(self):
        r = align(["the", "cat", "sat"], ["the", "cat", "sat"])
        assert r.errors == 0 and r.wer == 0.0

    def test_single_substitution(self):
        r = align(["the", "cat"], ["the", "hat"])
        assert (r.substitutions, r.deletions, r.insertions) == (1, 0, 0)

    def test_single_deletion(self):
        r = align(["the", "cat", "sat"], ["the", "sat"])
        assert r.deletions == 1 and r.errors == 1

    def test_single_insertion(self):
        r = align(["the", "cat"], ["the", "big", "cat"])
        assert r.insertions == 1 and r.errors == 1

    def test_wer_can_exceed_one(self):
        r = align(["hi"], ["a", "b", "c", "d"])
        assert r.wer > 1.0

    def test_empty_reference_is_flagged_not_scored(self):
        assert score_pair("", "anything", Normaliser()) is None
        assert score_pair("um uh", "anything", Normaliser()) is None

    def test_error_composition_sums_to_one(self):
        comp = error_composition([3], [2], [5])
        assert pytest.approx(sum(comp.values())) == 1.0


# =========================================================================
# Aggregation and uncertainty
# =========================================================================

class TestAggregation:
    def test_token_weighted_differs_from_mean_of_rates(self):
        """Short utterances must not dominate the aggregate.

        One error on a 1-word utterance and one on a 99-word utterance is
        2/100 tokens, not the 50.5% a mean of per-utterance rates would give.
        """
        errors, lens = [1, 1], [1, 99]
        assert token_weighted_wer(errors, lens) == pytest.approx(0.02)
        naive = np.mean([e / n for e, n in zip(errors, lens)])
        assert naive > 0.5

    def test_zero_length_references_are_excluded(self):
        assert token_weighted_wer([1, 5], [0, 10]) == pytest.approx(0.5)

    def test_speaker_bootstrap_is_wider_than_utterance_bootstrap(self):
        """Utterance-level resampling understates uncertainty.

        This is the central justification for clustering. When speakers differ
        from one another, treating their utterances as independent draws
        produces intervals that are too narrow, overstating precision. Speaker
        heterogeneity is built into the fixture because without it the two
        procedures are equivalent and the test would be vacuous.
        """
        rng = np.random.default_rng(0)
        speakers, errors, lens = [], [], []
        # Twelve speakers with genuinely different error rates.
        for i, rate in enumerate(np.linspace(0.05, 0.60, 12)):
            for _ in range(20):
                speakers.append(f"s{i}")
                lens.append(10.0)
                errors.append(float(rng.binomial(10, rate)))
        speakers = np.array(speakers)
        errors = np.array(errors)
        lens = np.array(lens)

        _, lo_spk, hi_spk = bootstrap_wer_by_speaker(
            speakers, errors, lens, n_iterations=800, seed=1)

        # Naive utterance-level resampling, the procedure being argued against.
        rng2 = np.random.default_rng(1)
        draws = []
        for _ in range(800):
            idx = rng2.integers(0, len(errors), len(errors))
            draws.append(errors[idx].sum() / lens[idx].sum())
        lo_utt, hi_utt = np.percentile(draws, [2.5, 97.5])

        assert (hi_spk - lo_spk) > (hi_utt - lo_utt), (
            "speaker-level interval should be wider than utterance-level "
            f"({hi_spk - lo_spk:.4f} vs {hi_utt - lo_utt:.4f})")

    def test_single_speaker_yields_no_interval(self):
        point, lo, hi = bootstrap_wer_by_speaker(
            ["s1"] * 5, [1.0] * 5, [10.0] * 5, n_iterations=100)
        assert np.isfinite(point)
        assert np.isnan(lo) and np.isnan(hi)

    def test_bootstrap_is_reproducible(self):
        args = (["a", "b", "c"] * 4, [1.0] * 12, [10.0] * 12)
        a = bootstrap_wer_by_speaker(*args, n_iterations=200, seed=42)
        b = bootstrap_wer_by_speaker(*args, n_iterations=200, seed=42)
        assert a == b


# =========================================================================
# Disparity model
# =========================================================================

def _synthetic(n_per_group=140, true_rr=2.0, seed=3, dispersion=None):
    """Generate data with a known disparity so the model can be validated."""
    rng = np.random.default_rng(seed)
    rows = []
    for group, rr in [("us_baseline", 1.0), ("indian", true_rr)]:
        for s in range(14):
            spk = f"{group}_s{s}"
            for _ in range(n_per_group // 14):
                ref_len = int(rng.integers(8, 25))
                mu = 0.10 * rr * ref_len
                err = (rng.poisson(mu) if dispersion is None
                       else rng.negative_binomial(dispersion, dispersion / (dispersion + mu)))
                rows.append({"accent_group": group, "speaker_id": spk,
                             "errors": float(err), "ref_len": float(ref_len),
                             "gender": "f" if s % 2 else "m"})
    return pd.DataFrame(rows)


class TestDisparityModel:
    def test_recovers_known_rate_ratio(self):
        df = _synthetic(true_rr=2.0)
        fit = fit_disparity_model(df, "errors", "ref_len")
        rr = fit.rate_ratio("indian")
        assert 1.7 < rr < 2.35, f"expected about 2.0, got {rr}"

    def test_parity_is_detected_as_no_disparity(self):
        df = _synthetic(true_rr=1.0)
        fit = fit_disparity_model(df, "errors", "ref_len")
        assert 0.85 < fit.rate_ratio("indian") < 1.18

    def test_offset_makes_result_invariant_to_utterance_length(self):
        """Doubling every utterance length must not change the rate ratio."""
        df = _synthetic(true_rr=2.0)
        a = fit_disparity_model(df, "errors", "ref_len").rate_ratio("indian")
        df2 = df.copy()
        df2["ref_len"] *= 2
        df2["errors"] *= 2
        b = fit_disparity_model(df2, "errors", "ref_len").rate_ratio("indian")
        assert abs(a - b) < 0.05

    def test_missing_reference_group_raises(self):
        df = _synthetic()
        df = df[df.accent_group != "us_baseline"]
        with pytest.raises(ValueError, match="Reference group"):
            fit_disparity_model(df, "errors", "ref_len")

    def test_empty_input_raises_rather_than_returning_empty(self):
        empty = pd.DataFrame(columns=["accent_group", "speaker_id", "errors", "ref_len"])
        with pytest.raises(ValueError):
            fit_disparity_model(empty, "errors", "ref_len")

    def test_overdispersion_triggers_negative_binomial(self):
        df = _synthetic(true_rr=2.0, dispersion=0.6)
        fit = fit_disparity_model(df, "errors", "ref_len")
        assert fit.pearson_chi2_dof > 1.0
        assert "negative_binomial" in fit.family or fit.pearson_chi2_dof <= 1.5

    def test_reports_sample_sizes(self):
        fit = fit_disparity_model(_synthetic(), "errors", "ref_len")
        assert fit.n_observations > 0 and fit.n_speakers > 0


class TestIntersectional:
    def test_returns_none_when_cells_are_sparse(self):
        df = _synthetic(n_per_group=28)
        assert fit_intersectional(df, "errors", "ref_len", "gender", min_cell=500) is None

    def test_fits_when_cells_are_adequate(self):
        df = _synthetic(n_per_group=280)
        fit = fit_intersectional(df, "errors", "ref_len", "gender", min_cell=10)
        assert fit is not None and not fit.table.empty


# =========================================================================
# Audio compatibility across datasets library versions
# =========================================================================

from src.audio_compat import get_duration, get_array_and_rate, to_pipeline_input


class _FakeMeta:
    def __init__(self, duration=None, frames=None, rate=None):
        if duration is not None:
            self.duration_seconds = duration
        if frames is not None:
            self.num_frames = frames
        if rate is not None:
            self.sample_rate = rate


class _FakeSamples:
    def __init__(self, data, rate):
        self.data = data
        self.sample_rate = rate


class _FakeDecoder:
    """Stands in for a datasets 4.x torchcodec AudioDecoder."""
    def __init__(self, data, rate, meta=None):
        self._data = data
        self._rate = rate
        if meta is not None:
            self.metadata = meta

    def get_all_samples(self):
        return _FakeSamples(self._data, self._rate)


class TestAudioCompat:
    def test_dict_form_duration(self):
        audio = {"array": np.zeros(16000), "sampling_rate": 16000}
        assert get_duration(audio) == pytest.approx(1.0)

    def test_decoder_duration_from_metadata_without_decoding(self):
        """Metadata path must not call get_all_samples."""
        class NoDecode(_FakeDecoder):
            def get_all_samples(self):
                raise AssertionError("should not decode when metadata is present")
        audio = NoDecode(None, None, meta=_FakeMeta(duration=2.5))
        assert get_duration(audio) == pytest.approx(2.5)

    def test_decoder_duration_from_frames_and_rate(self):
        audio = _FakeDecoder(None, None, meta=_FakeMeta(frames=32000, rate=16000))
        assert get_duration(audio) == pytest.approx(2.0)

    def test_decoder_array_extraction(self):
        audio = _FakeDecoder(np.ones((1, 8000), dtype=np.float32), 16000)
        array, sr = get_array_and_rate(audio)
        assert sr == 16000 and array.shape == (8000,)

    def test_multichannel_is_averaged_to_mono(self):
        stereo = np.stack([np.zeros(100), np.ones(100)]).astype(np.float32)
        array, _ = get_array_and_rate(_FakeDecoder(stereo, 16000))
        assert array.ndim == 1
        assert array[0] == pytest.approx(0.5)

    def test_pipeline_input_shape(self):
        item = to_pipeline_input(_FakeDecoder(np.ones((1, 100), np.float32), 16000))
        assert set(item) == {"array", "sampling_rate"}

    def test_unknown_object_returns_none_not_raises(self):
        assert get_array_and_rate(object()) is None
        assert get_duration(None) is None

    def test_header_duration_used_when_primary_is_none(self):
        """AudioStreamMetadata.duration_seconds is typed float | None."""
        class M:
            duration_seconds = None
            duration_seconds_from_header = 3.25
        class D(_FakeDecoder):
            def get_all_samples(self):
                raise AssertionError("should not decode when header duration exists")
        assert get_duration(D(None, None, meta=M())) == pytest.approx(3.25)

    def test_decoder_is_not_assumed_callable(self):
        """Regression: AudioDecoder defines no __call__; never invoke it."""
        class NotCallable(_FakeDecoder):
            pass
        d = NotCallable(np.ones((1, 16000), np.float32), 16000)
        assert get_duration(d) == pytest.approx(1.0)


# =========================================================================
# Inference memory handling
# =========================================================================

from src.inference import needs_chunking, _is_oom, WHISPER_WINDOW_S


class TestInferenceMemory:
    def test_oom_is_recognised(self):
        assert _is_oom(RuntimeError("CUDA out of memory. Tried to allocate 294.00 MiB"))
        assert not _is_oom(ValueError("some other failure"))

    def test_chunking_off_for_short_utterances(self):
        m = pd.DataFrame({"duration_s": [3.0, 8.5, 21.0]})
        assert needs_chunking(m) is False

    def test_chunking_on_when_any_utterance_is_long(self):
        m = pd.DataFrame({"duration_s": [3.0, WHISPER_WINDOW_S + 5]})
        assert needs_chunking(m) is True

    def test_missing_durations_do_not_crash(self):
        assert needs_chunking(pd.DataFrame({"x": [1]})) is False
        assert needs_chunking(pd.DataFrame({"duration_s": [None, None]})) is False


# =========================================================================
# Utterance-level inclusion criteria
# =========================================================================

from src.report import apply_inclusion_criteria
from src.config import Config as _Cfg


class TestInclusionCriteria:
    def _frame(self):
        return pd.DataFrame({"text": [
            "this reference has more than five words in it",   # keep
            "too short",                                       # drop, 2 words
            "WHO",                                             # drop, 1 word
            "<LAUGH>",                                         # drop, tag
            "a tagged <DTMF> line with plenty of words here",  # drop, tag
        ]})

    def test_short_references_excluded(self):
        out = apply_inclusion_criteria(self._frame(), _Cfg(min_reference_words=5))
        assert "too short" not in set(out["text"])
        assert "WHO" not in set(out["text"])

    def test_nonspeech_tags_excluded(self):
        out = apply_inclusion_criteria(self._frame(), _Cfg())
        assert not out["text"].str.contains("<").any()

    def test_long_clean_reference_retained(self):
        out = apply_inclusion_criteria(self._frame(), _Cfg())
        assert len(out) == 1

    def test_threshold_is_configurable(self):
        """Relaxing the threshold retains short references.

        A reference consisting only of a tag still cannot be retained: stripping
        the tag leaves zero words, so it falls below any positive threshold.
        That is correct, since there is nothing to score against.
        """
        cfg = _Cfg(min_reference_words=1, exclude_nonspeech_tags=False)
        out = apply_inclusion_criteria(self._frame(), cfg)
        assert len(out) == 4
        assert "WHO" in set(out["text"])
        assert "<LAUGH>" not in set(out["text"])

    def test_tag_stripped_before_counting_words(self):
        """A tag must not pad a short reference over the threshold."""
        df = pd.DataFrame({"text": ["<LAUGH> yes no"]})
        cfg = _Cfg(min_reference_words=5, exclude_nonspeech_tags=False)
        assert len(apply_inclusion_criteria(df, cfg)) == 0


class TestIntersectionalSparsity:
    def _frame(self, indian_male=2):
        rows = []
        for g, nm in [("us_baseline", 60), ("nigerian", 60), ("indian", indian_male)]:
            for i in range(60):
                rows.append({"accent_group": g, "speaker_id": f"{g}_{i%6}",
                             "gender": "female", "errors": 2.0, "ref_len": 10.0})
            for i in range(nm):
                rows.append({"accent_group": g, "speaker_id": f"{g}_m{i%6}",
                             "gender": "male", "errors": 2.0, "ref_len": 10.0})
        return pd.DataFrame(rows)

    def test_sparse_group_dropped_not_whole_analysis(self):
        fit = fit_intersectional(self._frame(indian_male=2), "errors", "ref_len",
                                 "gender", min_cell=20)
        assert fit is not None, "sparse group should be dropped, not abandon the fit"

    def test_returns_none_when_reference_is_sparse(self):
        df = self._frame()
        df = df[~((df.accent_group == "us_baseline") & (df.gender == "male"))]
        assert fit_intersectional(df, "errors", "ref_len", "gender",
                                  min_cell=20) is None
