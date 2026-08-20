"""Audio access compatible with datasets 2.x and 4.x.

`datasets` changed the return type of an Audio-typed column. Older versions
return a plain dict with "array" and "sampling_rate". Version 4 returns a
torchcodec AudioDecoder object. Every access to audio goes through this module
so the pipeline is indifferent to which is installed.

Duration is obtained from decoder metadata where available, which avoids
decoding the waveform entirely. For a manifest over roughly nineteen thousand
utterances this is the difference between minutes and hours.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

_warned = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(message)


def get_duration(audio) -> Optional[float]:
    """Duration in seconds, without decoding the waveform where possible."""
    if audio is None:
        return None

    # datasets 2.x: plain dict
    if isinstance(audio, dict):
        array, sr = audio.get("array"), audio.get("sampling_rate")
        if array is not None and sr:
            return len(array) / sr
        return None

    # datasets 4.x: AudioDecoder. Metadata first, no decode required.
    # AudioStreamMetadata inherits duration_seconds from StreamMetadata, where
    # it is a dataclass field typed float | None, so it may legitimately be
    # None. duration_seconds_from_header is the cheaper header-derived value.
    meta = getattr(audio, "metadata", None)
    if meta is not None:
        for attr in ("duration_seconds", "duration_seconds_from_header"):
            seconds = getattr(meta, attr, None)
            if seconds:
                return float(seconds)
        frames = getattr(meta, "num_frames", None)
        rate = getattr(meta, "sample_rate", None)
        if frames and rate:
            return float(frames) / float(rate)

    # Last resort: decode.
    pair = get_array_and_rate(audio)
    if pair is None:
        return None
    array, sr = pair
    return len(array) / sr if sr else None


def get_array_and_rate(audio) -> Optional[Tuple[np.ndarray, int]]:
    """Mono float waveform and sample rate, or None if unavailable.

    Multi-channel audio is averaged to mono, since both evaluated systems
    expect single-channel input at 16 kHz.
    """
    if audio is None:
        return None

    # datasets 2.x
    if isinstance(audio, dict):
        array, sr = audio.get("array"), audio.get("sampling_rate")
        if array is None or not sr:
            return None
        array = np.asarray(array, dtype=np.float32)
        if array.ndim > 1:
            array = array.mean(axis=0)
        return array, int(sr)

    # datasets 4.x AudioDecoder
    getter = getattr(audio, "get_all_samples", None)
    if callable(getter):
        try:
            samples = getter()
        except Exception as exc:  # noqa: BLE001
            _warn_once("decode_fail", f"AudioDecoder decode failed: {exc}")
            return None
        data = getattr(samples, "data", None)
        sr = getattr(samples, "sample_rate", None)
        if data is None or not sr:
            return None
        array = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
        array = np.asarray(array, dtype=np.float32)
        if array.ndim > 1:
            # torchcodec returns (channels, samples)
            array = array.mean(axis=0)
        return array, int(sr)

    # Unknown object: try attribute access before giving up.
    array = getattr(audio, "array", None)
    sr = getattr(audio, "sampling_rate", None) or getattr(audio, "sample_rate", None)
    if array is not None and sr:
        array = np.asarray(array, dtype=np.float32)
        if array.ndim > 1:
            array = array.mean(axis=0)
        return array, int(sr)

    _warn_once(
        "unknown_audio",
        f"Unrecognised audio object of type {type(audio).__name__}; "
        "audio cannot be read. Report this type so support can be added.",
    )
    return None


def to_pipeline_input(audio) -> Optional[dict]:
    """Shape audio for a transformers ASR pipeline call."""
    pair = get_array_and_rate(audio)
    if pair is None:
        return None
    array, sr = pair
    return {"array": array, "sampling_rate": sr}
