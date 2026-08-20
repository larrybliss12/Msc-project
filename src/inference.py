"""ASR inference for both evaluated systems.

Three memory behaviours are handled explicitly, because a mid-sized GPU such as
a Colab T4 will otherwise exhaust its memory on Whisper Large V3:

1. Half precision is requested through whichever keyword the installed
   transformers version accepts. The keyword was renamed from `torch_dtype` to
   `dtype`; passing the deprecated name can silently load the model in fp32,
   which for a 1.55B-parameter model is roughly 6.2 GB of weights rather than
   3.1 GB, before any activations.

2. Batch size backs off on out-of-memory rather than collapsing to per-item.
   Halving and retrying preserves most of the throughput; per-item is the last
   resort, not the first response.

3. The CUDA cache is released after any failure, so a transient spike does not
   permanently fragment the allocator.

Hypotheses are checkpointed to disk as they are produced, so an interrupted run
resumes rather than restarting.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .audio_compat import to_pipeline_input
from .config import Config
from .data import EmptyStageError

log = logging.getLogger(__name__)

# Whisper's feature extractor pads every input to 30 s. Utterances longer than
# this are truncated unless chunking is enabled, which would silently discard
# speech and inflate the measured error rate.
WHISPER_WINDOW_S = 30.0


def _device_index() -> int:
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def _empty_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _is_oom(exc: Exception) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def build_pipeline(model_id: str, cfg: Config, needs_chunking: bool = False):
    """Construct a transformers ASR pipeline in half precision where possible.

    The dtype keyword differs across transformers versions. Passing the wrong
    one does not raise; it warns and may load full precision, which is the
    difference between fitting on a T4 and not. Both spellings are attempted.
    """
    from transformers import pipeline

    device = _device_index()
    base = {"model": model_id, "device": device, "batch_size": cfg.batch_size}

    # Chunking is enabled only when the corpus needs it. It triggers an
    # experimental code path for seq2seq models and increases memory use.
    if needs_chunking and "whisper" in model_id.lower():
        base["chunk_length_s"] = WHISPER_WINDOW_S
        log.info("long-form chunking enabled")

    if device < 0:
        log.warning("no GPU detected; inference will be very slow")
        return pipeline("automatic-speech-recognition", **base)

    import torch
    for keyword in ("dtype", "torch_dtype"):
        try:
            asr = pipeline("automatic-speech-recognition",
                           **{keyword: torch.float16}, **base)
            log.info("pipeline built with %s=float16", keyword)
            return asr
        except TypeError as exc:
            log.debug("%s rejected: %s", keyword, exc)
        except Exception as exc:  # noqa: BLE001
            log.warning("half precision via %s failed: %s", keyword, exc)
            break

    log.warning("falling back to full precision; memory use will be higher")
    return pipeline("automatic-speech-recognition", **base)


def _iter_audio(splits: Dict[str, object], manifest: pd.DataFrame,
                audio_col: str) -> Iterable[Tuple[str, dict]]:
    """Yield (uid, pipeline_input) in manifest order, skipping unreadable audio."""
    for split_name, group in manifest.groupby("split", sort=False):
        dataset = splits[split_name]
        for _, row in group.iterrows():
            item = to_pipeline_input(dataset[int(row["row_index"])].get(audio_col))
            if item is not None:
                yield row["uid"], item


def needs_chunking(manifest: pd.DataFrame) -> bool:
    """Report whether any utterance exceeds Whisper's 30 s window."""
    if "duration_s" not in manifest.columns:
        return False
    durations = manifest["duration_s"].dropna()
    if durations.empty:
        return False
    long_ones = int((durations > WHISPER_WINDOW_S).sum())
    if long_ones:
        log.warning("%d of %d utterances exceed %.0fs; enabling chunking so "
                    "they are not truncated", long_ones, len(durations),
                    WHISPER_WINDOW_S)
        return True
    log.info("all utterances within the %.0fs window; chunking not required",
             WHISPER_WINDOW_S)
    return False


def transcribe(
    cfg: Config,
    splits: Dict[str, object],
    manifest: pd.DataFrame,
    model_id: str,
    output_path: str,
    hypothesis_column: str,
) -> pd.DataFrame:
    """Transcribe every manifest utterance, resuming from partial output."""
    done: Dict[str, str] = {}
    if os.path.exists(output_path):
        prior = pd.read_csv(output_path)
        done = dict(zip(prior["uid"], prior[hypothesis_column].fillna("")))
        log.info("resuming: %d utterances already transcribed", len(done))

    todo = manifest[~manifest["uid"].isin(done)]
    if cfg.max_utterances:
        todo = todo.head(cfg.max_utterances)
    log.info("transcribing %d utterances with %s", len(todo), model_id)

    def _save() -> None:
        pd.DataFrame({"uid": list(done),
                      hypothesis_column: list(done.values())}
                     ).to_csv(output_path, index=False)

    if not todo.empty:
        asr = build_pipeline(model_id, cfg, needs_chunking=needs_chunking(todo))
        gen_kwargs = ({"language": "en", "task": "transcribe"}
                      if "whisper" in model_id.lower() else None)

        def run(items: List[dict]):
            return (asr(items, generate_kwargs=gen_kwargs) if gen_kwargs
                    else asr(items))

        state = {"batch": max(1, cfg.batch_size)}
        buf_uid: List[str] = []
        buf_audio: List[dict] = []
        processed, started, last_report = 0, time.time(), 0

        def flush() -> int:
            """Transcribe the buffer, halving the batch on OOM until it fits."""
            if not buf_audio:
                return 0
            size = len(buf_audio)
            start = 0
            while start < size:
                span = min(state["batch"], size - start)
                chunk_uid = buf_uid[start:start + span]
                chunk_audio = buf_audio[start:start + span]
                try:
                    outputs = run(chunk_audio)
                except Exception as exc:  # noqa: BLE001
                    _empty_cache()
                    if _is_oom(exc) and state["batch"] > 1:
                        state["batch"] = max(1, state["batch"] // 2)
                        log.warning("out of memory; reducing batch size to %d "
                                    "and retrying", state["batch"])
                        continue
                    if _is_oom(exc):
                        log.error("out of memory at batch size 1; recording "
                                  "empty for %s", chunk_uid[0])
                        outputs = [{"text": ""}]
                    else:
                        log.error("batch failed (%s); recording empty", exc)
                        outputs = [{"text": ""}] * span
                for uid, out in zip(chunk_uid, outputs):
                    done[uid] = (out or {}).get("text", "")
                start += span
            n = len(buf_uid)
            buf_uid.clear()
            buf_audio.clear()
            return n

        for uid, item in _iter_audio(splits, todo, cfg.col_audio):
            buf_uid.append(uid)
            buf_audio.append(item)
            if len(buf_audio) >= state["batch"]:
                processed += flush()
                _save()
                if processed - last_report >= 200:
                    last_report = processed
                    rate = processed / max(time.time() - started, 1e-6)
                    remaining = (len(todo) - processed) / max(rate, 1e-6)
                    log.info("  %d/%d done (%.1f/s, ~%.0f min remaining)",
                             processed, len(todo), rate, remaining / 60)
        processed += flush()
        _save()
        log.info("finished: batch size settled at %d, %d utterances in %.0fs",
                 state["batch"], processed, time.time() - started)

    result = pd.DataFrame({"uid": list(done), hypothesis_column: list(done.values())})
    if result.empty:
        raise EmptyStageError(f"No hypotheses produced for {model_id}.")
    blank = result[hypothesis_column].fillna("").astype(str).str.strip().eq("")
    if blank.all():
        raise EmptyStageError(
            f"Every hypothesis from {model_id} is empty; the model is not "
            "receiving audio correctly."
        )
    if blank.any():
        log.warning("%d of %d hypotheses are empty; retained and scored as "
                    "complete deletions", int(blank.sum()), len(result))
    result.to_csv(output_path, index=False)
    return result
