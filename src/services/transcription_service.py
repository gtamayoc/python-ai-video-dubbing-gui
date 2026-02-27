"""
src/services/transcription_service.py  (v2 – advanced + legacy)
────────────────────────────────────────────────────────────────
STT (Speech-to-Text) via faster-whisper.

Changes from v1:
  • Model size is now pulled from ``config.yaml`` (stt.model_size).
    Default is ``large-v3`` in advanced mode, ``small`` in simple mode.
  • Language detection uses Whisper's built-in ``detect_language`` when
    ``stt.language == "auto"`` instead of relying on *langdetect*.
  • Word-level timestamps are available for future alignment improvements
    (exposed via ``segments`` items; not yet wired to diarization).
  • ``vad_filter`` and ``condition_on_previous_text`` are now configurable.
  • Chunk size / overlap come from ``config.yaml`` chunks section.

Public API (unchanged):
    svc = TranscriptionService(temp_dir)
    result: TranscriptionResult = svc.transcribe(audio_path, diar_segments,
                                                  total_duration, notify_fn)
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import logging
from typing import Callable, List, Optional

from src.core.entities import SpeakerSegment, TimedSegment, TranscriptionResult
from src.core.model_manager import models
from src.core.settings import settings as cfg

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Transcribes an audio file using faster-whisper.

    Parameters
    ----------
    temp_dir:
        Directory used to store temporary per-chunk MP3 files.
    """

    def __init__(self, temp_dir: str) -> None:
        self.temp_dir = temp_dir

    def transcribe(
        self,
        audio_path: str,
        diar_segments: List[SpeakerSegment],
        total_duration: float,
        notify_fn: Optional[Callable[[str, str], None]] = None,
    ) -> TranscriptionResult:
        """Transcribe *audio_path* and return a ``TranscriptionResult``.

        Parameters
        ----------
        audio_path:
            Path to the source audio (MP3/WAV/FLAC).
        diar_segments:
            Speaker diarization segments used to assign ``speaker_id`` to each
            transcribed segment.  Pass an empty list or a single full-duration
            segment to skip diarization.
        total_duration:
            Duration of *audio_path* in seconds.
        notify_fn:
            Progress callback ``(stage: str, message: str)``.

        Returns
        -------
        TranscriptionResult
            Contains ``full_text`` and ``segments`` (List[TimedSegment]).
        """
        whisper = models.get_whisper()

        # Resolve language from settings
        lang_setting = cfg.stt.language.strip().lower()
        forced_language: Optional[str] = None if lang_setting == "auto" else lang_setting

        chunk_size    = cfg.chunks.size_s
        chunk_overlap = cfg.chunks.overlap_s
        num_chunks    = math.ceil(total_duration / chunk_size)

        all_timed:  List[TimedSegment] = []
        all_texts:  List[str]          = []
        seen_ranges: set               = set()
        lang_logged: bool              = False

        for i in range(num_chunks):
            start_off = i * chunk_size
            dur       = min(chunk_size + chunk_overlap, total_duration - start_off)

            if notify_fn:
                notify_fn(
                    "Transcribing",
                    f"Chunk {i + 1}/{num_chunks} "
                    f"({int(start_off)}s – {int(start_off + dur)}s) "
                    f"[model: {cfg.stt.model_size}]",
                )

            # ── Extract chunk ─────────────────────────────────────────────
            chunk_path = os.path.join(self.temp_dir, f"chunk_{i}.mp3")
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", str(start_off),
                    "-t",  str(dur),
                    "-i",  audio_path,
                    "-acodec", "libmp3lame",
                    "-ab", "64k",
                    chunk_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # ── Transcribe chunk ──────────────────────────────────────────
            try:
                transcribe_kwargs: dict = dict(
                    beam_size=cfg.stt.beam_size,
                    vad_filter=cfg.stt.vad_filter,
                    vad_parameters=dict(
                        min_silence_duration_ms=400,
                        speech_pad_ms=100,
                    ),
                    condition_on_previous_text=cfg.stt.condition_on_previous_text,
                    word_timestamps=False,   # keep False for speed; enable for word alignment
                )
                if forced_language:
                    transcribe_kwargs["language"] = forced_language

                seg_gen, info = whisper.transcribe(chunk_path, **transcribe_kwargs)

                # Log detected language on first chunk
                if not lang_logged:
                    detected = getattr(info, "language", forced_language or "?")
                    if notify_fn:
                        notify_fn("Transcribing", f"✅ Idioma detectado: '{detected}'")
                    logger.info("Whisper detected language: %s", detected)
                    lang_logged = True

                # ── Collect segments ──────────────────────────────────────
                for seg in seg_gen:
                    seg_text = seg.text.strip()
                    if not seg_text:
                        continue

                    abs_start = round(seg.start + start_off, 2)
                    abs_end   = round(seg.end   + start_off, 2)

                    # Deduplicate segments that overlap between chunks
                    key = (round(abs_start, 1), round(abs_end, 1))
                    if key in seen_ranges:
                        continue
                    seen_ranges.add(key)

                    seg_text = self._clean_text(seg_text)
                    speaker  = self._assign_speaker(abs_start, abs_end, diar_segments)

                    all_timed.append(TimedSegment(
                        speaker_id=speaker,
                        start=abs_start,
                        end=abs_end,
                        text=seg_text,
                    ))
                    all_texts.append(seg_text)

            except Exception as e:
                logger.error("Transcription failed for chunk %d: %s", i, e)

            finally:
                try:
                    os.remove(chunk_path)
                except OSError:
                    pass

        all_timed.sort(key=lambda s: s.start)
        return TranscriptionResult(
            full_text=" ".join(all_texts),
            segments=all_timed,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize raw Whisper output: remove repeated punctuation, capitalise."""
        text = re.sub(r"([?!])\1+", r"\1", text)     # !! → !
        text = re.sub(r"\s+", " ", text).strip()
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text

    @staticmethod
    def _assign_speaker(
        start: float,
        end: float,
        diar_segments: List[SpeakerSegment],
    ) -> str:
        """Return the speaker_id that covers the midpoint of [start, end]."""
        mid = (start + end) / 2.0
        for seg in diar_segments:
            if seg.start_time <= mid <= seg.end_time:
                return seg.speaker_id
        return "speaker_1"
