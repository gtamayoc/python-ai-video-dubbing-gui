"""
src/services/tts_service.py  (v2 – edge-tts + XTTS v2 + voice cloning)
────────────────────────────────────────────────────────────────────────
Text-to-Speech service with two backends:

  "edge"  – Microsoft edge-tts (free, fast, no GPU, good quality)
  "xtts"  – Coqui XTTS v2   (voice cloning, GPU recommended)

Both backends respect the ``alignment`` settings from ``config.yaml`` and
use ffmpeg atempo to stretch/compress each segment to match the original
timing.

Public API (unchanged from v1):
    svc = TTSService(temp_dir)
    assembled_wav = svc.generate_and_assemble(segments, total_duration,
                                              notify_fn, max_threads)
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple

from src.core.entities import TimedSegment
from src.core.settings import settings as cfg
from src.adapters.media_adapter import ffmpeg_atempo

logger = logging.getLogger(__name__)


class TTSService:
    """Generates per-segment TTS audio and assembles them into a timeline.

    Parameters
    ----------
    temp_dir:
        Working directory for temporary segment files.
    source_audio_path:
        Original source audio (used only when voice cloning is enabled).
    """

    def __init__(
        self,
        temp_dir: str,
        source_audio_path: Optional[str] = None,
    ) -> None:
        self.temp_dir          = temp_dir
        self.source_audio_path = source_audio_path

        # Cache extracted reference clips per speaker_id (voice cloning)
        self._speaker_refs: dict = {}

    # ── Public ────────────────────────────────────────────────────────────────

    def generate_and_assemble(
        self,
        segments: List[TimedSegment],
        total_duration: float,
        notify_fn: Optional[Callable[[str, str], None]] = None,
        max_threads: int = 8,
    ) -> str:
        """Generate TTS for each segment and assemble into a full WAV file.

        Returns
        -------
        str
            Absolute path to the assembled output WAV.
        """
        backend = cfg.tts.backend
        if notify_fn:
            notify_fn("TTS", f"Generando voces con backend '{backend}'…")

        # When XTTS is requested but unavailable, fall back to edge gracefully
        if backend == "xtts":
            from src.core.model_manager import models
            xtts = models.get_xtts()
            if xtts is None:
                logger.warning("XTTS not available — falling back to edge-tts")
                backend = "edge"

        # If voice cloning is enabled, extract reference clips now
        if backend == "xtts" and cfg.tts.use_voice_cloning and self.source_audio_path:
            self._extract_speaker_refs(segments, notify_fn)

        processed_dir = os.path.join(self.temp_dir, "segments_tts")
        os.makedirs(processed_dir, exist_ok=True)

        # ── Determine next-segment gaps for overflow logic ────────────────────
        next_gaps: List[float] = []
        for i in range(len(segments)):
            if i < len(segments) - 1:
                next_gaps.append(segments[i + 1].start - segments[i].end)
            else:
                next_gaps.append(total_duration - segments[i].end)

        # ── Parallel TTS generation ───────────────────────────────────────────
        chunk_files: List[Tuple[float, str]] = []

        if backend == "edge":
            # edge-tts uses asyncio internally; safest to keep work sequential
            # per worker in a thread pool to avoid event-loop conflicts
            with ThreadPoolExecutor(max_workers=max_threads) as pool:
                futures = [
                    pool.submit(
                        self._process_edge,
                        i, seg, next_gaps[i], processed_dir
                    )
                    for i, seg in enumerate(segments)
                ]
                for i, fut in enumerate(futures):
                    try:
                        path = fut.result()
                        if path:
                            chunk_files.append((segments[i].start, path))
                        if notify_fn and i % 5 == 0:
                            notify_fn("TTS", f"edge-tts: {i + 1}/{len(segments)} segmentos…")
                    except Exception as e:
                        logger.error("TTS edge segment %d failed: %s", i, e)
        else:
            # XTTS must run serially to avoid CUDA OOM on small GPUs
            for i, seg in enumerate(segments):
                if notify_fn and i % 3 == 0:
                    notify_fn("TTS", f"XTTS: {i + 1}/{len(segments)} segmentos…")
                try:
                    path = self._process_xtts(i, seg, next_gaps[i], processed_dir)
                    if path:
                        chunk_files.append((seg.start, path))
                except Exception as e:
                    logger.error("TTS XTTS segment %d failed: %s", i, e)

        if notify_fn:
            notify_fn("TTS", f"Ensamblando {len(chunk_files)} segmentos en timeline…")

        final_wav = os.path.join(self.temp_dir, "assembled.wav")
        self._assemble_timeline(chunk_files, total_duration, final_wav)
        return final_wav

    # ── edge-tts backend ──────────────────────────────────────────────────────

    def _process_edge(
        self,
        idx: int,
        seg: TimedSegment,
        next_gap: float,
        out_dir: str,
    ) -> Optional[str]:
        """Generate one edge-tts segment and apply atempo alignment."""
        raw_mp3  = os.path.join(out_dir, f"seg_{idx}_raw.mp3")
        final_wav = os.path.join(out_dir, f"seg_{idx}_final.wav")

        voice = cfg.tts.speaker_voices.get(seg.speaker_id, cfg.tts.default_voice)

        # Run edge-tts (async) in a fresh event loop for this thread
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._edge_tts_save(seg.text, voice, raw_mp3))
        finally:
            loop.close()

        if not os.path.exists(raw_mp3) or os.path.getsize(raw_mp3) == 0:
            logger.warning("edge-tts produced empty file for segment %d", idx)
            return None

        actual_dur = self._get_duration(raw_mp3)
        target_dur = seg.end - seg.start
        speed_ratio = actual_dur / max(target_dur, 0.01)

        # Clamp speed ratio; allow overflow into silence if gap is large enough
        max_ratio = cfg.alignment.max_atempo_ratio
        min_ratio = cfg.alignment.min_atempo_ratio
        overflow_threshold = cfg.alignment.silence_overflow_s

        if speed_ratio > max_ratio:
            if next_gap >= overflow_threshold:
                # Allow TTS to extend into the silence gap – no speed-up needed
                speed_ratio = 1.0
                logger.debug("Seg %d: overflow allowed (gap=%.2fs)", idx, next_gap)
            else:
                speed_ratio = max_ratio
                logger.debug("Seg %d: clamped speed_ratio to %.2f", idx, max_ratio)
        elif speed_ratio < min_ratio:
            speed_ratio = min_ratio

        ffmpeg_atempo(raw_mp3, final_wav, speed_ratio)
        return final_wav

    @staticmethod
    async def _edge_tts_save(text: str, voice: str, output: str) -> None:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output)

    # ── XTTS v2 backend ──────────────────────────────────────────────────────

    def _process_xtts(
        self,
        idx: int,
        seg: TimedSegment,
        next_gap: float,
        out_dir: str,
    ) -> Optional[str]:
        """Generate one XTTS v2 segment with optional voice cloning.

        The ``speaker_wav`` reference is either:
          * Extracted from the source audio (when voice cloning is enabled), or
          * ``None`` (XTTS will use its default Spanish speaker).
        """
        from src.core.model_manager import models

        xtts = models.get_xtts()
        if xtts is None:
            return None

        raw_wav   = os.path.join(out_dir, f"seg_{idx}_raw.wav")
        final_wav = os.path.join(out_dir, f"seg_{idx}_final.wav")

        speaker_wav = self._speaker_refs.get(seg.speaker_id) if cfg.tts.use_voice_cloning else None

        try:
            if speaker_wav and os.path.exists(speaker_wav):
                # Voice cloning mode: use reference clip
                xtts.tts_to_file(
                    text=seg.text,
                    speaker_wav=speaker_wav,
                    language=cfg.tts.xtts_language,
                    file_path=raw_wav,
                )
            else:
                # Standard XTTS mode (no cloning)
                xtts.tts_to_file(
                    text=seg.text,
                    language=cfg.tts.xtts_language,
                    file_path=raw_wav,
                )
        except Exception as e:
            logger.error("XTTS synthesis failed for segment %d: %s", idx, e)
            return None

        if not os.path.exists(raw_wav) or os.path.getsize(raw_wav) == 0:
            return None

        actual_dur  = self._get_duration(raw_wav)
        target_dur  = seg.end - seg.start
        speed_ratio = actual_dur / max(target_dur, 0.01)
        max_ratio   = cfg.alignment.max_atempo_ratio
        min_ratio   = cfg.alignment.min_atempo_ratio

        if speed_ratio > max_ratio:
            if next_gap >= cfg.alignment.silence_overflow_s:
                speed_ratio = 1.0
            else:
                speed_ratio = max_ratio
        elif speed_ratio < min_ratio:
            speed_ratio = min_ratio

        ffmpeg_atempo(raw_wav, final_wav, speed_ratio)
        return final_wav

    # ── Voice cloning: extract speaker references ─────────────────────────────

    def _extract_speaker_refs(
        self,
        segments: List[TimedSegment],
        notify_fn: Optional[Callable[[str, str], None]],
    ) -> None:
        """Extract a short reference clip for each unique speaker.

        We pick the longest segment belonging to each speaker up to
        ``cfg.tts.reference_clip_duration`` seconds.  This gives XTTS v2 a
        good voice sample for cloning.
        """
        if not self.source_audio_path:
            return

        from collections import defaultdict
        speaker_segs: dict = defaultdict(list)
        for seg in segments:
            speaker_segs[seg.speaker_id].append(seg)

        ref_dir = os.path.join(self.temp_dir, "speaker_refs")
        os.makedirs(ref_dir, exist_ok=True)
        clip_dur = cfg.tts.reference_clip_duration

        for speaker_id, spk_segs in speaker_segs.items():
            # Choose the longest segment as reference
            best = max(spk_segs, key=lambda s: s.end - s.start)
            ref_path = os.path.join(ref_dir, f"{speaker_id}_ref.wav")
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-ss", str(best.start),
                        "-t",  str(min(best.end - best.start, clip_dur)),
                        "-i",  self.source_audio_path,
                        "-acodec", "pcm_s16le",
                        "-ar", "22050",
                        "-ac", "1",
                        ref_path,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._speaker_refs[speaker_id] = ref_path
                logger.info(
                    "Extracted reference clip for %s: %.2fs @ %.1fs",
                    speaker_id, clip_dur, best.start,
                )
            except Exception as e:
                logger.warning("Could not extract reference for %s: %s", speaker_id, e)

        if notify_fn:
            notify_fn("TTS", f"✅ Referencias de voz extraídas para {len(self._speaker_refs)} hablante(s)")

    # ── Timeline assembly ─────────────────────────────────────────────────────

    def _assemble_timeline(
        self,
        chunks: List[Tuple[float, str]],
        total_duration: float,
        output: str,
    ) -> None:
        """Mix all segments into a single WAV aligned to their original timestamps.

        Uses ffmpeg ``adelay`` + ``amix`` to position each segment on the
        correct timeline point, then hard-trims to ``total_duration``.
        """
        if not chunks:
            # Produce silence of the correct duration
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi",
                    "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100",
                    "-t", str(total_duration),
                    output,
                ],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return

        inputs: List[str] = []
        filter_parts: List[str] = []

        for i, (start, path) in enumerate(chunks):
            inputs += ["-i", path]
            delay_ms = int(start * 1000)
            filter_parts.append(
                f"[{i}:a]adelay={delay_ms}|{delay_ms}[a{i}];"
            )

        mix_inputs = "".join(f"[a{i}]" for i in range(len(chunks)))
        filter_parts.append(
            f"{mix_inputs}amix=inputs={len(chunks)}:normalize=0[out]"
        )

        cmd = (
            ["ffmpeg", "-y"]
            + inputs
            + [
                "-filter_complex", "".join(filter_parts),
                "-map", "[out]",
                "-t", str(total_duration),
                output,
            ]
        )
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _get_duration(path: str) -> float:
        """Return the duration of an audio file using ffprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    path,
                ],
                capture_output=True, text=True, check=True,
            )
            return float(result.stdout.strip())
        except Exception:
            # Fallback via moviepy if ffprobe fails
            try:
                try:
                    from moviepy.editor import AudioFileClip
                except ImportError:
                    from moviepy import AudioFileClip
                with AudioFileClip(path) as clip:
                    return clip.duration
            except Exception:
                return 0.0
