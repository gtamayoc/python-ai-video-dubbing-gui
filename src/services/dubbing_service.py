"""
src/services/dubbing_service.py  (v2 – orchestrator)
─────────────────────────────────────────────────────
High-level pipeline orchestrator.  The GUI calls ``run_pipeline()``; this
class delegates to the individual service modules.

Pipeline stages:
  1. Audio extraction   (ffmpeg via media_adapter)
  2. Diarization        (pyannote.audio  → fallback: single speaker)
  3. Transcription      (faster-whisper  large-v3 | small)
  4. Translation        (MarianMT | NLLB | GGUF | Argos)
  5. TTS + assembly     (edge-tts | XTTS v2)
  6. Final export       (ffmpeg)

The active pipeline mode is controlled by ``config.yaml → pipeline_mode``:
  • "simple"   → whisper-small + Argos + edge-tts  (legacy, fast)
  • "advanced" → whisper-large-v3 + MarianMT + edge-tts  (high quality)

The mode can be overridden at runtime by passing ``pipeline_mode`` to
``run_pipeline()``, or changed in config.yaml without restarting the app
(settings are reloaded on each pipeline run).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Callable, Optional

from src.core.entities import PipelineMetrics
from src.core.model_manager import models
from src.adapters.media_adapter import to_mp3, export_ffmpeg
from src.services.diarization_service import DiarizationService
from src.services.transcription_service import TranscriptionService
from src.services.translation_service import TranslationService
from src.services.tts_service import TTSService

logger = logging.getLogger(__name__)


class DubbingService:
    """Orchestrates the complete AI dubbing pipeline.

    Usage::

        svc = DubbingService()
        output = svc.run_pipeline(
            input_path="video.mp4",
            output_path="video_es.wav",
            progress_callback=my_callback,
        )
    """

    def __init__(self) -> None:
        self._metrics = PipelineMetrics()

    # ── Main entry point ──────────────────────────────────────────────────────

    def run_pipeline(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[str, str], None]] = None,
        max_threads: int = 8,
        pipeline_mode: Optional[str] = None,
    ) -> str:
        """Run the complete dubbing pipeline from *input_path* to *output_path*.

        Parameters
        ----------
        input_path:
            Path to source video or audio file.
        output_path:
            Desired output path (WAV or MP4).
        progress_callback:
            ``(stage: str, message: str) → None`` for GUI updates.
        max_threads:
            Number of threads for parallel edge-tts generation.
        pipeline_mode:
            ``"simple"`` or ``"advanced"``.  If ``None``, reads from
            ``config.yaml``.

        Returns
        -------
        str
            Absolute path to the finished file (same as *output_path*).
        """
        # Reload settings live so changes to config.yaml take effect immediately
        from src.core.settings import settings as cfg

        # Determine active mode
        active_mode = pipeline_mode or cfg.pipeline_mode
        logger.info("=== DubbingService: mode=%s ===", active_mode)

        # Apply mode overrides (simple mode forces smaller models + argos)
        if active_mode == "simple":
            models._whisper_model_size = "small"
            effective_translation_backend = "argos"
            effective_tts_backend = "edge"
        else:
            # Advanced mode — keep model sizes from config.yaml
            models.configure_from_settings()
            effective_translation_backend = cfg.translation.backend
            effective_tts_backend         = cfg.tts.backend

        pipeline_start = time.perf_counter()
        self._metrics  = PipelineMetrics()

        def _notify(stage: str, msg: str) -> None:
            logger.info("[%s] %s", stage, msg)
            if progress_callback:
                progress_callback(stage, msg)

        temp_dir      = self._get_cache_dir(input_path)
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        try:
            # ── Stage 1: Audio Extraction ─────────────────────────────────────
            t = time.perf_counter()
            _notify("Extracting", "Extrayendo audio…")
            audio_path, total_duration = to_mp3(input_path, temp_dir)
            self._metrics.audio_duration = total_duration
            self._metrics.stage_times["1_extraction"] = time.perf_counter() - t
            _notify("Extracting", f"✅ Audio listo ({total_duration:.2f}s)")

            # ── Stage 2: Diarization ──────────────────────────────────────────
            t = time.perf_counter()
            diar_service = DiarizationService()
            if cfg.diarization.enabled or active_mode == "advanced":
                diar_segments = diar_service.perform_diarization(audio_path, _notify)
            else:
                diar_segments = diar_service._fallback_diarization(audio_path)
                _notify("Diarizing", "Modo simple: hablante único")
            self._metrics.stage_times["2_diarization"] = time.perf_counter() - t
            _notify("Diarizing", f"✅ {len(diar_segments)} segmentos de hablante detectados")

            # ── Stage 3: Transcription ────────────────────────────────────────
            t = time.perf_counter()
            trans_service = TranscriptionService(temp_dir)
            transcription = trans_service.transcribe(
                audio_path, diar_segments, total_duration, _notify
            )
            self._metrics.stage_times["3_transcription"] = time.perf_counter() - t
            _notify(
                "Transcribing",
                f"✅ {len(transcription.segments)} segmentos trascritos "
                f"({len(transcription.full_text)} chars)",
            )

            # Save transcription checkpoint
            self._save_checkpoint(
                checkpoint_dir, "transcription",
                [{"speaker": s.speaker_id, "start": s.start, "end": s.end, "text": s.text}
                 for s in transcription.segments]
            )

            # ── Stage 4: Translation ──────────────────────────────────────────
            t = time.perf_counter()
            transl_service = TranslationService(backend=effective_translation_backend)
            transl_service.load_model(_notify)
            translated = transl_service.translate_segments(transcription.segments, _notify)
            self._metrics.stage_times["4_translation"] = time.perf_counter() - t
            _notify("Translating", f"✅ {len(translated)} segmentos traducidos")

            # Save translation checkpoint
            self._save_checkpoint(
                checkpoint_dir, "translation",
                [{"speaker": s.speaker_id, "start": s.start, "end": s.end,
                  "original": transcription.segments[i].text if i < len(transcription.segments) else "",
                  "translated": s.text}
                 for i, s in enumerate(translated)]
            )

            # ── Stage 5: TTS + Assembly ───────────────────────────────────────
            t = time.perf_counter()
            tts_service = TTSService(
                temp_dir=temp_dir,
                source_audio_path=audio_path,   # For voice cloning reference extraction
            )
            assembled_path = tts_service.generate_and_assemble(
                translated, total_duration, _notify, max_threads
            )
            self._metrics.stage_times["5_tts_assembly"] = time.perf_counter() - t
            _notify("TTS", "✅ Audio doblado ensamblado")

            # ── Stage 6: Export ───────────────────────────────────────────────
            t = time.perf_counter()
            _notify("Saving", "Exportando archivo final…")
            export_ffmpeg(assembled_path, output_path)
            self._metrics.stage_times["6_export"] = time.perf_counter() - t

            # ── Final metrics ─────────────────────────────────────────────────
            self._metrics.total_time = time.perf_counter() - pipeline_start
            if total_duration > 0:
                self._metrics.realtime_factor = total_duration / self._metrics.total_time

            summary = self._metrics.summary()
            logger.info("\n%s", summary)
            _notify("Completed", f"✅ Proceso finalizado en {self._metrics.total_time:.1f}s")

            # Duration-match validation
            self._validate_duration(output_path, total_duration, _notify)

            return output_path

        except Exception:
            logger.exception("Pipeline failed")
            raise

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_cache_dir(self, input_path: str) -> str:
        path_hash = hashlib.md5(input_path.encode()).hexdigest()[:12]
        base_name = os.path.splitext(os.path.basename(input_path))[0][:20]
        cache_dir = os.path.join(os.getcwd(), "dubbing_cache", f"{base_name}_{path_hash}")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    @staticmethod
    def _save_checkpoint(checkpoint_dir: str, name: str, data: object) -> None:
        path = os.path.join(checkpoint_dir, f"{name}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug("Checkpoint saved: %s", path)
        except Exception as e:
            logger.warning("Could not save checkpoint '%s': %s", name, e)

    @staticmethod
    def _validate_duration(
        output_path: str,
        expected_duration: float,
        notify_fn: Optional[Callable[[str, str], None]],
    ) -> None:
        """Verify the output duration matches the source within ±0.5 s."""
        try:
            import subprocess
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    output_path,
                ],
                capture_output=True, text=True, check=True,
            )
            actual = float(result.stdout.strip())
            diff   = abs(actual - expected_duration)
            msg    = (
                f"Duración salida: {actual:.2f}s  "
                f"(esperado: {expected_duration:.2f}s,  Δ={diff:.2f}s)"
            )
            if diff <= 0.5:
                logger.info("✅ Duration OK — %s", msg)
                if notify_fn:
                    notify_fn("Validation", f"✅ {msg}")
            else:
                logger.warning("⚠ Duration mismatch — %s", msg)
                if notify_fn:
                    notify_fn("Validation", f"⚠ {msg}")
        except Exception as e:
            logger.debug("Duration validation skipped: %s", e)
