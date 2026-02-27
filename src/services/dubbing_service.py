import os
import time
import logging
import json
import hashlib
from typing import Optional, Callable
from src.core.entities import PipelineMetrics, TimedSegment, SpeakerSegment, TranscriptionResult
from src.core.model_manager import models
from src.adapters.media_adapter import to_mp3, export_ffmpeg
from src.services.diarization_service import DiarizationService
from src.services.transcription_service import TranscriptionService
from src.services.translation_service import TranslationService
from src.services.tts_service import TTSService

logger = logging.getLogger(__name__)

class DubbingService:
    """High-level orchestrator for the AI Dubbing pipeline."""

    def __init__(self):
        self._metrics = PipelineMetrics()
        self._temp_dir = None

    def run_pipeline(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[str, str], None]] = None,
        max_threads: int = 10,
    ) -> str:
        pipeline_start = time.perf_counter()
        self._metrics = PipelineMetrics()
        
        def _notify(stage: str, msg: str):
            if progress_callback: progress_callback(stage, msg)

        self._temp_dir = self._get_cache_dir(input_path)
        checkpoint_dir = os.path.join(self._temp_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        try:
            # 1. Audio Extraction
            logger.info("Stage 1: Audio Extraction...")
            audio_path, total_duration = to_mp3(input_path, self._temp_dir)
            self._metrics.audio_duration = total_duration
            _notify("Extracting", f"Audio listo ({total_duration:.2f}s)")

            # 2. Diarization
            logger.info("Stage 2: Diarization...")
            diar_service = DiarizationService()
            diar_segments = diar_service.perform_diarization(audio_path, _notify)
            logger.info(f"Diarization found {len(diar_segments)} segments.")

            # 3. Transcription
            logger.info("Stage 3: Transcription...")
            trans_service = TranscriptionService(self._temp_dir)
            transcription = trans_service.transcribe(audio_path, diar_segments, total_duration, _notify)
            logger.info(f"Transcription finished. Length: {len(transcription.full_text)} chars.")

            # 4. Translation
            logger.info("Stage 4: Translation...")
            transl_service = TranslationService()
            transl_service.load_model(_notify)
            translated_segments = transl_service.translate_segments(transcription.segments, _notify)
            logger.info(f"Translation finished. Segments: {len(translated_segments)}")

            # 5. TTS & Assembly
            logger.info("Stage 5: TTS & Assembly...")
            tts_service = TTSService(self._temp_dir)
            assembled_path = tts_service.generate_and_assemble(
                translated_segments, total_duration, _notify, max_threads
            )

            # 6. Export
            logger.info("Stage 6: Exporting...")
            _notify("Saving", "Generando archivo final...")
            export_ffmpeg(assembled_path, output_path)

            self._metrics.total_time = time.perf_counter() - pipeline_start
            _notify("Completed", "Proceso finalizado con éxito.")
            logger.info(f"Pipeline finished in {self._metrics.total_time:.2f}s")
            return output_path

        except Exception as e:
            logger.exception("Pipeline failed")
            raise

    def _get_cache_dir(self, input_path: str) -> str:
        path_hash = hashlib.md5(input_path.encode()).hexdigest()[:12]
        base_name = os.path.splitext(os.path.basename(input_path))[0][:20]
        cache_dir = os.path.join(os.getcwd(), "dubbing_cache", f"{base_name}_{path_hash}")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
