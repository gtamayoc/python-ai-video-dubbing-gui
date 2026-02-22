"""
translator_service.py
=====================
Backend pipeline for English → Spanish audio translation with voice cloning.

OPTIMIZED VERSION — Targets < 10s latency for ~3-minute audio files.

Key optimizations:
  1. Singleton model loading (warm start) — models stay in memory
  2. GPU-first inference with automatic CPU fallback
  3. Chunked/streaming transcription → translation pipeline
  4. Batch-parallel translation of segments
  5. Direct FFmpeg subprocess instead of moviepy overhead
  6. Quantized models (INT8 on GPU, INT8 on CPU)
  7. Async TTS generation with edge-tts
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time

# Suppress HuggingFace symlink warnings on Windows before importing anything else
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpeakerSegment:
    """One continuous segment attributed to a single speaker."""
    speaker_id: str
    start_time: float   # seconds
    end_time: float     # seconds


@dataclass
class TranscriptionSegment:
    """A single transcription segment with timing and text."""
    speaker_id: str
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Transcription output mapped to speaker segments."""
    full_text: str
    segments: List[TranscriptionSegment] = field(default_factory=list)


@dataclass
class PipelineMetrics:
    """Profiling metrics for each pipeline stage."""
    stage_times: Dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    audio_duration: float = 0.0
    realtime_factor: float = 0.0  # audio_duration / total_time

    def summary(self) -> str:
        lines = ["┌─── Pipeline Profiling ───"]
        for stage, t in self.stage_times.items():
            lines.append(f"│  {stage:<25s} {t:6.2f}s")
        lines.append(f"├──────────────────────────")
        lines.append(f"│  TOTAL                    {self.total_time:6.2f}s")
        if self.audio_duration > 0:
            lines.append(f"│  Audio duration           {self.audio_duration:6.2f}s")
            lines.append(f"│  Realtime factor          {self.realtime_factor:6.2f}x")
        lines.append(f"└──────────────────────────")
        return "\n".join(lines)


ProgressCallback = Callable[[str, str], None]


# ---------------------------------------------------------------------------
# Model Manager — Singleton pattern for warm starts
# ---------------------------------------------------------------------------

class _ModelManager:
    """Thread-safe singleton that keeps ML models loaded in memory.

    First call incurs the loading cost; subsequent calls return instantly.
    Models are lazily loaded only when first needed.
    """

    _instance: Optional["_ModelManager"] = None
    _lock = Lock()

    def __new__(cls) -> "_ModelManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._whisper_model = None
        self._whisper_lock = Lock()
        self._argos_ready = False
        self._argos_lock = Lock()
        self._device: str = "auto"  # auto-detect GPU
        self._compute_type: str = "auto"  # auto-select best precision
        self._whisper_model_size: str = "small"  # better quality than "base"

    @property
    def device(self) -> str:
        return self._device

    def configure(
        self,
        device: str = "auto",
        compute_type: str = "auto",
        model_size: str = "small",
    ) -> None:
        """Configure model parameters before first load.

        Args:
            device: "cuda", "cpu", or "auto" (detect GPU availability).
            compute_type: "float16", "int8", "int8_float16", or "auto".
            model_size: Whisper model size — "tiny", "base", "small",
                        "medium", "large-v3".
        """
        self._device = device
        self._compute_type = compute_type
        self._whisper_model_size = model_size
        # Reset models so they reload with new config
        self._whisper_model = None
        self._argos_ready = False

    def _resolve_device(self) -> str:
        """Resolve 'auto' to the best available device."""
        if self._device != "auto":
            return self._device
        try:
            import torch
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_mem
                logger.info(
                    "CUDA available — GPU: %s, VRAM: %.1f GB",
                    torch.cuda.get_device_name(0),
                    vram / 1e9,
                )
                return "cuda"
        except ImportError:
            pass
        logger.info("CUDA not available — falling back to CPU")
        return "cpu"

    def _resolve_compute_type(self, device: str) -> str:
        """Choose optimal compute type for the detected device."""
        if self._compute_type != "auto":
            return self._compute_type
        if device == "cuda":
            return "float16"  # fastest on GPU with good quality
        return "int8"  # best CPU throughput

    def get_whisper(self):
        """Return the cached WhisperModel, loading it on first call."""
        with self._whisper_lock:
            if self._whisper_model is None:
                from faster_whisper import WhisperModel

                device = self._resolve_device()
                ctype = self._resolve_compute_type(device)
                logger.info(
                    "Loading Whisper '%s' on %s (%s)… "
                    "(Note: downloading on first run, this may take a minute)",
                    self._whisper_model_size, device, ctype,
                )
                t0 = time.perf_counter()
                self._whisper_model = WhisperModel(
                    self._whisper_model_size,
                    device=device,
                    compute_type=ctype,
                    cpu_threads=max(4, (os.cpu_count() or 4)),
                    num_workers=max(2, (os.cpu_count() or 4) // 2),
                )
                logger.info("Whisper loaded in %.2fs", time.perf_counter() - t0)
            return self._whisper_model

    def ensure_argos(self) -> None:
        """Ensure the argos-translate en→es package is installed (once)."""
        with self._argos_lock:
            if self._argos_ready:
                return

            import argostranslate.package
            import argostranslate.translate

            installed_langs = argostranslate.translate.get_installed_languages()
            en_ok = any(lang.code == "en" for lang in installed_langs)
            es_ok = any(lang.code == "es" for lang in installed_langs)

            if not (en_ok and es_ok):
                logger.info("Installing argos-translate en→es package…")
                available = argostranslate.package.get_available_packages()
                pkg = next(
                    (p for p in available
                     if p.from_code == "en" and p.to_code == "es"),
                    None,
                )
                if pkg:
                    pkg.install()
                    logger.info("Argos en→es package installed.")

            self._argos_ready = True

    def preload_all(self) -> float:
        """Eagerly load all models. Returns total load time in seconds."""
        t0 = time.perf_counter()
        self.get_whisper()
        self.ensure_argos()
        return time.perf_counter() - t0

    def release(self) -> None:
        """Release all models and free GPU memory."""
        with self._whisper_lock:
            if self._whisper_model is not None:
                del self._whisper_model
                self._whisper_model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        self._argos_ready = False
        logger.info("Models released.")


# Module-level accessor
models = _ModelManager()


# ---------------------------------------------------------------------------
# FFmpeg helpers — avoid moviepy overhead
# ---------------------------------------------------------------------------

def _find_ffmpeg() -> str:
    """Return path to ffmpeg binary.
    
    Tries: 
    1. System PATH
    2. imageio_ffmpeg (often available in venvs with moviepy)
    """
    # 1. Check system PATH
    path = shutil.which("ffmpeg")
    if path:
        return path

    # 2. Check imageio_ffmpeg
    try:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and os.path.exists(path):
            return path
    except ImportError:
        pass

    # 3. Last resort/Error
    raise RuntimeError(
        "FFmpeg not found! Please install FFmpeg and add it to your PATH, "
        "or install imageio-ffmpeg: 'pip install imageio-ffmpeg'"
    )


def _to_wav_ffmpeg(input_path: str, output_path: str) -> str:
    """Convert any audio file to 16 kHz mono WAV using FFmpeg directly.

    16 kHz mono is optimal for Whisper and avoids unnecessary I/O.
    """
    cmd = [
        _find_ffmpeg(),
        "-y",                       # overwrite
        "-i", input_path,
        "-ar", "16000",             # 16 kHz (Whisper's native rate)
        "-ac", "1",                 # mono
        "-c:a", "pcm_s16le",       # 16-bit PCM WAV
        "-f", "wav",
        output_path,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr[:500]}")
    return output_path


def _export_ffmpeg(input_path: str, output_path: str) -> None:
    """Export audio to the desired format using FFmpeg directly."""
    ext = os.path.splitext(output_path)[1].lower()
    codec_map = {
        ".mp3": ["libmp3lame", "-q:a", "2"],
        ".ogg": ["libvorbis", "-q:a", "4"],
        ".aac": ["aac", "-b:a", "192k"],
        ".m4a": ["aac", "-b:a", "192k"],
        ".flac": ["flac"],
    }

    cmd = [_find_ffmpeg(), "-y", "-i", input_path]

    codec_args = codec_map.get(ext)
    if codec_args:
        cmd += ["-c:a", codec_args[0]] + codec_args[1:]
    # else: let FFmpeg auto-detect codec from extension

    cmd.append(output_path)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg export failed: {result.stderr[:500]}")


def _get_audio_duration(path: str) -> float:
    """Get audio duration in seconds. Tries ffprobe, then falls back to 0.0."""
    ffmpeg_path = _find_ffmpeg()
    ffprobe_path = ffmpeg_path.replace("ffmpeg", "ffprobe")
    
    # Check if ffprobe actually exists at that path
    if not os.path.exists(ffprobe_path) and not shutil.which("ffprobe"):
        # If no ffprobe, we could use a library, but for now just return 0
        # to avoid breaking the pipeline.
        return 0.0

    cmd = [
        ffprobe_path if os.path.exists(ffprobe_path) else "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Translation helpers — batch-parallel segment translation
# ---------------------------------------------------------------------------

def _translate_segment(text: str) -> str:
    """Translate a single text segment EN→ES using argos-translate."""
    import argostranslate.translate
    if not text or not text.strip():
        return ""
    return argostranslate.translate.translate(text, "en", "es")


def _translate_batch(segments: List[str], max_workers: int = 4) -> List[str]:
    """Translate a list of text segments in parallel using ThreadPoolExecutor.

    Argos-translate is CPU-bound but releases the GIL during native ops,
    so threading gives moderate speedup for multiple segments.
    """
    if not segments:
        return []

    # For very short lists, sequential is faster (avoids thread overhead)
    if len(segments) <= 2:
        return [_translate_segment(s) for s in segments]

    results: List[Optional[str]] = [None] * len(segments)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(segments))) as pool:
        future_to_idx = {
            pool.submit(_translate_segment, seg): i
            for i, seg in enumerate(segments)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.warning("Translation failed for segment %d: %s", idx, exc)
                results[idx] = segments[idx]  # fallback: original text

    return results


# ---------------------------------------------------------------------------
# Chunked/Streaming Transcription + Translation
# ---------------------------------------------------------------------------

def _chunked_transcribe_and_translate(
    wav_path: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[TranscriptionResult, str]:
    """Transcribe and translate in a streaming/chunked fashion.

    Instead of waiting for full transcription before translating:
    1. Whisper yields segments as a generator (streaming).
    2. We collect segments in batches.
    3. After each batch, we immediately translate those segments in parallel.
    4. This overlaps transcription I/O wait with translation compute.

    Returns (transcription_result, translated_text).
    """
    whisper_model = models.get_whisper()
    models.ensure_argos()

    BATCH_SIZE = 10  # translate every N segments

    segments_iter, info = whisper_model.transcribe(
        wav_path,
        language="en",
        beam_size=1,            # CRITICAL: 1 is much faster than 5
        best_of=1,              # Disable multiple candidates
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 700, # More aggressive silence skipping
        },
    )

    all_segments: List[TranscriptionSegment] = []
    all_texts: List[str] = []
    translated_parts: List[str] = []
    batch_buffer: List[str] = []
    char_count = 0 

    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue

        all_segments.append(TranscriptionSegment(
            speaker_id="speaker_1",
            start=seg.start,
            end=seg.end,
            text=text,
        ))
        all_texts.append(text)
        batch_buffer.append(text)
        char_count += len(text)

        # Buffer by physical length (characters) rather than segment count
        # This reduces overhead for short phrases.
        if char_count >= 1000: 
            if progress_callback:
                progress_callback(
                    "Transcribing",
                    f"Transcribed {int(seg.end)}s, translating batch…",
                )
            translated_batch = _translate_batch(batch_buffer)
            translated_parts.extend(translated_batch)
            batch_buffer = []
            char_count = 0

    # Translate remaining segments
    if batch_buffer:
        if progress_callback:
            progress_callback(
                "Translating",
                f"Translating final {len(batch_buffer)} segments…",
            )
        translated_batch = _translate_batch(batch_buffer)
        translated_parts.extend(translated_batch)

    full_text = " ".join(all_texts)
    translated_text = " ".join(translated_parts)

    transcription = TranscriptionResult(
        full_text=full_text,
        segments=all_segments,
    )

    return transcription, translated_text


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class TranslatorService:
    """Orchestrates the English → Spanish audio translation pipeline.

    OPTIMIZED for latency < 10s on ~3-minute audio files:
    - Models are loaded once and kept in memory (singleton)
    - GPU is used when available (auto-detected)
    - Transcription and translation are interleaved (chunked streaming)
    - FFmpeg is called directly (no moviepy overhead)
    - Translation segments are processed in parallel batches
    """

    AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".wma", ".m4a"}

    def __init__(self, preload: bool = False) -> None:
        self._temp_dir: Optional[str] = None
        self._metrics = PipelineMetrics()

        if preload:
            self.preload_models()

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    @staticmethod
    def preload_models() -> float:
        """Preload all ML models into memory. Returns load time in seconds.

        Call this once at application startup for warm-start performance.
        """
        return models.preload_all()

    @staticmethod
    def configure_models(
        device: str = "auto",
        compute_type: str = "auto",
        model_size: str = "small",
    ) -> None:
        """Configure model parameters before loading.

        Args:
            device: "cuda", "cpu", or "auto".
            compute_type: "float16", "int8", "int8_float16", or "auto".
            model_size: "tiny", "base", "small", "medium", "large-v3".
        """
        models.configure(device=device, compute_type=compute_type,
                         model_size=model_size)

    @staticmethod
    def release_models() -> None:
        """Release all models and free GPU memory."""
        models.release()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> str:
        """Run the full translation pipeline. Returns *output_path*.

        Pipeline:
          1. Audio → WAV (FFmpeg, 16 kHz mono)
          2. Speaker diarization
          3. Chunked transcription + interleaved translation
          4. Voice synthesis (edge-tts, async)
          5. Export to output format
        """
        pipeline_start = time.perf_counter()
        self._metrics = PipelineMetrics()

        def _notify(stage: str, msg: str) -> None:
            if progress_callback:
                progress_callback(stage, msg)

        self._temp_dir = tempfile.mkdtemp(prefix="translator_")

        try:
            # 1 — Convert to WAV (FFmpeg direct, ~0.5s)
            _notify("Extracting", "Converting audio to WAV (16 kHz mono)…")
            t0 = time.perf_counter()
            wav_path = os.path.join(self._temp_dir, "source.wav")
            _to_wav_ffmpeg(input_path, wav_path)
            self._metrics.stage_times["1_extract"] = time.perf_counter() - t0
            self._metrics.audio_duration = _get_audio_duration(wav_path)
            _notify("Extracting", f"Audio ready ({self._metrics.audio_duration:.1f}s)")

            # 2 — Speaker diarization (placeholder, ~0s)
            _notify("Diarizing", "Detecting speakers…")
            t0 = time.perf_counter()
            segments = self.perform_diarization(wav_path)
            self._metrics.stage_times["2_diarize"] = time.perf_counter() - t0
            _notify("Diarizing", f"{len(segments)} segment(s) detected.")

            # 3+4 — Chunked transcription + interleaved translation
            _notify("Transcribing", "Starting chunked transcription + translation…")
            t0 = time.perf_counter()
            transcription, translated_text = _chunked_transcribe_and_translate(
                wav_path, progress_callback=progress_callback,
            )
            self._metrics.stage_times["3_transcribe+translate"] = (
                time.perf_counter() - t0
            )
            _notify(
                "Translating",
                f"Done — {len(transcription.segments)} segments, "
                f"{len(translated_text)} chars translated.",
            )

            # 5 — Voice cloning + speech generation (async edge-tts)
            _notify("Cloning Voice", "Generating Spanish speech…")
            t0 = time.perf_counter()
            generated_path = self.clone_voice_and_generate_speech(
                segments, translated_text,
            )
            self._metrics.stage_times["4_tts"] = time.perf_counter() - t0
            _notify("Cloning Voice", "Spanish audio generated.")

            # 6 — Export (FFmpeg direct)
            _notify("Saving", f"Writing output to {output_path}…")
            t0 = time.perf_counter()
            self._export(generated_path, output_path)
            self._metrics.stage_times["5_export"] = time.perf_counter() - t0

            # Finalize metrics
            self._metrics.total_time = time.perf_counter() - pipeline_start
            if self._metrics.audio_duration > 0:
                self._metrics.realtime_factor = (
                    self._metrics.audio_duration / self._metrics.total_time
                )

            profile_summary = self._metrics.summary()
            logger.info("\n%s", profile_summary)
            _notify("Completed", f"Done → {output_path}")
            _notify("Completed", profile_summary)

            return output_path
        finally:
            self._cleanup()

    @property
    def last_metrics(self) -> PipelineMetrics:
        """Access profiling metrics from the last pipeline run."""
        return self._metrics

    # ------------------------------------------------------------------
    # Pipeline methods
    # ------------------------------------------------------------------

    def perform_diarization(self, wav_path: str) -> List[SpeakerSegment]:
        """Detect *who* speaks *when*.

        **Placeholder** — returns two fake segments.
        TODO: integrate pyannote.audio or NeMo for real diarization.
        """
        duration = _get_audio_duration(wav_path)
        return [
            SpeakerSegment("speaker_1", 0.0, duration / 2),
            SpeakerSegment("speaker_2", duration / 2, duration),
        ]

    def transcribe_audio(self, wav_path: str) -> TranscriptionResult:
        """Transcribe English speech to text using faster-whisper (cached).

        Uses the model configured via configure_models() (default: 'small').
        Model is loaded once and kept in memory for subsequent calls.
        """
        whisper_model = models.get_whisper()

        segments_iter, info = whisper_model.transcribe(
            wav_path,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        all_segments = []
        texts = []
        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue
            texts.append(text)
            all_segments.append(TranscriptionSegment(
                speaker_id="speaker_1",
                start=seg.start,
                end=seg.end,
                text=text,
            ))

        return TranscriptionResult(
            full_text=" ".join(texts),
            segments=all_segments,
        )

    def translate_text(self, text: str) -> str:
        """Translate English text to Spanish using argos-translate (cached).

        Package installation check runs only once (cached in ModelManager).
        """
        models.ensure_argos()

        if not text:
            return ""

        import argostranslate.translate
        return argostranslate.translate.translate(text, "en", "es")

    def clone_voice_and_generate_speech(
        self, segments: List[SpeakerSegment], translated_text: str,
    ) -> str:
        """Generate Spanish speech using edge-tts (Microsoft neural voice).

        Uses asyncio for non-blocking I/O during synthesis.
        """
        import edge_tts

        if not translated_text:
            raise ValueError("No translated text to synthesize.")

        out_path = os.path.join(self._temp_dir, "cloned_es.mp3")

        async def _synth():
            communicate = edge_tts.Communicate(
                translated_text, "es-ES-AlvaroNeural",
            )
            await communicate.save(out_path)

        # Use existing event loop if available, otherwise create one
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context already
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(asyncio.run, _synth()).result()
        else:
            asyncio.run(_synth())

        return out_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _export(self, audio_path: str, output_path: str) -> None:
        """Export the generated audio to the desired output format."""
        in_ext = os.path.splitext(audio_path)[1].lower()
        out_ext = os.path.splitext(output_path)[1].lower()

        # If same format, just copy (fastest path)
        if in_ext == out_ext:
            shutil.copy2(audio_path, output_path)
        else:
            _export_ffmpeg(audio_path, output_path)

    def _cleanup(self) -> None:
        if self._temp_dir and os.path.isdir(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
