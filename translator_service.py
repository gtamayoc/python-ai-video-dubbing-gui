"""
translator_service.py
=====================
Backend pipeline for English → Spanish audio translation with voice cloning.

Pipeline overview
-----------------
1. Extract / normalise audio → WAV + capture total_duration
2. Speaker diarisation → List[SpeakerSegment]  (who speaks when)
3. Transcription     → per-segment text with timestamps, speaker joined from step 2
4. Translation       → per-segment Spanish text  (timing & speaker preserved)
5. TTS + time-stretch→ each segment synthesised then stretched to its original window
6. Assembly          → segments overlaid on a silent timeline of exactly total_duration
7. Export            → write output file
"""

from __future__ import annotations

import asyncio
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional

try:
    from moviepy.editor import AudioFileClip
except ImportError:
    from moviepy import AudioFileClip


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpeakerSegment:
    """One continuous time-window attributed to a single speaker."""
    speaker_id: str
    start_time: float   # seconds
    end_time: float     # seconds


@dataclass
class TimedSegment:
    """A transcribed (and later translated) segment with full timing metadata."""
    speaker_id: str
    start: float        # seconds
    end: float          # seconds
    text: str           # original or translated text


@dataclass
class TranscriptionResult:
    """Transcription output: full text + per-segment detail."""
    full_text: str
    segments: List[TimedSegment] = field(default_factory=list)


ProgressCallback = Callable[[str, str], None]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

import json
import hashlib

# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class TranslatorService:
    """Orchestrates the English → Spanish audio translation pipeline with persistence."""

    AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".wma", ".m4a"}

    def __init__(self) -> None:
        self._temp_dir: Optional[str] = None
        self._cache_dir: Optional[str] = None

    def _get_cache_dir(self, input_path: str) -> str:
        """Create a persistent cache dir based on input file path."""
        path_hash = hashlib.md5(input_path.encode()).hexdigest()[:12]
        base_name = os.path.splitext(os.path.basename(input_path))[0][:20]
        cache_dir = os.path.join(os.getcwd(), "dubbing_cache", f"{base_name}_{path_hash}")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> str:
        """Run the full translation pipeline with resumption support."""

        def _notify(stage: str, msg: str) -> None:
            if progress_callback:
                progress_callback(stage, msg)

        # Use persistent cache instead of random temp dir
        self._temp_dir = self._get_cache_dir(input_path)
        checkpoint_dir = os.path.join(self._temp_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        try:
            # 1 — Extract / normalise audio
            audio_path = os.path.join(self._temp_dir, "source.mp3")
            if os.path.exists(audio_path):
                _notify("Extracting", "Using cached audio file…")
                with AudioFileClip(audio_path) as clip:
                    total_duration = clip.duration
            else:
                _notify("Extracting", "Reading audio file (this may take a while)…")
                audio_path, total_duration = self._to_mp3(input_path)
            
            _notify("Extracting", f"Audio ready ({total_duration:.2f}s)")

            # 2 — Speaker diarisation
            diar_json = os.path.join(checkpoint_dir, "diarization.json")
            if os.path.exists(diar_json):
                _notify("Diarizing", "Loading cached speakers…")
                with open(diar_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    diar_segments = [SpeakerSegment(**s) for s in data]
            else:
                _notify("Diarizing", "Detecting speakers…")
                diar_segments = self.perform_diarization(audio_path)
                with open(diar_json, 'w', encoding='utf-8') as f:
                    json.dump([s.__dict__ for s in diar_segments], f)
            _notify("Diarizing", f"{len(diar_segments)} segment(s) detected.")

            # 3 — Transcription
            trans_json = os.path.join(checkpoint_dir, "transcription.json")
            if os.path.exists(trans_json):
                _notify("Transcribing", "Loading cached transcription…")
                with open(trans_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    transcription = TranscriptionResult(
                        full_text=data['full_text'],
                        segments=[TimedSegment(**s) for s in data['segments']]
                    )
            else:
                _notify("Transcribing", "Transcribing English speech (chunked process)…")
                transcription = self.transcribe_audio(audio_path, diar_segments, total_duration, _notify)
                with open(trans_json, 'w', encoding='utf-8') as f:
                    json.dump({
                        'full_text': transcription.full_text,
                        'segments': [s.__dict__ for s in transcription.segments]
                    }, f)
            _notify("Transcribing", f"Transcription done ({len(transcription.segments)} segments).")

            # 4 — Translation
            trad_json = os.path.join(checkpoint_dir, "translation.json")
            if os.path.exists(trad_json):
                _notify("Translating", "Loading cached translation…")
                with open(trad_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    translated_segments = [TimedSegment(**s) for s in data]
            else:
                _notify("Translating", "Translating segments to Spanish…")
                translated_segments = self.translate_segments(transcription.segments)
                with open(trad_json, 'w', encoding='utf-8') as f:
                    json.dump([s.__dict__ for s in translated_segments], f)
            _notify("Translating", f"{len(translated_segments)} segment(s) translated.")

            # 5 — TTS + Assembly
            _notify("Cloning Voice", "Generating & synchronising Spanish speech…")
            assembled_path = self.clone_voice_and_generate_speech(
                translated_segments, total_duration, _notify
            )
            
            # 6 — Save
            _notify("Saving", f"Writing output to {output_path}…")
            self._export(assembled_path, output_path)
            _notify("Completed", f"Done → {output_path}")

            return output_path
        except Exception as e:
            raise e

    def perform_diarization(self, audio_path: str) -> List[SpeakerSegment]:
        """Detect *who* speaks *when*.

        **Placeholder** — returns two fake segments covering the full file.
        TODO: integrate pyannote.audio or similar.
        """
        # Fallback: treat whole file as speaker_1 (single segment).
        # Replace with real diarisation to get per-speaker windows.
        with AudioFileClip(audio_path) as clip:
            duration = clip.duration
        return [
            SpeakerSegment("speaker_1", 0.0, duration),
        ]

    def transcribe_audio(

        self, audio_path: str, diar_segments: List[SpeakerSegment], total_duration: float, notify_fn: Optional[Callable] = None
    ) -> TranscriptionResult:
        """Transcribe audio using faster-whisper. 
        For long files, processes in 10-minute chunks to avoid OOM.
        """
        from faster_whisper import WhisperModel
        
        # Load model with conservative settings
        model = WhisperModel("base", device="cpu", compute_type="int8") # Use CPU to be safer with memory
        
        chunk_size = 600 # 10 minutes per chunk
        all_timed: List[TimedSegment] = []
        all_texts: List[str] = []
        
        num_chunks = math.ceil(total_duration / chunk_size)
        
        for i in range(num_chunks):
            start_off = i * chunk_size
            dur = min(chunk_size, total_duration - start_off)
            
            if notify_fn:
                notify_fn("Transcribing", f"Processing chunk {i+1}/{num_chunks} ({int(start_off)}s - {int(start_off+dur)}s)")

            # Transcribe this specific window
            # We use ffmpeg to extract a small MP3 chunk to save disk and memory
            chunk_mp3 = os.path.join(self._temp_dir, f"chunk_{i}.mp3")
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(start_off), "-t", str(dur), 
                "-i", audio_path, "-acodec", "libmp3lame", "-ab", "64k", chunk_mp3
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            c_segments, _ = model.transcribe(chunk_mp3, language="en")
            for seg in c_segments:
                speaker_id = self._assign_speaker(seg.start + start_off, seg.end + start_off, diar_segments)
                all_timed.append(TimedSegment(
                    speaker_id=speaker_id,
                    start=seg.start + start_off,
                    end=seg.end + start_off,
                    text=seg.text.strip(),
                ))
                all_texts.append(seg.text.strip())
            
            # Clean up chunk mp3 immediately
            try: os.remove(chunk_mp3)
            except: pass

        return TranscriptionResult(
            full_text=" ".join(all_texts),
            segments=all_timed,
        )

    def translate_segments(
        self, segments: List[TimedSegment]
    ) -> List[TimedSegment]:
        """Translate each segment's text from English to Spanish.

        Timing (start / end) and speaker_id are preserved exactly so that the
        downstream TTS step can place audio at the correct position.
        Uses argos-translate (local, ~30 MB en→es pack on first run).
        """
        import argostranslate.package
        import argostranslate.translate

        self._ensure_argos_en_es()

        translated: List[TimedSegment] = []
        for seg in segments:
            es_text = (
                argostranslate.translate.translate(seg.text, "en", "es")
                if seg.text else ""
            )
            translated.append(TimedSegment(
                speaker_id=seg.speaker_id,
                start=seg.start,
                end=seg.end,
                text=es_text,
            ))
        return translated

    def clone_voice_and_generate_speech(
        self,
        segments: List[TimedSegment],
        total_duration: float,
        notify_fn: Optional[Callable] = None,
    ) -> str:
        """Generate time-synchronised Spanish audio with per-segment caching."""
        from pydub import AudioSegment

        tts_cache_dir = os.path.join(self._temp_dir, "tts_cache")
        os.makedirs(tts_cache_dir, exist_ok=True)

        # Silent baseline of exactly total_duration
        # 16kHz Mono is enough for speech and saves massive RAM/disk space
        sample_rate = 16000
        baseline = AudioSegment.silent(
            duration=int(total_duration * 1000),  # pydub uses ms
            frame_rate=sample_rate,
        ).set_channels(1)

        total_count = len(segments)
        for idx, seg in enumerate(segments):
            if not seg.text.strip():
                continue  # leave this window as silence

            if notify_fn and (idx % 5 == 0 or idx == total_count - 1):
                notify_fn("Cloning Voice", f"Processing segment {idx+1}/{total_count}")

            original_ms = int((seg.end - seg.start) * 1000)
            if original_ms <= 0:
                continue

            # Cache check for the final stretched version
            seg_sig = f"{seg.text}_{seg.speaker_id}_{original_ms}"
            seg_hash = hashlib.md5(seg_sig.encode()).hexdigest()[:10]
            stretched_wav = os.path.join(tts_cache_dir, f"fin_{seg_hash}.wav")

            if os.path.exists(stretched_wav):
                stretched_audio = AudioSegment.from_wav(stretched_wav)
            else:
                # --- 5a: TTS → temp MP3 ---
                tts_mp3 = os.path.join(tts_cache_dir, f"tts_{idx}.mp3")
                self._run_tts(seg.text, seg.speaker_id, tts_mp3)

                # --- 5b: Convert TTS to WAV ---
                tts_wav = os.path.join(tts_cache_dir, f"tts_{idx}.wav")
                self._ffmpeg_convert(tts_mp3, tts_wav)

                # --- 5c: Measure TTS duration ---
                tts_audio = AudioSegment.from_wav(tts_wav)
                tts_ms = len(tts_audio)
                if tts_ms == 0:
                    continue

                # --- 5d: Speed ratio and atempo ---
                raw_ratio = tts_ms / original_ms
                speed_ratio = 1.0 + (raw_ratio - 1.0) * 0.5
                self._ffmpeg_atempo(tts_wav, stretched_wav, speed_ratio)
                stretched_audio = AudioSegment.from_wav(stretched_wav)
                
                # Cleanup intermediate
                try:
                    os.remove(tts_mp3)
                    os.remove(tts_wav)
                except: pass

            # --- 5e: Overlay on baseline ---
            # Trim to the original window
            stretched_audio = stretched_audio[: original_ms]
            offset_ms = int(seg.start * 1000)
            baseline = baseline.overlay(stretched_audio, position=offset_ms)

        # Export assembled audio as MP3 to avoid multi-GB WAV files
        assembled_path = os.path.join(self._temp_dir, "assembled_es.mp3")
        baseline.export(assembled_path, format="mp3", bitrate="128k")
        return assembled_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_speaker(
        start: float, end: float, diar: List[SpeakerSegment]
    ) -> str:
        """Return the speaker_id whose window overlaps [start, end] the most."""
        best_speaker = "speaker_1"
        best_overlap = -1.0
        for d in diar:
            overlap = min(end, d.end_time) - max(start, d.start_time)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d.speaker_id
        return best_speaker

    @staticmethod
    def _ensure_argos_en_es() -> None:
        """Download and install the argos-translate en→es pack if missing."""
        import argostranslate.package
        import argostranslate.translate

        installed = argostranslate.translate.get_installed_languages()
        if any(l.code == "en" for l in installed) and any(
            l.code == "es" for l in installed
        ):
            return
        available = argostranslate.package.get_available_packages()
        pkg = next(
            (p for p in available if p.from_code == "en" and p.to_code == "es"),
            None,
        )
        if pkg:
            pkg.install()

    # Voice mapping per speaker (add more speakers here as needed)
    _SPEAKER_VOICE: dict = {
        "speaker_1": "es-ES-AlvaroNeural",
        "speaker_2": "es-ES-ElviraNeural",
    }
    _DEFAULT_VOICE = "es-ES-AlvaroNeural"

    def _run_tts(self, text: str, speaker_id: str, out_mp3: str) -> None:
        """Synthesise *text* with edge-tts and save to *out_mp3*."""
        import edge_tts

        voice = self._SPEAKER_VOICE.get(speaker_id, self._DEFAULT_VOICE)

        async def _synth():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(out_mp3)

        asyncio.run(_synth())

    @staticmethod
    def _ffmpeg_convert(src: str, dst: str) -> None:
        """Convert audio file format via FFmpeg (e.g. MP3 → WAV)."""
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, dst],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def _ffmpeg_atempo(src: str, dst: str, speed_ratio: float) -> None:
        """Stretch/compress *src* by *speed_ratio* using the atempo filter.

        atempo is constrained to [0.5, 2.0]; chain multiple instances for
        values outside this range.
        """
        # Build chained atempo filter string
        # speed_ratio > 1 → speed up (TTS was too slow)
        # speed_ratio < 1 → slow down (TTS was too fast)
        ratio = max(0.1, min(speed_ratio, 10.0))  # guard extreme values

        filters: List[str] = []
        remaining = ratio
        if remaining >= 1.0:
            while remaining > 2.0:
                filters.append("atempo=2.0")
                remaining /= 2.0
            filters.append(f"atempo={remaining:.6f}")
        else:
            while remaining < 0.5:
                filters.append("atempo=0.5")
                remaining /= 0.5
            filters.append(f"atempo={remaining:.6f}")

        filter_str = ",".join(filters)
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-filter:a", filter_str, dst],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _to_mp3(self, input_path: str) -> tuple[str, float]:
        """Convert any supported audio/video to 128k MP3; returns (audio_path, duration_s)."""
        mp3_out = os.path.join(self._temp_dir, "source.mp3")
        
        # Use ffmpeg directly for much better control and performance
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vn",              # ignore video
            "-acodec", "libmp3lame",
            "-ab", "128k",      # small bitrate but good for speech
            "-ar", "44100",
            "-ac", "2",
            mp3_out
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with AudioFileClip(mp3_out) as clip:
            duration = clip.duration
        return mp3_out, duration

    def _export(self, audio_path: str, output_path: str) -> None:
        """Export the assembled audio to the desired output format."""
        clip = AudioFileClip(audio_path)
        ext = os.path.splitext(output_path)[1].lower()
        codec_map = {
            ".mp3": "libmp3lame",
            ".ogg": "libvorbis",
            ".aac": "aac",
            ".m4a": "aac",
        }
        codec = codec_map.get(ext)
        
        # If exporting to MP3/AAC, ensure we don't blow up size again
        if codec:
            clip.write_audiofile(output_path, codec=codec, bitrate="128k", logger=None)
        else:
            clip.write_audiofile(output_path, logger=None)
        clip.close()

    def _cleanup(self) -> None:
        if self._temp_dir and os.path.isdir(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    # ------------------------------------------------------------------
    # Legacy / compatibility shim
    # ------------------------------------------------------------------

    def translate_text(self, text: str) -> str:
        """Translate a single English string to Spanish (compatibility shim)."""
        self._ensure_argos_en_es()
        import argostranslate.translate
        return argostranslate.translate.translate(text, "en", "es") if text else ""
