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

class TranslatorService:
    """Orchestrates the English → Spanish audio translation pipeline."""

    AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".wma", ".m4a"}

    def __init__(self) -> None:
        self._temp_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> str:
        """Run the full translation pipeline.  Returns *output_path*."""

        def _notify(stage: str, msg: str) -> None:
            if progress_callback:
                progress_callback(stage, msg)

        self._temp_dir = tempfile.mkdtemp(prefix="translator_")

        try:
            # 1 — Extract / normalise audio
            _notify("Extracting", "Reading audio file…")
            wav_path, total_duration = self._to_wav(input_path)
            _notify("Extracting", f"Audio ready ({total_duration:.2f}s) → {wav_path}")

            # 2 — Speaker diarisation
            _notify("Diarizing", "Detecting speakers…")
            diar_segments = self.perform_diarization(wav_path)
            _notify("Diarizing", f"{len(diar_segments)} segment(s) detected.")

            # 3 — Transcription (per segment, with timestamps)
            _notify("Transcribing", "Transcribing English speech…")
            transcription = self.transcribe_audio(wav_path, diar_segments)
            _notify("Transcribing",
                    f"Transcription done ({len(transcription.segments)} segment(s), "
                    f"{len(transcription.full_text)} chars).")

            # 4 — Translation (preserves segment timing)
            _notify("Translating", "Translating segments to Spanish…")
            translated_segments = self.translate_segments(transcription.segments)
            _notify("Translating", f"{len(translated_segments)} segment(s) translated.")

            # 5 — TTS per segment + time-stretch + assembly
            _notify("Cloning Voice", "Generating & synchronising Spanish speech…")
            assembled_path = self.clone_voice_and_generate_speech(
                translated_segments, total_duration
            )
            _notify("Cloning Voice", "Spanish audio assembled and synchronised.")

            # 6 — Save
            _notify("Saving", f"Writing output to {output_path}…")
            self._export(assembled_path, output_path)
            _notify("Completed", f"Done → {output_path}")

            return output_path
        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def perform_diarization(self, wav_path: str) -> List[SpeakerSegment]:
        """Detect *who* speaks *when*.

        **Placeholder** — returns two fake segments covering the full file.
        TODO: integrate pyannote.audio or similar.
        """
        # Fallback: treat whole file as speaker_1 (single segment).
        # Replace with real diarisation to get per-speaker windows.
        from moviepy import AudioFileClip
        with AudioFileClip(wav_path) as clip:
            duration = clip.duration
        return [
            SpeakerSegment("speaker_1", 0.0, duration),
        ]

    def transcribe_audio(
        self, wav_path: str, diar_segments: List[SpeakerSegment]
    ) -> TranscriptionResult:
        """Transcribe English speech to text using faster-whisper (local).

        Each Whisper segment is assigned the speaker whose diarisation window
        it overlaps the most.

        Model sizes: tiny | base | small | medium | large-v3
        """
        from faster_whisper import WhisperModel

        model = WhisperModel("base", compute_type="int8")
        segments_iter, _info = model.transcribe(wav_path, language="en")

        timed: List[TimedSegment] = []
        texts: List[str] = []
        for seg in segments_iter:
            text = seg.text.strip()
            speaker_id = self._assign_speaker(seg.start, seg.end, diar_segments)
            timed.append(TimedSegment(
                speaker_id=speaker_id,
                start=seg.start,
                end=seg.end,
                text=text,
            ))
            texts.append(text)

        return TranscriptionResult(
            full_text=" ".join(texts),
            segments=timed,
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
    ) -> str:
        """Generate time-synchronised Spanish audio.

        For every segment:
          1. Synthesise Spanish text with edge-tts (Microsoft neural voice).
          2. Apply FFmpeg atempo time-stretching at 50 % intensity so the
             speed change is subtle and never sounds artificial.
             Full ratio would be  tts_ms / original_ms ;  we instead use
             1.0 + (ratio - 1.0) * 0.5  so the effect is halved.
          3. Overlay the stretched segment at its start offset on a silent
             WAV baseline of *total_duration* seconds.  Any overflow past
             the segment window is trimmed; shortfall becomes natural silence.

        Returns the path to the assembled WAV file.
        """
        from pydub import AudioSegment

        # Silent baseline of exactly total_duration
        sample_rate = 22050
        baseline = AudioSegment.silent(
            duration=int(total_duration * 1000),  # pydub uses ms
            frame_rate=sample_rate,
        )

        for idx, seg in enumerate(segments):
            if not seg.text.strip():
                continue  # leave this window as silence

            original_ms = int((seg.end - seg.start) * 1000)
            if original_ms <= 0:
                continue

            # --- 5a: TTS → temp MP3 ------------------------------------------
            tts_mp3 = os.path.join(self._temp_dir, f"tts_{idx}.mp3")
            self._run_tts(seg.text, seg.speaker_id, tts_mp3)

            # --- 5b: Convert TTS to WAV for easier manipulation ---------------
            tts_wav = os.path.join(self._temp_dir, f"tts_{idx}.wav")
            self._ffmpeg_convert(tts_mp3, tts_wav)

            # --- 5c: Measure TTS duration -------------------------------------
            tts_audio = AudioSegment.from_wav(tts_wav)
            tts_ms = len(tts_audio)
            if tts_ms == 0:
                continue

            # --- 5d: Compute speed ratio (dampened to 50 %) and apply atempo --
            #
            # raw_ratio >1 → TTS is longer than the slot → we'd speed it up.
            # raw_ratio <1 → TTS is shorter than the slot → we'd slow it down.
            #
            # Applying the full ratio often produces jarring chipmunk / slow-mo
            # artefacts, especially for short audio clips.  We therefore apply
            # only *half* of the deviation from 1.0 so the effect is subtle:
            #
            #   adjusted = 1.0 + (raw_ratio - 1.0) * 0.5
            #
            # Examples:
            #   raw 2.0  → adjusted 1.5   (still speeds up, but less so)
            #   raw 0.5  → adjusted 0.75  (still slows down, but less so)
            #   raw 1.0  → adjusted 1.0   (no change)
            raw_ratio = tts_ms / original_ms
            speed_ratio = 1.0 + (raw_ratio - 1.0) * 0.5

            stretched_wav = os.path.join(self._temp_dir, f"stretched_{idx}.wav")
            self._ffmpeg_atempo(tts_wav, stretched_wav, speed_ratio)

            # --- 5e: Overlay on baseline at original start offset -------------
            stretched_audio = AudioSegment.from_wav(stretched_wav)
            # Trim to the original window; any shortfall is left as silence.
            stretched_audio = stretched_audio[: original_ms]
            offset_ms = int(seg.start * 1000)
            baseline = baseline.overlay(stretched_audio, position=offset_ms)

        # Export assembled audio
        assembled_path = os.path.join(self._temp_dir, "assembled_es.wav")
        baseline.export(assembled_path, format="wav")
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

    def _to_wav(self, input_path: str) -> tuple[str, float]:
        """Convert any supported audio/video to WAV; returns (wav_path, duration_s)."""
        from moviepy import AudioFileClip

        wav_out = os.path.join(self._temp_dir, "source.wav")
        clip = AudioFileClip(input_path)
        duration = clip.duration
        clip.write_audiofile(wav_out, logger=None)
        clip.close()
        return wav_out, duration

    def _export(self, wav_path: str, output_path: str) -> None:
        """Export the assembled WAV to the desired output format."""
        from moviepy import AudioFileClip

        clip = AudioFileClip(wav_path)
        ext = os.path.splitext(output_path)[1].lower()
        codec_map = {
            ".mp3": "libmp3lame",
            ".ogg": "libvorbis",
            ".aac": "aac",
            ".m4a": "aac",
        }
        codec = codec_map.get(ext)
        if codec:
            clip.write_audiofile(output_path, codec=codec, logger=None)
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
