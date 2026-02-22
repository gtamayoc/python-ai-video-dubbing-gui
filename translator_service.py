"""
translator_service.py
=====================
Backend pipeline for English → Spanish audio translation with voice cloning.

Five clearly-isolated placeholder methods ready for ML model integration.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional


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
class TranscriptionResult:
    """Transcription output mapped to speaker segments."""
    full_text: str
    segments: List[dict] = field(default_factory=list)


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
        """Run the full translation pipeline. Returns *output_path*."""

        def _notify(stage: str, msg: str) -> None:
            if progress_callback:
                progress_callback(stage, msg)

        self._temp_dir = tempfile.mkdtemp(prefix="translator_")

        try:
            # 1 — Extract / normalize audio
            _notify("Extracting", "Reading audio file…")
            wav_path = self._to_wav(input_path)
            _notify("Extracting", f"Audio ready → {wav_path}")

            # 2 — Speaker diarization
            _notify("Diarizing", "Detecting speakers…")
            segments = self.perform_diarization(wav_path)
            _notify("Diarizing", f"{len(segments)} segment(s) detected.")

            # 3 — Transcription
            _notify("Transcribing", "Transcribing English speech…")
            transcription = self.transcribe_audio(wav_path)
            _notify("Transcribing", f"Transcription done ({len(transcription.full_text)} chars).")

            # 4 — Translation
            _notify("Translating", "Translating to Spanish…")
            translated = self.translate_text(transcription.full_text)
            _notify("Translating", "Translation done.")

            # 5 — Voice cloning + speech generation
            _notify("Cloning Voice", "Generating Spanish speech per speaker…")
            generated_path = self.clone_voice_and_generate_speech(segments, translated)
            _notify("Cloning Voice", "Spanish audio generated.")

            # 6 — Save
            _notify("Saving", f"Writing output to {output_path}…")
            self._export(generated_path, output_path)
            _notify("Completed", f"Done → {output_path}")

            return output_path
        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    # Placeholder methods — swap for real ML models
    # ------------------------------------------------------------------

    def perform_diarization(self, wav_path: str) -> List[SpeakerSegment]:
        """Detect *who* speaks *when*.

        **Placeholder** — returns two fake segments.
        TODO: integrate pyannote.audio or similar.
        """
        return [
            SpeakerSegment("speaker_1", 0.0, 5.0),
            SpeakerSegment("speaker_2", 5.0, 10.0),
        ]

    def transcribe_audio(self, wav_path: str) -> TranscriptionResult:
        """Transcribe English speech to text using faster-whisper (local).

        Uses the 'base' model (~150 MB, downloaded on first run).
        Model sizes: tiny, base, small, medium, large-v3
        """
        from faster_whisper import WhisperModel

        model = WhisperModel("base", compute_type="int8")
        segments_iter, info = model.transcribe(wav_path, language="en")

        texts = []
        seg_list = []
        for seg in segments_iter:
            texts.append(seg.text.strip())
            seg_list.append({
                "speaker_id": "speaker_1",
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })

        full_text = " ".join(texts)

        return TranscriptionResult(full_text=full_text, segments=seg_list)

    def translate_text(self, text: str) -> str:
        """Translate English text to Spanish using argos-translate (local).

        Downloads the en→es language pack (~30 MB) on first run.
        """
        import argostranslate.package
        import argostranslate.translate

        if not text:
            return ""

        # Ensure the en→es package is installed
        available = argostranslate.package.get_available_packages()
        pkg = next(
            (p for p in available if p.from_code == "en" and p.to_code == "es"),
            None,
        )
        installed_langs = argostranslate.translate.get_installed_languages()
        en_installed = any(l.code == "en" for l in installed_langs)
        es_installed = any(l.code == "es" for l in installed_langs)

        if pkg and not (en_installed and es_installed):
            pkg.install()

        return argostranslate.translate.translate(text, "en", "es")

    def clone_voice_and_generate_speech(
        self, segments: List[SpeakerSegment], translated_text: str
    ) -> str:
        """Generate Spanish speech using edge-tts (Microsoft neural voice).

        Uses a single Spanish voice for all segments.
        Voice options: es-ES-AlvaroNeural, es-ES-ElviraNeural,
                       es-MX-DaliaNeural, es-MX-JorgeNeural, etc.
        """
        import asyncio
        import edge_tts

        if not translated_text:
            raise ValueError("No translated text to synthesize.")

        out_path = os.path.join(self._temp_dir, "cloned_es.mp3")

        async def _synth():
            communicate = edge_tts.Communicate(translated_text, "es-ES-AlvaroNeural")
            await communicate.save(out_path)

        asyncio.run(_synth())
        return out_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_wav(self, input_path: str) -> str:
        """Convert any supported audio to WAV in the temp dir."""
        from moviepy import AudioFileClip

        wav_out = os.path.join(self._temp_dir, "source.wav")
        clip = AudioFileClip(input_path)
        clip.write_audiofile(wav_out, logger=None)
        clip.close()
        return wav_out

    def _export(self, wav_path: str, output_path: str) -> None:
        """Export the generated WAV to the desired output format."""
        from moviepy import AudioFileClip

        clip = AudioFileClip(wav_path)
        ext = os.path.splitext(output_path)[1].lower()
        codec_map = {".mp3": "libmp3lame", ".ogg": "libvorbis", ".aac": "aac", ".m4a": "aac"}
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
