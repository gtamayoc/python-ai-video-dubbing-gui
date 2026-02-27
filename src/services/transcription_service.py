import os
import math
import subprocess
import logging
import re
from typing import List, Optional, Callable
import langdetect
from src.core.model_manager import models
from src.core.entities import SpeakerSegment, TimedSegment, TranscriptionResult
from src.core.config import CHUNK_SIZE_S, CHUNK_OVERLAP_S

logger = logging.getLogger(__name__)

class TranscriptionService:
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir

    def transcribe(
        self,
        audio_path: str,
        diar_segments: List[SpeakerSegment],
        total_duration: float,
        notify_fn: Optional[Callable[[str, str], None]] = None,
    ) -> TranscriptionResult:
        model = models.get_whisper()
        all_timed: List[TimedSegment] = []
        all_texts: List[str] = []
        conf_reported = False
        seen_ranges: set = set()

        num_chunks = math.ceil(total_duration / CHUNK_SIZE_S)

        for i in range(num_chunks):
            start_off = i * CHUNK_SIZE_S
            dur = min(CHUNK_SIZE_S + CHUNK_OVERLAP_S, total_duration - start_off)

            if notify_fn:
                notify_fn("Transcribing", f"Chunk {i+1}/{num_chunks} ({int(start_off)}s – {int(start_off+dur)}s)")

            chunk_mp3 = os.path.join(self.temp_dir, f"chunk_{i}.mp3")
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(start_off), "-t", str(dur),
                "-i", audio_path,
                "-acodec", "libmp3lame", "-ab", "64k",
                chunk_mp3
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            c_segments_gen, _ = model.transcribe(
                chunk_mp3,
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=100),
                condition_on_previous_text=False,
            )

            for seg in c_segments_gen:
                seg_text = seg.text.strip()
                if not seg_text:
                    continue

                abs_start = round(seg.start + start_off, 2)
                abs_end   = round(seg.end   + start_off, 2)

                dedup_key = (round(abs_start, 1), round(abs_end, 1))
                if dedup_key in seen_ranges:
                    continue
                seen_ranges.add(dedup_key)

                seg_text = re.sub(r'([?!])\1+', r'\1', seg_text)
                if seg_text and seg_text[0].islower():
                    seg_text = seg_text[0].upper() + seg_text[1:]

                speaker_id = self._assign_speaker(abs_start, abs_end, diar_segments)
                all_timed.append(TimedSegment(
                    speaker_id=speaker_id,
                    start=abs_start,
                    end=abs_end,
                    text=seg_text,
                ))
                all_texts.append(seg_text)

            if not conf_reported and len(all_texts) > 0:
                try:
                    lang = langdetect.detect(" ".join(all_texts[:5]))
                    if notify_fn:
                        notify_fn("Transcribing", f"✅ Idioma detectado: '{lang}'")
                except Exception:
                    pass
                conf_reported = True

            try:
                os.remove(chunk_mp3)
            except Exception:
                pass

        all_timed.sort(key=lambda s: s.start)
        return TranscriptionResult(full_text=" ".join(all_texts), segments=all_timed)

    def _assign_speaker(self, start: float, end: float, diar_segments: List[SpeakerSegment]) -> str:
        mid = (start + end) / 2
        for s in diar_segments:
            if s.start_time <= mid <= s.end_time:
                return s.speaker_id
        return "speaker_1"
