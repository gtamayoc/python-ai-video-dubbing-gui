"""
src/core/entities.py  (v2)
──────────────────────────
Core data models used across the pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SpeakerSegment:
    """One continuous time-window attributed to a single speaker."""
    speaker_id: str
    start_time: float   # seconds
    end_time:   float   # seconds


@dataclass
class TimedSegment:
    """A transcribed (and optionally translated) segment with full timing metadata.

    Attributes
    ----------
    speaker_id:
        Speaker label (e.g. ``"speaker_1"``).
    start, end:
        Segment boundaries in seconds.
    text:
        Current text — either the original transcription or the translated
        version depending on the pipeline stage.
    text_original:
        Preserved original transcription text (set after translation so that
        checkpoints can record both original and translated text side-by-side).
    """
    speaker_id: str
    start: float
    end:   float
    text:  str
    text_original: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptionResult:
    """Transcription output: full concatenated text + per-segment detail."""
    full_text: str
    segments:  List[TimedSegment] = field(default_factory=list)


@dataclass
class PipelineMetrics:
    """Profiling metrics for each pipeline stage."""
    stage_times:    Dict[str, float] = field(default_factory=dict)
    total_time:     float = 0.0
    audio_duration: float = 0.0
    realtime_factor: float = 0.0   # audio_duration / total_time

    def summary(self) -> str:
        lines = ["┌─── Pipeline Profiling ─────────────────"]
        for stage, t in self.stage_times.items():
            lines.append(f"│  {stage:<32s} {t:7.2f}s")
        lines.append(f"├────────────────────────────────────────")
        lines.append(f"│  {'TOTAL':<32s} {self.total_time:7.2f}s")
        if self.audio_duration > 0:
            lines.append(f"│  {'Audio duration':<32s} {self.audio_duration:7.2f}s")
            rf = self.audio_duration / max(self.total_time, 0.001)
            lines.append(f"│  {'Realtime factor':<32s} {rf:7.2f}×")
        lines.append(f"└────────────────────────────────────────")
        return "\n".join(lines)
