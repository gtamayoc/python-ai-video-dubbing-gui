from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

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
