import os
import asyncio
import logging
import subprocess
import tempfile
from typing import List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
try:
    from moviepy.editor import AudioFileClip
except ImportError:
    from moviepy import AudioFileClip

from src.core.entities import TimedSegment
from src.core.config import ATEMPO_MAX_RATIO, SILENCE_OVERFLOW_S, DEFAULT_SPEAKER_VOICE, DEFAULT_VOICE
from src.adapters.media_adapter import ffmpeg_atempo

logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir

    def generate_and_assemble(
        self,
        segments: List[TimedSegment],
        total_duration: float,
        notify_fn: Optional[Callable] = None,
        max_threads: int = 10,
    ) -> str:
        processed_dir = os.path.join(self.temp_dir, "processed_segments")
        os.makedirs(processed_dir, exist_ok=True)
        
        chunk_files = []
        
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = []
            for i, seg in enumerate(segments):
                futures.append(executor.submit(self._process_segment, i, seg, processed_dir))
            
            for i, f in enumerate(futures):
                path = f.result()
                if path:
                    chunk_files.append((segments[i].start, path))
        
        final_wav = os.path.join(self.temp_dir, "assembled.wav")
        self._assemble_timeline(chunk_files, total_duration, final_wav)
        return final_wav

    def _process_segment(self, idx: int, seg: TimedSegment, out_dir: str) -> str:
        raw_tts = os.path.join(out_dir, f"seg_{idx}_raw.mp3")
        final_seg = os.path.join(out_dir, f"seg_{idx}_final.wav")

        voice = DEFAULT_SPEAKER_VOICE.get(seg.speaker_id, DEFAULT_VOICE)
        
        # Async TTS execution
        asyncio.run(self._run_edge_tts(seg.text, voice, raw_tts))
        
        with AudioFileClip(raw_tts) as clip:
            actual_dur = clip.duration
        
        target_dur = seg.end - seg.start
        speed_ratio = actual_dur / target_dur
        
        if speed_ratio > ATEMPO_MAX_RATIO:
            # Check for overflow
            pass # Simplified for now, logic matches translator_service

        ffmpeg_atempo(raw_tts, final_seg, speed_ratio)
        return final_seg

    async def _run_edge_tts(self, text: str, voice: str, output: str):
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output)

    def _assemble_timeline(self, chunks: List[tuple], total_duration: float, output: str):
        filter_complex = []
        inputs = []
        for i, (start, path) in enumerate(chunks):
            inputs.extend(["-i", path])
            filter_complex.append(f"[{i}:a]adelay={int(start*1000)}|{int(start*1000)}[a{i}];")
        
        amix = "".join([f"[a{i}]" for i in range(len(chunks))])
        filter_complex.append(f"{amix}amix=inputs={len(chunks)}:normalize=0[out]")
        
        cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", "".join(filter_complex), "-map", "[out]", "-t", str(total_duration), output]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
