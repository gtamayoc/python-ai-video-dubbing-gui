import os
import subprocess
import shutil
import logging
from typing import Tuple, Optional, Callable
try:
    from moviepy.editor import AudioFileClip
except ImportError:
    from moviepy import AudioFileClip


logger = logging.getLogger(__name__)

def to_mp3(input_path: str, temp_dir: str) -> Tuple[str, float]:
    """Convert any supported audio/video to 128k MP3; returns (audio_path, duration_s)."""
    mp3_out = os.path.join(temp_dir, "source.mp3")
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-ab", "128k",
        "-ar", "44100",
        "-ac", "2",
        mp3_out,
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with AudioFileClip(mp3_out) as clip:
        duration = clip.duration
    return mp3_out, duration

def ffmpeg_atempo(src: str, dst: str, speed_ratio: float) -> None:
    """Stretch/compress *src* by *speed_ratio* using the atempo filter."""
    ratio = max(0.1, min(speed_ratio, 10.0))
    filters = []
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

    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-filter:a", ",".join(filters), dst],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def export_ffmpeg(audio_path: str, output_path: str) -> None:
    """Fallback export using ffmpeg directly for formats not handled by MoviePy."""
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-acodec", "libmp3lame" if output_path.endswith(".mp3") else "pcm_s16le",
            output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Fallo el exportado con FFmpeg: {e}")

def replace_audio_in_video(video_path: str, audio_path: str, output_path: str, progress_callback: Optional[Callable] = None) -> None:
    """Replace the audio track of a video with a new audio file using ffmpeg."""
    if progress_callback:
        progress_callback("Saving", "Uniendo audio traducido con el video original...")
        
    try:
        tmp_output = output_path + ".tmp.mp4"
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            tmp_output
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        shutil.move(tmp_output, output_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Failed to merge audio into video") from e
