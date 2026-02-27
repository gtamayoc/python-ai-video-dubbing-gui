import yt_dlp
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)

def download_web_audio(url: str, output_path: str, progress_callback: Optional[Callable] = None) -> str:
    """Download best audio from web and save it as .wav using yt-dlp."""
    if progress_callback:
        progress_callback("Extracting", "Descargando audio de la web...")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': output_path.replace('.wav', ''),
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError("Unsupported platform or private video") from e

def download_web_video(url: str, output_path: str, progress_callback: Optional[Callable] = None) -> str:
    """Download best video containing audio from web using yt-dlp."""
    if progress_callback:
        progress_callback("Extracting", "Descargando video de la web...")
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
        'merge_output_format': 'mp4'
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError("Unsupported platform or private video") from e
