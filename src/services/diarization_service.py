import os
import sys
import logging
from typing import List, Optional, Callable

# Monkey-patch torchcodec to avoid DLL errors on Windows if it's broken
try:
    import torchcodec
except Exception:
    sys.modules["torchcodec"] = None

try:
    from moviepy.editor import AudioFileClip
except ImportError:
    from moviepy import AudioFileClip

from src.core.entities import SpeakerSegment

logger = logging.getLogger(__name__)

class DiarizationService:
    def perform_diarization(
        self, audio_path: str, notify_fn: Optional[Callable] = None
    ) -> List[SpeakerSegment]:
        hf_token = os.environ.get("HF_TOKEN")
        
        try:
            import torch
            import torchaudio
            from pyannote.audio import Pipeline as PyannotePipeline
            
            if not hf_token:
                logger.warning("HF_TOKEN not found in environment. Diarization will likely fail or use fallback.")
                # We let it proceed to trigger the specific 403/401 error handling below if it fails

            if notify_fn:
                notify_fn("Diarizing", "Cargando pyannote.audio e inicializando pipeline…")
            
            # use_auth_token is the most compatible parameter across pyannote versions
            pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token,
            )
            
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
            
            # Load audio manually with torchaudio to avoid internal decoding errors (like torchcodec/ffmpeg issues)
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Some versions of torchaudio might return different channel formats
            # Pyannote expects (channels, samples)
            diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
            
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(SpeakerSegment(
                    speaker_id=speaker,
                    start_time=turn.start,
                    end_time=turn.end,
                ))
            return segments if segments else self._fallback_diarization(audio_path)

        except ImportError:
            # Silent fallback if pyannote is not installed, but log internally
            logger.info("pyannote.audio not installed, using single-speaker fallback.")
            return self._fallback_diarization(audio_path)
            
        except Exception as e:
            msg = str(e)
            if "locate the file on the Hub" in msg or "403" in msg or "401" in msg:
                error_tip = (
                    "⚠ Error de acceso a Hugging Face. Asegúrate de:\n"
                    "1. Haber aceptado los términos en: 'hf.co/pyannote/speaker-diarization-3.1'\n"
                    "2. Y en: 'hf.co/pyannote/segmentation-3.0'\n"
                    "3. Que tu Token sea válido y tenga permisos de lectura."
                )
                if notify_fn:
                    notify_fn("Diarizing", error_tip)
                logger.error(f"Hugging Face Access Error: {e}")
            elif "libtorchcodec" in msg or "torchcodec" in msg:
                error_tip = (
                    "⚠ Error de dependencia (torchcodec).\n"
                    "Pyannote 3.1 tiene problemas con FFmpeg en Windows.\n"
                    "Se usará el fallback de un solo hablante."
                )
                if notify_fn:
                    notify_fn("Diarizing", error_tip)
                logger.error(f"TorchCodec Error: {e}")
            else:
                if notify_fn:
                    notify_fn("Diarizing", f"⚠ Diarization failed, using fallback: {e}")
                logger.error(f"Diarization error: {e}")
            return self._fallback_diarization(audio_path)

    def _fallback_diarization(self, audio_path: str) -> List[SpeakerSegment]:
        with AudioFileClip(audio_path) as clip:
            duration = clip.duration
        return [SpeakerSegment("speaker_1", 0.0, duration)]
