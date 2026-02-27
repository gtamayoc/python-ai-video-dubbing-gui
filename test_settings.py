from src.services.translation_service import TranslationService
from src.services.tts_service import TTSService
from src.services.transcription_service import TranscriptionService
from src.services.diarization_service import DiarizationService
from src.services.dubbing_service import DubbingService
print("All services import OK")

svc = DubbingService()
print("DubbingService instantiated OK")
