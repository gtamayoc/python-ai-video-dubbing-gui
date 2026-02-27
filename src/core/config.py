# Pipeline Configuration Constants

CHUNK_SIZE_S       = 600   # 10-minute primary chunks for STT
CHUNK_OVERLAP_S    = 10    # Overlap to avoid clipping words at chunk boundary
MERGE_GAP_S        = 1.5   # Max silence gap to merge consecutive same-speaker segments
MERGE_MAX_DUR_S    = 15.0  # Max duration of a merged segment
ATEMPO_MAX_RATIO   = 1.20  # Maximum allowed speed-up ratio before we prefer overflow
SILENCE_OVERFLOW_S  = 2.0  # If next segment is ≥ this many seconds away, allow TTS to overflow
CHARS_PER_SEC      = 15    # Approx characters per second for isochrony hints
TTS_BATCH_SIZE     = 8     # How many segments to send to LLM per batch

# Default Speaker Voice Mapping
DEFAULT_SPEAKER_VOICE = {
    "speaker_1": "es-ES-AlvaroNeural",
    "speaker_2": "es-ES-ElviraNeural",
    "speaker_3": "es-MX-JorgeNeural",
    "speaker_4": "es-MX-DaliaNeural",
}
DEFAULT_VOICE = "es-ES-AlvaroNeural"

# Supported Audio Extensions
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".wma", ".m4a"}
