"""
src/core/config.py  (v2 – kept for backward compatibility)

All pipeline parameters are now in config.yaml / settings.py.
This file keeps legacy constants that other modules import.

NOTE: We avoid importing settings at module level here to prevent
circular-import / blocking startup. Constants are defined with defaults
and updated lazily when the settings module is explicitly requested.
"""

# ── Legacy constants (defaults match config.yaml) ─────────────────────────────

CHUNK_SIZE_S        = 600
CHUNK_OVERLAP_S     = 10
MERGE_GAP_S         = 1.5
MERGE_MAX_DUR_S     = 15.0
ATEMPO_MAX_RATIO    = 1.30
SILENCE_OVERFLOW_S  = 2.0
TTS_BATCH_SIZE      = 8
CHARS_PER_SEC       = 15

# Default Speaker Voice Mapping (edge-tts voices)
DEFAULT_SPEAKER_VOICE = {
    "speaker_1": "es-ES-AlvaroNeural",
    "speaker_2": "es-ES-ElviraNeural",
    "speaker_3": "es-MX-JorgeNeural",
    "speaker_4": "es-MX-DaliaNeural",
}
DEFAULT_VOICE = "es-ES-AlvaroNeural"

# Supported Audio Extensions
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".wma", ".m4a"}
