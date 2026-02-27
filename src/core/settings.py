"""
src/core/settings.py
────────────────────
Central settings loader.

Reads ``config.yaml`` (repo root) and exposes one ``PipelineSettings``
dataclass that all services import.  Falls back to sensible defaults when
the YAML file is missing or a key is absent.

Usage::

    from src.core.settings import settings  # singleton
    print(settings.stt.model_size)
    print(settings.translation.backend)
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ─── Locate config.yaml ──────────────────────────────────────────────────────

def _find_config_yaml() -> Optional[str]:
    """Walk up from this file to find config.yaml."""
    candidate = os.path.join(os.getcwd(), "config.yaml")
    if os.path.exists(candidate):
        return candidate
    here = os.path.dirname(__file__)
    for _ in range(4):                     # Walk up at most 4 levels
        candidate = os.path.join(here, "config.yaml")
        if os.path.exists(candidate):
            return candidate
        here = os.path.dirname(here)
    return None


def _load_yaml(path: str) -> dict:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        logger.warning("PyYAML not installed — using built-in fallback parser for config.yaml")
        return _minimal_yaml_parse(path)
    except Exception as e:
        logger.warning("Could not load config.yaml: %s — using defaults", e)
        return {}


def _minimal_yaml_parse(path: str) -> dict:
    """Extremely simple key: value parser (no nesting) used if PyYAML is absent."""
    result: dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, _, v = line.partition(":")
                result[k.strip()] = v.strip()
    return result


# ─── Settings dataclasses ────────────────────────────────────────────────────

@dataclass
class STTSettings:
    model_size: str = "large-v3"
    language: str = "en"
    beam_size: int = 5
    vad_filter: bool = True
    condition_on_previous_text: bool = False


@dataclass
class DiarizationSettings:
    enabled: bool = True
    model: str = "pyannote/speaker-diarization-3.1"


@dataclass
class TranslationSettings:
    backend: str = "marian"            # marian | nllb | gguf | argos
    marian_model: str = "Helsinki-NLP/opus-mt-en-es"
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    nllb_src_lang: str = "eng_Latn"
    nllb_tgt_lang: str = "spa_Latn"
    gguf_repo: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    gguf_filename: str = "*q4_k_m*"
    source_lang: str = "en"
    target_lang: str = "es"
    batch_size: int = 8


@dataclass
class TTSSettings:
    backend: str = "edge"              # edge | xtts
    default_voice: str = "es-ES-AlvaroNeural"
    speaker_voices: Dict[str, str] = field(default_factory=lambda: {
        "speaker_1": "es-ES-AlvaroNeural",
        "speaker_2": "es-ES-ElviraNeural",
        "speaker_3": "es-MX-JorgeNeural",
        "speaker_4": "es-MX-DaliaNeural",
    })
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    xtts_language: str = "es"
    use_voice_cloning: bool = False
    reference_clip_duration: float = 6.0


@dataclass
class AlignmentSettings:
    max_atempo_ratio: float = 1.30
    min_atempo_ratio: float = 0.75
    silence_overflow_s: float = 2.0


@dataclass
class MergingSettings:
    merge_gap_s: float = 1.5
    merge_max_dur_s: float = 15.0


@dataclass
class ChunkSettings:
    size_s: int = 600
    overlap_s: int = 10


@dataclass
class ComputeSettings:
    device: str = "auto"
    compute_type: str = "auto"


@dataclass
class PipelineSettings:
    """Top-level container for all pipeline settings.

    Attributes:
        pipeline_mode: ``"simple"`` (legacy) or ``"advanced"`` (new high-quality).
    """
    pipeline_mode: str = "advanced"    # simple | advanced
    stt: STTSettings = field(default_factory=STTSettings)
    diarization: DiarizationSettings = field(default_factory=DiarizationSettings)
    translation: TranslationSettings = field(default_factory=TranslationSettings)
    tts: TTSSettings = field(default_factory=TTSSettings)
    alignment: AlignmentSettings = field(default_factory=AlignmentSettings)
    merging: MergingSettings = field(default_factory=MergingSettings)
    chunks: ChunkSettings = field(default_factory=ChunkSettings)
    compute: ComputeSettings = field(default_factory=ComputeSettings)


# ─── Builder ─────────────────────────────────────────────────────────────────

def _str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")


def _build_settings(raw: dict) -> PipelineSettings:
    s = PipelineSettings()
    s.pipeline_mode = raw.get("pipeline_mode", s.pipeline_mode)

    stt = raw.get("stt", {})
    if stt:
        s.stt.model_size = stt.get("model_size", s.stt.model_size)
        s.stt.language   = stt.get("language",   s.stt.language)
        s.stt.beam_size  = int(stt.get("beam_size", s.stt.beam_size))
        s.stt.vad_filter = _str_to_bool(stt.get("vad_filter", s.stt.vad_filter))
        s.stt.condition_on_previous_text = _str_to_bool(
            stt.get("condition_on_previous_text", s.stt.condition_on_previous_text)
        )

    diar = raw.get("diarization", {})
    if diar:
        s.diarization.enabled = _str_to_bool(diar.get("enabled", s.diarization.enabled))
        s.diarization.model   = diar.get("model", s.diarization.model)

    tr = raw.get("translation", {})
    if tr:
        s.translation.backend      = tr.get("backend",      s.translation.backend)
        s.translation.marian_model = tr.get("marian_model", s.translation.marian_model)
        s.translation.nllb_model   = tr.get("nllb_model",   s.translation.nllb_model)
        s.translation.nllb_src_lang = tr.get("nllb_src_lang", s.translation.nllb_src_lang)
        s.translation.nllb_tgt_lang = tr.get("nllb_tgt_lang", s.translation.nllb_tgt_lang)
        s.translation.gguf_repo    = tr.get("gguf_repo",    s.translation.gguf_repo)
        s.translation.gguf_filename = tr.get("gguf_filename", s.translation.gguf_filename)
        s.translation.source_lang  = tr.get("source_lang",  s.translation.source_lang)
        s.translation.target_lang  = tr.get("target_lang",  s.translation.target_lang)
        s.translation.batch_size   = int(tr.get("batch_size", s.translation.batch_size))

    tts = raw.get("tts", {})
    if tts:
        s.tts.backend    = tts.get("backend",       s.tts.backend)
        s.tts.default_voice = tts.get("default_voice", s.tts.default_voice)
        sv = tts.get("speaker_voices", {})
        if sv:
            s.tts.speaker_voices.update(sv)
        s.tts.xtts_model    = tts.get("xtts_model",    s.tts.xtts_model)
        s.tts.xtts_language = tts.get("xtts_language", s.tts.xtts_language)
        s.tts.use_voice_cloning = _str_to_bool(tts.get("use_voice_cloning", s.tts.use_voice_cloning))
        s.tts.reference_clip_duration = float(tts.get("reference_clip_duration", s.tts.reference_clip_duration))

    aln = raw.get("alignment", {})
    if aln:
        s.alignment.max_atempo_ratio   = float(aln.get("max_atempo_ratio",   s.alignment.max_atempo_ratio))
        s.alignment.min_atempo_ratio   = float(aln.get("min_atempo_ratio",   s.alignment.min_atempo_ratio))
        s.alignment.silence_overflow_s = float(aln.get("silence_overflow_s", s.alignment.silence_overflow_s))

    mrg = raw.get("merging", {})
    if mrg:
        s.merging.merge_gap_s    = float(mrg.get("merge_gap_s",    s.merging.merge_gap_s))
        s.merging.merge_max_dur_s = float(mrg.get("merge_max_dur_s", s.merging.merge_max_dur_s))

    chk = raw.get("chunks", {})
    if chk:
        s.chunks.size_s    = int(chk.get("size_s",    s.chunks.size_s))
        s.chunks.overlap_s = int(chk.get("overlap_s", s.chunks.overlap_s))

    cmp = raw.get("compute", {})
    if cmp:
        s.compute.device       = cmp.get("device",       s.compute.device)
        s.compute.compute_type = cmp.get("compute_type", s.compute.compute_type)

    return s


# ─── Singleton ───────────────────────────────────────────────────────────────

_cfg_path = _find_config_yaml()
if _cfg_path:
    logger.info("Loading pipeline config from: %s", _cfg_path)
    _raw = _load_yaml(_cfg_path)
else:
    logger.warning("config.yaml not found — using built-in defaults")
    _raw = {}

settings: PipelineSettings = _build_settings(_raw)

# Adjust STT model size based on pipeline mode if not explicitly set in YAML
# (simple mode uses a smaller model for speed unless the user forced a specific size)
if settings.pipeline_mode == "simple" and _raw.get("stt", {}).get("model_size") is None:
    settings.stt.model_size = "small"
