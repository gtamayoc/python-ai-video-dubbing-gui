"""
src/core/model_manager.py  (v2 – advanced pipeline)
─────────────────────────────────────────────────────
Thread-safe singleton that keeps ALL heavy ML models loaded in memory.

Added in v2:
  • get_marian()   → MarianMT en→es translation model (transformers)
  • get_nllb()     → NLLB-200 multilingual translation model
  • get_xtts()     → Coqui XTTS v2 TTS model
  • configure_from_settings()  → reads src.core.settings at runtime
"""

from __future__ import annotations

import os
import time
import logging
from threading import Lock
from typing import Optional, Tuple

# Force huggingface_hub to never use symlinks on Windows to prevent WinError 1314
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
try:
    import huggingface_hub.file_download
    huggingface_hub.file_download.are_symlinks_supported = lambda *args, **kwargs: False
except ImportError:
    pass

logger = logging.getLogger(__name__)


class ModelManager:
    """Thread-safe singleton that keeps ML models loaded in memory.

    Design principles
    -----------------
    * Each model family has its own Lock so multiple model types can
      be loaded concurrently when the GUI first starts.
    * Models are loaded lazily (on first use) to avoid blocking startup.
    * The ``release()`` method unloads all models and frees VRAM/RAM.
    """

    _instance: Optional["ModelManager"] = None
    _lock: Lock = Lock()

    # ── Singleton ────────────────────────────────────────────────────────────

    def __new__(cls) -> "ModelManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:                       # type: ignore[attr-defined]
            return
        self._initialized = True

        # ── faster-whisper ───────────────────────────────────────────────────
        self._whisper_model = None
        self._whisper_lock = Lock()
        self._whisper_model_size: str = "large-v3"

        # ── Translation ──────────────────────────────────────────────────────
        self._marian_model = None
        self._marian_tokenizer = None
        self._marian_lock = Lock()
        self._marian_model_id: str = "Helsinki-NLP/opus-mt-en-es"

        self._nllb_model = None
        self._nllb_tokenizer = None
        self._nllb_lock = Lock()
        self._nllb_model_id: str = "facebook/nllb-200-distilled-600M"
        self._nllb_src_lang: str = "eng_Latn"
        self._nllb_tgt_lang: str = "spa_Latn"

        self._gguf_model = None
        self._gguf_lock = Lock()
        self._gguf_repo: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
        self._gguf_filename: str = "*q4_k_m*"

        self._argos_ready: bool = False
        self._argos_lock = Lock()

        # ── TTS (XTTS v2) ────────────────────────────────────────────────────
        self._xtts_model = None
        self._xtts_lock = Lock()
        self._xtts_model_id: str = "tts_models/multilingual/multi-dataset/xtts_v2"

        # ── Compute config ───────────────────────────────────────────────────
        self._device: str = "auto"
        self._compute_type: str = "auto"

    # ── Config ───────────────────────────────────────────────────────────────

    def configure_from_settings(self) -> None:
        """Apply settings from ``src.core.settings.settings`` to the manager."""
        try:
            from src.core.settings import settings as cfg
            self._whisper_model_size = cfg.stt.model_size
            self._marian_model_id    = cfg.translation.marian_model
            self._nllb_model_id      = cfg.translation.nllb_model
            self._nllb_src_lang      = cfg.translation.nllb_src_lang
            self._nllb_tgt_lang      = cfg.translation.nllb_tgt_lang
            self._gguf_repo          = cfg.translation.gguf_repo
            self._gguf_filename      = cfg.translation.gguf_filename
            self._xtts_model_id      = cfg.tts.xtts_model
            self._device             = cfg.compute.device
            self._compute_type       = cfg.compute.compute_type
            logger.info("ModelManager configured from settings.py")
        except Exception as e:
            logger.warning("Could not configure ModelManager from settings: %s", e)

    def configure(
        self,
        device: str = "auto",
        compute_type: str = "auto",
        model_size: str = "large-v3",
    ) -> None:
        """Legacy helper – still works for callers that set params directly."""
        self._device = device
        self._compute_type = compute_type
        self._whisper_model_size = model_size
        # Invalidate cached models so they are reloaded with new settings
        self._whisper_model = None
        self._argos_ready = False

    # ── Device helpers ───────────────────────────────────────────────────────

    def _resolve_device(self) -> str:
        if self._device != "auto":
            return self._device
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                logger.info(
                    "CUDA available — GPU: %s, VRAM: %.1f GB",
                    props.name,
                    props.total_memory / 1e9,
                )
                return "cuda"
        except ImportError:
            pass
        logger.info("CUDA not available — falling back to CPU")
        return "cpu"

    def _resolve_compute_type(self, device: str) -> str:
        if self._compute_type != "auto":
            return self._compute_type
        return "float16" if device == "cuda" else "int8"

    def _torch_device(self):
        """Return a ``torch.device`` matching ``_resolve_device()``."""
        import torch
        return torch.device(self._resolve_device())

    # ── faster-whisper ───────────────────────────────────────────────────────

    def get_whisper(self):
        """Return the loaded faster-whisper ``WhisperModel`` (lazy, cached)."""
        with self._whisper_lock:
            if self._whisper_model is None:
                from faster_whisper import WhisperModel
                device = self._resolve_device()
                ctype  = self._resolve_compute_type(device)
                logger.info(
                    "Loading Whisper '%s' on %s (%s)…",
                    self._whisper_model_size, device, ctype,
                )
                t0 = time.perf_counter()
                cores = os.cpu_count() or 4
                threads = max(1, cores - 2)
                self._whisper_model = WhisperModel(
                    self._whisper_model_size,
                    device=device,
                    compute_type=ctype,
                    cpu_threads=threads,
                    num_workers=max(1, threads // 2),
                )
                logger.info("Whisper loaded in %.2fs", time.perf_counter() - t0)
            return self._whisper_model

    # ── MarianMT (Helsinki-NLP) ──────────────────────────────────────────────

    def get_marian(self) -> Tuple:
        """Return ``(model, tokenizer)`` for MarianMT (lazy, cached).

        MarianMT is the recommended default for en→es: ~600 MB, fast on CPU,
        supervised seq2seq – minimal hallucinations.
        """
        with self._marian_lock:
            if self._marian_model is None:
                import torch
                from transformers import MarianMTModel, MarianTokenizer
                logger.info("Loading MarianMT '%s'…", self._marian_model_id)
                t0 = time.perf_counter()
                self._marian_tokenizer = MarianTokenizer.from_pretrained(
                    self._marian_model_id
                )
                self._marian_model = MarianMTModel.from_pretrained(
                    self._marian_model_id
                )
                device = self._torch_device()
                # Use float16 on GPU for speed, float32 on CPU for compatibility
                if device.type == "cuda":
                    self._marian_model = self._marian_model.half().to(device)
                else:
                    self._marian_model = self._marian_model.to(device)
                self._marian_model.eval()
                logger.info("MarianMT loaded in %.2fs", time.perf_counter() - t0)
            return self._marian_model, self._marian_tokenizer

    # ── NLLB-200 ─────────────────────────────────────────────────────────────

    def get_nllb(self) -> Tuple:
        """Return ``(model, tokenizer)`` for NLLB-200 (lazy, cached).

        Better coverage for non-English source languages at the cost of
        slightly higher memory usage (~1.2 GB) vs MarianMT.
        """
        with self._nllb_lock:
            if self._nllb_model is None:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                logger.info("Loading NLLB '%s'…", self._nllb_model_id)
                t0 = time.perf_counter()
                self._nllb_tokenizer = AutoTokenizer.from_pretrained(
                    self._nllb_model_id,
                    src_lang=self._nllb_src_lang,
                )
                self._nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self._nllb_model_id
                )
                device = self._torch_device()
                if device.type == "cuda":
                    self._nllb_model = self._nllb_model.half().to(device)
                else:
                    self._nllb_model = self._nllb_model.to(device)
                self._nllb_model.eval()
                logger.info("NLLB loaded in %.2fs", time.perf_counter() - t0)
            return self._nllb_model, self._nllb_tokenizer

    # ── GGUF / LLM (Qwen2.5 via llama-cpp-python) ───────────────────────────

    def get_gguf(self):
        """Return the loaded Llama GGUF model (lazy, cached)."""
        with self._gguf_lock:
            if self._gguf_model is None:
                from llama_cpp import Llama
                cores = os.cpu_count() or 4
                threads = max(1, cores - 2)
                logger.info("Loading GGUF '%s' (file=%s)…", self._gguf_repo, self._gguf_filename)
                t0 = time.perf_counter()
                self._gguf_model = Llama.from_pretrained(
                    repo_id=self._gguf_repo,
                    filename=self._gguf_filename,
                    n_ctx=4096,
                    n_threads=threads,
                    n_gpu_layers=-1,
                    verbose=False,
                )
                logger.info("GGUF loaded in %.2fs", time.perf_counter() - t0)
            return self._gguf_model

    # ── Legacy argostranslate ────────────────────────────────────────────────

    def ensure_argos(self) -> None:
        """Ensure Argos en→es package is installed (legacy simple mode)."""
        with self._argos_lock:
            if self._argos_ready:
                return
            import argostranslate.package
            import argostranslate.translate
            installed = argostranslate.translate.get_installed_languages()
            en_ok = any(l.code == "en" for l in installed)
            es_ok = any(l.code == "es" for l in installed)
            if not (en_ok and es_ok):
                logger.info("Installing argostranslate en→es package…")
                pkgs = argostranslate.package.get_available_packages()
                pkg = next(
                    (p for p in pkgs if p.from_code == "en" and p.to_code == "es"),
                    None,
                )
                if pkg:
                    pkg.install()
            self._argos_ready = True

    # ── XTTS v2 ──────────────────────────────────────────────────────────────

    def get_xtts(self):
        """Return loaded Coqui XTTS v2 model (lazy, cached).

        Voice-cloning TTS – requires ``coqui-tts`` (TTS package from Coqui).
        Falls back gracefully if not installed.
        """
        with self._xtts_lock:
            if self._xtts_model is None:
                try:
                    from TTS.api import TTS as CoquiTTS
                    logger.info("Loading XTTS v2 '%s'…", self._xtts_model_id)
                    t0 = time.perf_counter()
                    device = self._resolve_device()
                    self._xtts_model = CoquiTTS(self._xtts_model_id).to(device)
                    logger.info("XTTS v2 loaded in %.2fs", time.perf_counter() - t0)
                except Exception as e:
                    logger.error("XTTS v2 not available: %s  — TTS will use edge-tts", e)
                    self._xtts_model = None
            return self._xtts_model

    # ── Preload ───────────────────────────────────────────────────────────────

    def preload_all(self) -> float:
        """Eagerly load all models configured for the current pipeline mode.

        Returns total elapsed seconds.  Useful to warm up during app start.
        """
        from src.core.settings import settings as cfg
        t0 = time.perf_counter()
        self.get_whisper()
        if cfg.translation.backend == "marian":
            self.get_marian()
        elif cfg.translation.backend == "nllb":
            self.get_nllb()
        elif cfg.translation.backend == "gguf":
            try:
                self.get_gguf()
            except Exception:
                pass
        elif cfg.translation.backend == "argos":
            self.ensure_argos()
        if cfg.tts.backend == "xtts":
            self.get_xtts()
        return time.perf_counter() - t0

    # ── Release ───────────────────────────────────────────────────────────────

    def release(self) -> None:
        """Unload all models and free GPU memory."""
        with self._whisper_lock:
            self._whisper_model  = None
        with self._marian_lock:
            self._marian_model      = None
            self._marian_tokenizer  = None
        with self._nllb_lock:
            self._nllb_model        = None
            self._nllb_tokenizer    = None
        with self._gguf_lock:
            self._gguf_model        = None
        self._argos_ready = False
        with self._xtts_lock:
            self._xtts_model        = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("All models released and VRAM cleared.")


# ── Global singleton ──────────────────────────────────────────────────────────
# configure_from_settings() is NOT called here to avoid circular imports and
# blocking I/O at module load time.  Each service calls it lazily on first use.

models = ModelManager()
