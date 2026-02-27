import os
import time
import logging
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)

class ModelManager:
    """Thread-safe singleton that keeps ML models loaded in memory."""

    _instance: Optional["ModelManager"] = None
    _lock = Lock()

    def __new__(cls) -> "ModelManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._whisper_model = None
        self._whisper_lock = Lock()
        self._argos_ready = False
        self._argos_lock = Lock()
        self._device: str = "auto"
        self._compute_type: str = "auto"
        self._whisper_model_size: str = "small"

    def configure(
        self,
        device: str = "auto",
        compute_type: str = "auto",
        model_size: str = "small",
    ) -> None:
        self._device = device
        self._compute_type = compute_type
        self._whisper_model_size = model_size
        self._whisper_model = None
        self._argos_ready = False

    def _resolve_device(self) -> str:
        if self._device != "auto":
            return self._device
        try:
            import torch
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_mem
                logger.info(
                    "CUDA available — GPU: %s, VRAM: %.1f GB",
                    torch.cuda.get_device_name(0),
                    vram / 1e9,
                )
                return "cuda"
        except ImportError:
            pass
        logger.info("CUDA not available — falling back to CPU")
        return "cpu"

    def _resolve_compute_type(self, device: str) -> str:
        if self._compute_type != "auto":
            return self._compute_type
        if device == "cuda":
            return "float16"
        return "int8"

    def get_whisper(self):
        with self._whisper_lock:
            if self._whisper_model is None:
                from faster_whisper import WhisperModel
                device = self._resolve_device()
                ctype = self._resolve_compute_type(device)
                logger.info("Loading Whisper '%s' on %s (%s)…", self._whisper_model_size, device, ctype)
                t0 = time.perf_counter()
                system_cores = os.cpu_count() or 4
                safe_threads = max(1, system_cores - 2)
                self._whisper_model = WhisperModel(
                    self._whisper_model_size,
                    device=device,
                    compute_type=ctype,
                    cpu_threads=safe_threads,
                    num_workers=max(1, safe_threads // 2),
                )
                logger.info("Whisper loaded in %.2fs", time.perf_counter() - t0)
            return self._whisper_model

    def ensure_argos(self) -> None:
        with self._argos_lock:
            if self._argos_ready:
                return
            import argostranslate.package
            import argostranslate.translate
            installed_langs = argostranslate.translate.get_installed_languages()
            en_ok = any(lang.code == "en" for lang in installed_langs)
            es_ok = any(lang.code == "es" for lang in installed_langs)
            if not (en_ok and es_ok):
                logger.info("Installing argos-translate en→es package…")
                available = argostranslate.package.get_available_packages()
                pkg = next((p for p in available if p.from_code == "en" and p.to_code == "es"), None)
                if pkg:
                    pkg.install()
            self._argos_ready = True

    def preload_all(self) -> float:
        t0 = time.perf_counter()
        self.get_whisper()
        self.ensure_argos()
        return time.perf_counter() - t0

    def release(self) -> None:
        with self._whisper_lock:
            if self._whisper_model is not None:
                del self._whisper_model
                self._whisper_model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        self._argos_ready = False
        logger.info("Models released.")

# Global instance
models = ModelManager()
