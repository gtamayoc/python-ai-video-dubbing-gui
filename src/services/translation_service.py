"""
src/services/translation_service.py  (v2 – advanced + legacy)
──────────────────────────────────────────────────────────────
Pluggable translation backend with four implementations:

  ┌──────────────────────────────────────────────────────────────┐
  │ backend     │ model                          │ quality │ speed│
  ├─────────────┼────────────────────────────────┼─────────┼──────┤
  │ "marian"    │ Helsinki-NLP/opus-mt-en-es     │ ★★★★☆  │ fast │
  │ "nllb"      │ facebook/nllb-200-distilled-   │ ★★★★★  │ med  │
  │             │   600M                         │         │      │
  │ "gguf"      │ Qwen2.5-1.5B-Instruct-GGUF     │ ★★★★☆  │ slow │
  │ "argos"     │ argostranslate (legacy)         │ ★★☆☆☆  │ fast │
  └─────────────┴────────────────────────────────┴─────────┴──────┘

The backend is selected from ``config.yaml`` → ``translation.backend``.
You can override it at runtime via ``TranslationService(backend="nllb")``.

Public API (unchanged from v1):
    svc = TranslationService()
    svc.load_model(notify_fn)
    translated = svc.translate_segments(segments, notify_fn)
"""

from __future__ import annotations

import copy
import json
import logging
import math
import re
import time
from typing import Callable, List, Optional

from src.core.entities import TimedSegment
from src.core.settings import settings as cfg

logger = logging.getLogger(__name__)

# Minimum acceptable length of a translated segment (used to detect empty/failed output)
_MIN_TRANSLATION_LEN = 2


class TranslationService:
    """Translates ``List[TimedSegment]`` from English to a target language.

    Parameters
    ----------
    backend:
        Override the backend from ``config.yaml``.  One of:
        ``"marian"`` | ``"nllb"`` | ``"gguf"`` | ``"argos"``.
    """

    def __init__(self, backend: Optional[str] = None) -> None:
        # ── Choose backend ────────────────────────────────────────────────────
        self._backend: str = backend or cfg.translation.backend
        logger.info("TranslationService: backend=%s", self._backend)

        # Internal state for legacy GGUF path
        self._llm = None
        self._llm_tokenizer = None
        self._llm_backend: str = "none"

    # ── Public: load_model ────────────────────────────────────────────────────

    def load_model(self, notify_fn: Optional[Callable] = None) -> None:
        """Eagerly load the translation model.

        This is optional – ``translate_segments`` will load it lazily if
        ``load_model`` was not called explicitly.

        Parameters
        ----------
        notify_fn:
            ``(stage: str, message: str) → None`` callback for progress updates.
        """
        from src.core.model_manager import models
        b = self._backend

        if b == "marian":
            if notify_fn:
                notify_fn("Translating", "Cargando MarianMT (Helsinki-NLP/opus-mt-en-es)…")
            models.get_marian()

        elif b == "nllb":
            if notify_fn:
                notify_fn("Translating", "Cargando NLLB-200 (facebook/nllb-200-distilled-600M)…")
            models.get_nllb()

        elif b == "gguf":
            if notify_fn:
                notify_fn("Translating", "Cargando LLM GGUF (Qwen2.5-1.5B)…")
            try:
                self._llm = models.get_gguf()
                self._llm_backend = "gguf"
            except Exception as e:
                logger.warning("GGUF unavailable (%s) — falling back to Marian", e)
                self._backend = "marian"
                models.get_marian()

        elif b == "argos":
            if notify_fn:
                notify_fn("Translating", "Preparando argostranslate (en→es)…")
            models.ensure_argos()

        else:
            logger.warning("Unknown translation backend '%s' — defaulting to marian", b)
            self._backend = "marian"
            models.get_marian()

    # ── Public: translate_segments ────────────────────────────────────────────

    def translate_segments(
        self,
        segments: List[TimedSegment],
        notify_fn: Optional[Callable[[str, str], None]] = None,
    ) -> List[TimedSegment]:
        """Translate all segments in *segments*.

        Segments are first merged by speaker/gap to build more context-rich
        batches, translated, then the timing from the original segments is
        re-applied to the translated output.

        Parameters
        ----------
        segments:
            Input ``TimedSegment`` objects with ``text`` in the source language.
        notify_fn:
            Progress callback ``(stage, message)``.

        Returns
        -------
        List[TimedSegment]
            Same structure as input but with ``text`` replaced by the
            translated version.  Timing is preserved.
        """
        if not segments:
            return []

        # Merge neighbouring same-speaker segments for better translation context
        merged = self._merge_segments(segments)
        total  = len(merged)

        if notify_fn:
            notify_fn("Translating", f"Traduciendo {total} segmentos (backend: {self._backend})…")

        t0 = time.perf_counter()

        # ── Dispatch to active backend ────────────────────────────────────────
        if self._backend == "marian":
            translated_texts = self._translate_marian(
                [s.text for s in merged], notify_fn, total
            )
        elif self._backend == "nllb":
            translated_texts = self._translate_nllb(
                [s.text for s in merged], notify_fn, total
            )
        elif self._backend == "gguf":
            translated_texts = self._translate_gguf(merged, notify_fn, total)
        elif self._backend == "argos":
            translated_texts = self._translate_argos(
                [s.text for s in merged], notify_fn, total
            )
        else:
            logger.warning("Unrecognised backend at translate time – using marian fallback")
            translated_texts = self._translate_marian(
                [s.text for s in merged], notify_fn, total
            )

        elapsed = time.perf_counter() - t0
        logger.info("Translation done in %.2fs  (%.1f seg/s)", elapsed, total / max(elapsed, 0.01))

        # ── Re-attach timing and speaker metadata ─────────────────────────────
        result: List[TimedSegment] = []
        for seg, new_text in zip(merged, translated_texts):
            good = new_text and len(new_text.strip()) >= _MIN_TRANSLATION_LEN
            result.append(TimedSegment(
                speaker_id=seg.speaker_id,
                start=seg.start,
                end=seg.end,
                text=new_text.strip() if good else seg.text,   # fallback to original
            ))

        if notify_fn:
            notify_fn("Translating", f"✅ Traducción completada ({len(result)} segmentos en {elapsed:.1f}s)")

        return result

    # ── Backend: MarianMT ─────────────────────────────────────────────────────

    def _translate_marian(
        self,
        texts: List[str],
        notify_fn: Optional[Callable],
        total: int,
    ) -> List[str]:
        """Batch-translate *texts* with MarianMT.

        MarianMT is a supervised seq2seq model trained specifically for
        en→es translation – minimal hallucinations, deterministic output,
        very fast even on CPU.

        Batch size is from ``cfg.translation.batch_size``.
        """
        from src.core.model_manager import models
        import torch

        mt_model, tokenizer = models.get_marian()
        device = next(mt_model.parameters()).device
        batch_size = cfg.translation.batch_size
        results: List[str] = []

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            num_batches = math.ceil(len(texts) / batch_size)

            if notify_fn:
                notify_fn(
                    "Translating",
                    f"MarianMT: lote {batch_num}/{num_batches} ({len(batch)} segs)…",
                )

            try:
                # Tokenize
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)

                with torch.no_grad():
                    translated_ids = mt_model.generate(
                        **inputs,
                        num_beams=4,
                        max_length=512,
                        early_stopping=True,
                    )

                batch_results = tokenizer.batch_decode(
                    translated_ids,
                    skip_special_tokens=True,
                )
                results.extend(batch_results)

            except Exception as e:
                logger.error("MarianMT batch %d failed: %s — keeping original", batch_num, e)
                results.extend(batch)   # Fallback: keep source text

        return results

    # ── Backend: NLLB-200 ────────────────────────────────────────────────────

    def _translate_nllb(
        self,
        texts: List[str],
        notify_fn: Optional[Callable],
        total: int,
    ) -> List[str]:
        """Batch-translate *texts* with NLLB-200.

        Use this for non-English source languages or when MarianMT quality
        is insufficient.  Supports 200 languages via BCP-47 codes.
        """
        from src.core.model_manager import models
        import torch

        nllb_model, tokenizer = models.get_nllb()
        device = next(nllb_model.parameters()).device
        batch_size = cfg.translation.batch_size
        results: List[str] = []

        # Resolve target language token ID
        tgt_lang = cfg.translation.nllb_tgt_lang
        try:
            forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
        except Exception:
            try:
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
            except Exception:
                logger.warning("NLLB: could not resolve target lang '%s'", tgt_lang)
                forced_bos_token_id = None

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            num_batches = math.ceil(len(texts) / batch_size)

            if notify_fn:
                notify_fn(
                    "Translating",
                    f"NLLB-200: lote {batch_num}/{num_batches} ({len(batch)} segs)…",
                )

            try:
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)

                generate_kwargs: dict = dict(
                    **inputs,
                    num_beams=4,
                    max_length=512,
                    early_stopping=True,
                )
                if forced_bos_token_id is not None:
                    generate_kwargs["forced_bos_token_id"] = forced_bos_token_id

                with torch.no_grad():
                    translated_ids = nllb_model.generate(**generate_kwargs)

                batch_results = tokenizer.batch_decode(
                    translated_ids,
                    skip_special_tokens=True,
                )
                results.extend(batch_results)

            except Exception as e:
                logger.error("NLLB batch %d failed: %s — keeping original", batch_num, e)
                results.extend(batch)

        return results

    # ── Backend: GGUF (LLM via llama-cpp-python) ─────────────────────────────

    def _translate_gguf(
        self,
        segments: List[TimedSegment],
        notify_fn: Optional[Callable],
        total: int,
    ) -> List[str]:
        """Translate using a Qwen2.5 GGUF model as a structured JSON task.

        This is the most context-aware backend but also the slowest.
        The LLM is instructed to return a strict JSON array matching the
        input batch size, making it easy to map results back to segments.
        """
        from src.core.model_manager import models

        llm = self._llm or models.get_gguf()
        batch_size = cfg.translation.batch_size

        system_prompt = (
            "Eres un traductor experto de doblaje (Inglés → Español neutro).\n"
            "Traduce los textos del JSON manteniendo la misma estructura.\n"
            "REGLAS CRÍTICAS:\n"
            "1. Devuelve EXCLUSIVAMENTE el arreglo JSON.\n"
            "2. No incluyas explicaciones ni bloques de código markdown.\n"
            "3. Mantén nombres propios e intenta que la longitud sea similar al original.\n"
            "4. Preserva los IDs originales."
        )

        results: List[str] = []

        for batch_start in range(0, len(segments), batch_size):
            batch = segments[batch_start : batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            num_batches = math.ceil(len(segments) / batch_size)

            if notify_fn:
                notify_fn(
                    "Translating",
                    f"GGUF: lote {batch_num}/{num_batches} ({len(batch)} segs)…",
                )

            batch_json = [
                {"id": i, "text": s.text, "duration": round(s.end - s.start, 2)}
                for i, s in enumerate(batch)
            ]
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\nTraduce este JSON:\n"
                f"{json.dumps(batch_json, ensure_ascii=False)}<|im_end|>\n"
                f"<|im_start|>assistant\n["
            )

            raw = ""
            try:
                res  = llm(prompt, max_tokens=1024, temperature=0.0, stop=["<|im_end|>"])
                raw  = res["choices"][0]["text"]
                if not raw.strip().startswith("["):
                    raw = "[" + raw
                end_pos = raw.rfind("]")
                json_str = raw[: end_pos + 1] if end_pos != -1 else raw
                parsed = json.loads(json_str)
                if len(parsed) != len(batch):
                    raise ValueError(f"Expected {len(batch)} items, got {len(parsed)}")
                results.extend(item.get("text", batch[i].text) for i, item in enumerate(parsed))

            except Exception as e:
                logger.error("GGUF batch %d failed: %s", batch_num, e)
                if raw:
                    logger.debug("Raw LLM output: %s", raw[:300])
                results.extend(s.text for s in batch)

        return results

    # ── Backend: legacy Argos ─────────────────────────────────────────────────

    def _translate_argos(
        self,
        texts: List[str],
        notify_fn: Optional[Callable],
        total: int,
    ) -> List[str]:
        """Translate with argostranslate (legacy simple-mode backend)."""
        from src.core.model_manager import models
        models.ensure_argos()

        import argostranslate.translate

        installed = argostranslate.translate.get_installed_languages()
        en_lang   = next((l for l in installed if l.code == "en"), None)
        es_lang   = next((l for l in installed if l.code == "es"), None)

        if en_lang is None or es_lang is None:
            logger.error("argostranslate en/es pair not available — returning source texts")
            return texts

        translator = en_lang.get_translation(es_lang)
        results: List[str] = []

        for i, text in enumerate(texts):
            if notify_fn and i % 5 == 0:
                notify_fn("Translating", f"Argos: seg {i+1}/{total}…")
            try:
                results.append(translator.translate(text))
            except Exception as e:
                logger.warning("Argos failed on segment %d: %s", i, e)
                results.append(text)

        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _merge_segments(self, segments: List[TimedSegment]) -> List[TimedSegment]:
        """Merge consecutive same-speaker segments within gap/duration limits.

        This gives the translation more context and reduces per-segment overhead.
        Settings are read from ``cfg.merging``.
        """
        if not segments:
            return []
        max_gap = cfg.merging.merge_gap_s
        max_dur = cfg.merging.merge_max_dur_s

        merged: List[TimedSegment] = []
        curr = copy.deepcopy(segments[0])

        for nxt in segments[1:]:
            gap = nxt.start - curr.end
            dur = nxt.end  - curr.start
            same_spk = nxt.speaker_id == curr.speaker_id
            if same_spk and gap < max_gap and dur < max_dur:
                curr.text += " " + nxt.text
                curr.end   = nxt.end
            else:
                merged.append(curr)
                curr = copy.deepcopy(nxt)
        merged.append(curr)
        return merged

    # ── Text post-processing ──────────────────────────────────────────────────

    @staticmethod
    def _clean_translation(text: str) -> str:
        """Light cleanup of model output: collapse whitespace, fix punctuation."""
        text = re.sub(r"\s+", " ", text).strip()
        # Remove leftover markdown artefacts (e.g. ** or __)
        text = re.sub(r"[*_`]{2,}", "", text)
        return text
