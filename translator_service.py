"""
translator_service.py
=====================
Backend pipeline for English → Spanish audio translation with voice cloning.

Pipeline overview
-----------------
1. Extract / normalise audio → WAV + capture total_duration
2. Speaker diarisation       → List[SpeakerSegment]  (who speaks when)
3. Transcription             → per-segment text with timestamps, speaker joined from step 2
4. Translation               → per-segment Spanish text  (timing & speaker preserved)
                               — Uses a local GGUF LLM (llama-cpp-python) with
                                 batched, JSON-structured contextual translation.
5. TTS + time-stretch        → each segment synthesised then stretched to its original window
                               — Silence-trimmed before atempo; desbordamiento inteligente.
6. Assembly                  → segments overlaid on a silent timeline of exactly total_duration
7. Export                    → write output file

Improvement Notes
-----------------
- Model Singleton: STT and LLM models are loaded once and reused across calls.
- STT Overlap: Audio chunks overlap by CHUNK_OVERLAP_S seconds to avoid word cuts at boundaries.
- Batched Translation: LLM receives N segments as a JSON block and returns N translations in one call.
- Silence Trimming: Leading/trailing silence from edge-tts is trimmed before time-stretching.
- Smart Overflow: If the dubbed clip is only slightly longer, we allow it to bleed into adjacent silence.
- Diarization Scaffold: Ready to plug pyannote.audio for real multi-speaker detection.
"""

from __future__ import annotations

import asyncio
import json
import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

try:
    from moviepy.editor import AudioFileClip
except ImportError:
    from moviepy import AudioFileClip


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpeakerSegment:
    """One continuous time-window attributed to a single speaker."""
    speaker_id: str
    start_time: float   # seconds
    end_time: float     # seconds


@dataclass
class TimedSegment:
    """A transcribed (and later translated) segment with full timing metadata."""
    speaker_id: str
    start: float        # seconds
    end: float          # seconds
    text: str           # original or translated text


@dataclass
class TranscriptionResult:
    """Transcription output: full text + per-segment detail."""
    full_text: str
    segments: List[TimedSegment] = field(default_factory=list)


ProgressCallback = Callable[[str, str], None]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE_S       = 600   # 10-minute primary chunks for STT
CHUNK_OVERLAP_S    = 10    # Overlap to avoid clipping words at chunk boundary
MERGE_GAP_S        = 1.5   # Max silence gap to merge consecutive same-speaker segments
MERGE_MAX_DUR_S    = 15.0  # Max duration of a merged segment
ATEMPO_MAX_RATIO   = 1.20  # Maximum allowed speed-up ratio before we prefer overflow
SILENCE_OVERFLOW_S  = 2.0  # If next segment is ≥ this many seconds away, allow TTS to overflow
CHARS_PER_SEC      = 15    # Approx characters per second for isochrony hints
TTS_BATCH_SIZE     = 15    # How many segments to send to LLM per batch

# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class TranslatorService:
    """Orchestrates the English → Spanish audio translation pipeline with persistence."""

    AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".wma", ".m4a"}

    # Voice mapping per speaker (add more speakers here as needed)
    _SPEAKER_VOICE: Dict[str, str] = {
        "speaker_1": "es-ES-AlvaroNeural",
        "speaker_2": "es-ES-ElviraNeural",
        "speaker_3": "es-MX-JorgeNeural",
        "speaker_4": "es-MX-DaliaNeural",
    }
    _DEFAULT_VOICE = "es-ES-AlvaroNeural"

    def __init__(self) -> None:
        self._temp_dir: Optional[str] = None

        # --- Singleton holders ---
        # STT
        self._whisper_model = None
        # LLM (llama-cpp GGUF preferred; fallback to transformers)
        self._llm = None
        self._llm_tokenizer = None
        self._llm_backend: str = "none"   # "gguf" | "transformers" | "none"

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_cache_dir(self, input_path: str) -> str:
        """Create a persistent cache dir based on input file path."""
        path_hash = hashlib.md5(input_path.encode()).hexdigest()[:12]
        base_name = os.path.splitext(os.path.basename(input_path))[0][:20]
        cache_dir = os.path.join(os.getcwd(), "dubbing_cache", f"{base_name}_{path_hash}")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    # ------------------------------------------------------------------
    # Model singleton loaders
    # ------------------------------------------------------------------

    def _load_whisper(self, notify_fn: Optional[Callable] = None):
        """Load faster-whisper model once; reuse on subsequent calls."""
        if self._whisper_model is not None:
            return self._whisper_model
        from faster_whisper import WhisperModel
        if notify_fn:
            notify_fn("Transcribing", "Cargando modelo STT (small, int8)…")
        # 'small' model: better accuracy than base, still fast on CPU/GPU with int8
        self._whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        return self._whisper_model

    def _load_llm(self, notify_fn: Optional[Callable] = None):
        """
        Load translation LLM once.
        Priority:
        1. llama-cpp-python with a GGUF Q4_K_M model (fastest, ~1.5 GB RAM)
        2. Transformers AutoModelForCausalLM (fallback, ~3-6 GB RAM, slower)
        """
        if self._llm is not None:
            return

        # --- Attempt 1: GGUF via llama-cpp-python ---
        # Tries to load Qwen2.5-1.5B-Instruct in Q4_K_M format from HuggingFace Hub.
        # The GGUF file is cached automatically by llama-cpp-python / huggingface_hub.
        try:
            from llama_cpp import Llama
            if notify_fn:
                notify_fn("Translating", "Cargando LLM GGUF (Qwen2.5-1.5B Q4_K_M)…")
            self._llm = Llama.from_pretrained(
                repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                filename="*q4_k_m*",
                n_ctx=4096,   # Context window (fits ~15 segments comfortably)
                n_threads=os.cpu_count() or 4,
                verbose=False,
            )
            self._llm_backend = "gguf"
            if notify_fn:
                notify_fn("Translating", "✅ LLM GGUF listo (modo rápido)")
            return
        except Exception as e:
            if notify_fn:
                notify_fn("Translating", f"⚠ GGUF no disponible ({e}), usando transformers…")

        # --- Attempt 2: Transformers fallback ---
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            if notify_fn:
                notify_fn("Translating", f"Cargando LLM transformers ({model_name})…")
            self._llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            if torch.cuda.is_available():
                self._llm = self._llm.cuda()
            self._llm_backend = "transformers"
            if notify_fn:
                notify_fn("Translating", "✅ LLM transformers listo (modo CPU/GPU)")
        except Exception as e:
            raise RuntimeError(f"No se pudo cargar ningún backend LLM: {e}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[ProgressCallback] = None,
        max_threads: int = 10,
    ) -> str:
        """Run the full translation pipeline with resumption support."""

        def _notify(stage: str, msg: str) -> None:
            if progress_callback:
                progress_callback(stage, msg)

        self._temp_dir = self._get_cache_dir(input_path)
        checkpoint_dir = os.path.join(self._temp_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        try:
            # 1 — Extract / normalise audio
            audio_path = os.path.join(self._temp_dir, "source.mp3")
            if os.path.exists(audio_path):
                _notify("Extracting", "Usando audio cacheado…")
                with AudioFileClip(audio_path) as clip:
                    total_duration = clip.duration
            else:
                _notify("Extracting", "Leyendo archivo de audio (puede tardar)…")
                audio_path, total_duration = self._to_mp3(input_path)
            _notify("Extracting", f"Audio listo ({total_duration:.2f}s)")

            # 2 — Speaker diarisation
            diar_json = os.path.join(checkpoint_dir, "diarization.json")
            if os.path.exists(diar_json):
                _notify("Diarizing", "Cargando hablantes cacheados…")
                with open(diar_json, 'r', encoding='utf-8') as f:
                    diar_segments = [SpeakerSegment(**s) for s in json.load(f)]
            else:
                _notify("Diarizing", "Detectando hablantes…")
                diar_segments = self.perform_diarization(audio_path, _notify)
                with open(diar_json, 'w', encoding='utf-8') as f:
                    json.dump([s.__dict__ for s in diar_segments], f, indent=2)
            _notify("Diarizing", f"{len(diar_segments)} segmento(s) detectados.")

            # 3 — Transcription
            trans_json = os.path.join(checkpoint_dir, "transcription.json")
            if os.path.exists(trans_json):
                _notify("Transcribing", "Cargando transcripción cacheada…")
                with open(trans_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    transcription = TranscriptionResult(
                        full_text=data['full_text'],
                        segments=[TimedSegment(**s) for s in data['segments']]
                    )
            else:
                _notify("Transcribing", "Transcribiendo inglés en chunks solapados…")
                transcription = self.transcribe_audio(audio_path, diar_segments, total_duration, _notify)
                with open(trans_json, 'w', encoding='utf-8') as f:
                    json.dump({
                        'full_text': transcription.full_text,
                        'segments': [s.__dict__ for s in transcription.segments]
                    }, f, indent=2, ensure_ascii=False)
            _notify("Transcribing", f"Transcripción lista ({len(transcription.segments)} segmentos).")

            # 4 — Translation
            trad_json = os.path.join(checkpoint_dir, "translation.json")
            if os.path.exists(trad_json):
                _notify("Translating", "Cargando traducción cacheada…")
                with open(trad_json, 'r', encoding='utf-8') as f:
                    translated_segments = [TimedSegment(**s) for s in json.load(f)]
            else:
                _notify("Translating", "Traduciendo con IA local (procesamiento por lotes)…")
                self._load_llm(_notify)
                translated_segments = self.translate_segments(transcription.segments, notify_fn=_notify)
                with open(trad_json, 'w', encoding='utf-8') as f:
                    json.dump([s.__dict__ for s in translated_segments], f, indent=2, ensure_ascii=False)
            _notify("Translating", f"{len(translated_segments)} segmento(s) traducidos.")

            # 5 — TTS + Assembly
            _notify("Cloning Voice", "Generando y sincronizando voz española…")
            assembled_path = self.clone_voice_and_generate_speech(
                translated_segments, total_duration, _notify, max_threads=max_threads
            )

            # 6 — Save
            _notify("Saving", f"Exportando resultado a {output_path}…")
            self._export(assembled_path, output_path)
            _notify("Completed", f"Done → {output_path}")

            return output_path
        except Exception as e:
            raise

    # ------------------------------------------------------------------
    # Step 2: Diarization
    # ------------------------------------------------------------------

    def perform_diarization(
        self, audio_path: str, notify_fn: Optional[Callable] = None
    ) -> List[SpeakerSegment]:
        """
        Detect *who* speaks *when* using pyannote.audio if available.
        Falls back to a single-speaker placeholder if pyannote is not installed
        or if no HuggingFace token is configured.

        To enable real diarization:
          pip install pyannote.audio
          Set env var HF_TOKEN to your HuggingFace token with access to
          'pyannote/speaker-diarization-3.1'.
        """
        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            if notify_fn:
                notify_fn("Diarizing", "⚠ HF_TOKEN no configurado — usando un solo hablante. "
                           "Para diarización real define la variable de entorno HF_TOKEN.")

        try:
            from pyannote.audio import Pipeline as PyannotePipeline
            if not hf_token:
                raise EnvironmentError("HF_TOKEN requerido para pyannote")

            if notify_fn:
                notify_fn("Diarizing", "Cargando pyannote.audio (puede ser lento la primera vez)…")
            pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
            diarization = pipeline(audio_path)
            segments: List[SpeakerSegment] = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(SpeakerSegment(
                    speaker_id=speaker,
                    start_time=turn.start,
                    end_time=turn.end,
                ))
            return segments if segments else self._fallback_diarization(audio_path)

        except ImportError:
            if notify_fn:
                notify_fn("Diarizing", "pyannote.audio no instalado → fallback a 1 hablante.")
            return self._fallback_diarization(audio_path)
        except Exception as e:
            if notify_fn:
                notify_fn("Diarizing", f"Error en diarización: {e} → fallback a 1 hablante.")
            return self._fallback_diarization(audio_path)

    def _fallback_diarization(self, audio_path: str) -> List[SpeakerSegment]:
        with AudioFileClip(audio_path) as clip:
            duration = clip.duration
        return [SpeakerSegment("speaker_1", 0.0, duration)]

    # ------------------------------------------------------------------
    # Step 3: Transcription (with overlapping chunks)
    # ------------------------------------------------------------------

    def transcribe_audio(
        self,
        audio_path: str,
        diar_segments: List[SpeakerSegment],
        total_duration: float,
        notify_fn: Optional[Callable] = None,
    ) -> TranscriptionResult:
        """
        Transcribe using faster-whisper.
        - Chunks overlap by CHUNK_OVERLAP_S seconds to prevent word cuts.
        - Duplicate segments at chunk boundaries are de-duplicated by timestamp.
        """
        import langdetect

        model = self._load_whisper(notify_fn)

        all_timed: List[TimedSegment] = []
        all_texts: List[str] = []
        conf_reported = False
        seen_ranges: set = set()   # (round(start,1), round(end,1)) de-dup key

        num_chunks = math.ceil(total_duration / CHUNK_SIZE_S)

        for i in range(num_chunks):
            start_off = i * CHUNK_SIZE_S
            # Include overlap on the right edge (except for the last chunk)
            dur = min(CHUNK_SIZE_S + CHUNK_OVERLAP_S, total_duration - start_off)

            if notify_fn:
                notify_fn("Transcribing",
                           f"Chunk {i+1}/{num_chunks} ({int(start_off)}s – {int(start_off+dur)}s)")

            chunk_mp3 = os.path.join(self._temp_dir, f"chunk_{i}.mp3")
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(start_off), "-t", str(dur),
                "-i", audio_path,
                "-acodec", "libmp3lame", "-ab", "64k",
                chunk_mp3
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            c_segments_gen, _ = model.transcribe(
                chunk_mp3,
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=100),
                condition_on_previous_text=False,
                initial_prompt=(
                    "This is a highly accurate English transcription. "
                    "Text is clear, grammatical, and nicely punctuated."
                ),
            )

            for seg in c_segments_gen:
                seg_text = seg.text.strip()
                if not seg_text:
                    continue

                abs_start = round(seg.start + start_off, 2)
                abs_end   = round(seg.end   + start_off, 2)

                # De-duplicate: skip if this exact (start, end) was already captured
                # by the previous chunk's overlap region.
                dedup_key = (round(abs_start, 1), round(abs_end, 1))
                if dedup_key in seen_ranges:
                    continue
                seen_ranges.add(dedup_key)

                # Basic text cleanup
                seg_text = re.sub(r'([?!])\1+', r'\1', seg_text)
                if seg_text and seg_text[0].islower():
                    seg_text = seg_text[0].upper() + seg_text[1:]

                speaker_id = self._assign_speaker(abs_start, abs_end, diar_segments)
                all_timed.append(TimedSegment(
                    speaker_id=speaker_id,
                    start=abs_start,
                    end=abs_end,
                    text=seg_text,
                ))
                all_texts.append(seg_text)

            # Language confirmation on first valid chunk
            if not conf_reported and len(all_texts) > 0:
                try:
                    lang = langdetect.detect(" ".join(all_texts[:5]))
                    if notify_fn:
                        notify_fn("Transcribing", f"✅ Idioma detectado: '{lang}'")
                except Exception:
                    pass
                conf_reported = True

            try:
                os.remove(chunk_mp3)
            except Exception:
                pass

        # Sort by start time (overlapping chunks may insert out of order)
        all_timed.sort(key=lambda s: s.start)

        return TranscriptionResult(full_text=" ".join(all_texts), segments=all_timed)

    # ------------------------------------------------------------------
    # Step 4: Translation (Batched contextual JSON via LLM)
    # ------------------------------------------------------------------

    def translate_segments(
        self, segments: List[TimedSegment], notify_fn: Optional[Callable] = None
    ) -> List[TimedSegment]:
        """
        Translate English segments to Spanish using a local LLM.

        Key improvements over segment-by-segment approach:
        - Segments are first merged for TTS naturalness.
        - LLM receives TTS_BATCH_SIZE segments as a JSON array in each call.
        - LLM is asked to return a JSON array with the same IDs and translated text.
        - This reduces LLM calls by ~15x vs one-at-a-time and gives broader context.
        """

        # --- Merge short segments for TTS naturalness ---
        merged_segments = self._merge_segments(segments)
        total = len(merged_segments)
        translated: List[TimedSegment] = []

        system_prompt = (
            "Eres un traductor experto de doblaje (Inglés → Español neutro).\n"
            "Recibirás un JSON con N objetos, cada uno con: id, text (inglés), duration_s, target_chars.\n"
            "Debes devolver EXCLUSIVAMENTE un JSON válido con la misma estructura pero con el campo "
            "'text' traducido al español. Reglas estrictas:\n"
            "1. CERO EXPLICACIONES: Devuelve solo el JSON, sin texto adicional fuera de él.\n"
            "2. NOMBRES PROPIOS Y FICCIÓN: Mantén nombres y títulos originales salvo traducción oficial canónica.\n"
            "3. ISOCRONÍA: Cada 'text' traducido debe tener aproximadamente 'target_chars' caracteres.\n"
            "4. NATURALIDAD TTS: No uses abreviaturas, corchetes ni texto entre paréntesis.\n"
            "5. COMPLETITUD: Si una frase parece incompleta, adáptala usando el contexto del batch."
        )

        batch_idx = 0
        for batch_start in range(0, total, TTS_BATCH_SIZE):
            batch_end = min(batch_start + TTS_BATCH_SIZE, total)
            batch = merged_segments[batch_start:batch_end]
            batch_idx += 1

            if notify_fn:
                notify_fn("Translating",
                           f"Lote {batch_idx}/{math.ceil(total/TTS_BATCH_SIZE)} "
                           f"(segs {batch_start+1}–{batch_end}/{total})")

            # Build the JSON payload for this batch
            payload = []
            for j, seg in enumerate(batch):
                dur = seg.end - seg.start
                payload.append({
                    "id": j,
                    "text": seg.text.strip(),
                    "duration_s": round(dur, 2),
                    "target_chars": max(10, int(dur * CHARS_PER_SEC)),
                })

            user_prompt = json.dumps(payload, ensure_ascii=False)

            raw_response = self._llm_infer(system_prompt, user_prompt)
            batch_translations = self._parse_batch_response(raw_response, batch)

            for j, seg in enumerate(batch):
                es_text = batch_translations.get(j, seg.text)  # fallback: keep original
                es_text = self._postprocess_translation(es_text)
                translated.append(TimedSegment(
                    speaker_id=seg.speaker_id,
                    start=seg.start,
                    end=seg.end,
                    text=es_text,
                ))

        return translated

    def _merge_segments(self, segments: List[TimedSegment]) -> List[TimedSegment]:
        """Merge short same-speaker consecutive segments with small gaps."""
        if not segments:
            return []
        merged: List[TimedSegment] = []
        current = TimedSegment(
            speaker_id=segments[0].speaker_id,
            start=segments[0].start,
            end=segments[0].end,
            text=segments[0].text,
        )
        for i in range(1, len(segments)):
            nxt = segments[i]
            gap = nxt.start - current.end
            ends_with_punct = current.text.strip().endswith(('.', '?', '!', '"', '"'))
            duration = current.end - current.start
            if (
                nxt.speaker_id == current.speaker_id
                and gap < MERGE_GAP_S
                and not ends_with_punct
                and duration < MERGE_MAX_DUR_S
            ):
                current.end = nxt.end
                current.text = f"{current.text.strip()} {nxt.text.strip()}"
            else:
                merged.append(current)
                current = TimedSegment(nxt.speaker_id, nxt.start, nxt.end, nxt.text)
        merged.append(current)
        return merged

    def _llm_infer(self, system_prompt: str, user_prompt: str) -> str:
        """Run inference via GGUF (llama-cpp) or transformers backend."""
        if self._llm_backend == "gguf":
            response = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=1024,
                temperature=0.15,
            )
            return response["choices"][0]["message"]["content"].strip()

        elif self._llm_backend == "transformers":
            import torch
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ]
            text_input = self._llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._llm_tokenizer([text_input], return_tensors="pt")
            if next(self._llm.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                generated_ids = self._llm.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.15,
                    do_sample=True,
                    pad_token_id=self._llm_tokenizer.eos_token_id,
                )
            gen_only = [
                out[len(inp):]
                for inp, out in zip(inputs["input_ids"], generated_ids)
            ]
            return self._llm_tokenizer.batch_decode(gen_only, skip_special_tokens=True)[0].strip()

        raise RuntimeError("LLM no cargado. Llama _load_llm() primero.")

    def _parse_batch_response(
        self, raw: str, batch: List[TimedSegment]
    ) -> Dict[int, str]:
        """
        Parse the LLM's JSON response. Returns {id: translated_text}.
        Handles common failure modes gracefully.
        """
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        try:
            items = json.loads(raw)
            if isinstance(items, list):
                return {item["id"]: item["text"] for item in items if "id" in item and "text" in item}
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # Fallback: try to extract individual JSON objects with regex
        result: Dict[int, str] = {}
        for match in re.finditer(r'\{"id"\s*:\s*(\d+)\s*,.*?"text"\s*:\s*"(.*?)"', raw, re.DOTALL):
            idx = int(match.group(1))
            text = match.group(2).replace('\\"', '"')
            result[idx] = text

        if result:
            return result

        # Last resort: split by newline and assign in order
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        return {i: lines[i] for i in range(min(len(lines), len(batch)))}

    @staticmethod
    def _postprocess_translation(text: str) -> str:
        """Remove LLM artifacts from translated text."""
        # Remove conversational intros
        text = re.sub(
            r'(?i)^(aquí tienes.*?traducción.*?:\s*|traducción:\s*|segmento:\s*|texto traducido:\s*)',
            '', text
        ).strip()
        # Remove stage directions / notes
        text = re.sub(r'\[.*?\]|<.*?>|\(.*?\)', '', text).strip()
        # Remove leading/trailing markdown or quote artifacts
        text = re.sub(r'^["\'`\-]+|["\'`\-]+$', '', text)
        text = text.replace('**', '').replace('*', '').strip()
        return text

    # ------------------------------------------------------------------
    # Step 5: TTS + Assembly
    # ------------------------------------------------------------------

    def clone_voice_and_generate_speech(
        self,
        segments: List[TimedSegment],
        total_duration: float,
        notify_fn: Optional[Callable] = None,
        max_threads: int = 10,
    ) -> str:
        """
        Generate time-synchronised Spanish audio with:
        - Silence trimming before atempo adjustment.
        - Smart overflow: allow TTS to bleed into following silence.
        - Parallel execution with per-segment caching.
        """
        from pydub import AudioSegment, silence as pydub_silence
        import concurrent.futures
        import audioop

        tts_cache_dir = os.path.join(self._temp_dir, "tts_cache")
        os.makedirs(tts_cache_dir, exist_ok=True)

        sample_rate  = 16000
        sample_width = 2
        channels     = 1
        bytes_per_ms = int(sample_rate * channels * sample_width / 1000)

        target_total_bytes = int(total_duration * 1000) * bytes_per_ms
        final_audio_bytes  = bytearray(target_total_bytes)

        sorted_segments = sorted(segments, key=lambda s: s.start)
        total_count     = len(sorted_segments)

        # Build a next-segment start lookup so we know how much gap follows each seg
        next_starts: Dict[int, float] = {}
        for i in range(len(sorted_segments) - 1):
            next_starts[i] = sorted_segments[i + 1].start
        next_starts[len(sorted_segments) - 1] = total_duration

        def _process_segment(item: Tuple[int, TimedSegment]):
            idx, seg = item
            if not seg.text.strip():
                return idx, seg, None, 0

            original_ms = int((seg.end - seg.start) * 1000)
            if original_ms <= 0:
                return idx, seg, None, 0

            # How much room do we have before the next speaker starts?
            next_start    = next_starts.get(idx, total_duration)
            available_ms  = int((next_start - seg.start) * 1000)

            seg_sig     = f"{seg.text}_{seg.speaker_id}_{original_ms}"
            seg_hash    = hashlib.md5(seg_sig.encode()).hexdigest()[:10]
            stretched_wav = os.path.join(tts_cache_dir, f"fin_{seg_hash}.wav")

            if os.path.exists(stretched_wav):
                return idx, seg, AudioSegment.from_wav(stretched_wav), available_ms

            # --- 5a: TTS (edge-tts) ---
            tts_mp3 = os.path.join(tts_cache_dir, f"tts_{idx}.mp3")
            self._run_tts(seg.text, seg.speaker_id, tts_mp3)

            tts_audio = AudioSegment.from_file(tts_mp3, format="mp3")

            if len(tts_audio) == 0:
                try: os.remove(tts_mp3)
                except: pass
                return idx, seg, None, 0

            # --- 5b: Trim leading/trailing silence generated by edge-tts ---
            tts_audio = self._trim_silence(tts_audio, pydub_silence)
            if len(tts_audio) == 0:
                return idx, seg, None, 0

            tts_ms    = len(tts_audio)
            raw_ratio = tts_ms / original_ms

            # --- 5c: Smart speed adjustment ---
            if raw_ratio <= 1.0:
                # TTS is shorter → no stretching needed, natural pace preserved
                tts_audio.export(stretched_wav, format="wav")
                stretched_audio = tts_audio
            elif tts_ms <= available_ms:
                # TTS is longer than original window but still fits in the silence gap
                # → Allow overflow without speed change (preserves prosody)
                tts_audio.export(stretched_wav, format="wav")
                stretched_audio = tts_audio
            else:
                # Must speed up; cap at ATEMPO_MAX_RATIO to avoid robotic sound
                speed_ratio = min(ATEMPO_MAX_RATIO, tts_ms / available_ms)
                self._ffmpeg_atempo(tts_mp3, stretched_wav, speed_ratio)
                stretched_audio = AudioSegment.from_wav(stretched_wav)

            try:
                os.remove(tts_mp3)
            except: pass

            return idx, seg, stretched_audio, available_ms

        processed_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            results = executor.map(_process_segment, enumerate(sorted_segments))

            for idx, seg, stretched_audio, avail_ms in results:
                processed_count += 1
                if notify_fn and (processed_count % 5 == 0 or processed_count == total_count):
                    notify_fn("Cloning Voice",
                               f"Segmento {processed_count}/{total_count} ({max_threads} hilos)")

                if stretched_audio is None:
                    continue

                stretched_audio = (
                    stretched_audio
                    .set_frame_rate(sample_rate)
                    .set_channels(channels)
                    .set_sample_width(sample_width)
                )

                seg_data  = stretched_audio.raw_data
                start_byte = int(seg.start * 1000) * bytes_per_ms
                start_byte -= start_byte % (sample_width * channels)

                copy_len = min(len(seg_data), target_total_bytes - start_byte)
                if copy_len > 0:
                    current_slice = final_audio_bytes[start_byte: start_byte + copy_len]
                    try:
                        mixed = audioop.add(current_slice, seg_data[:copy_len], sample_width)
                        final_audio_bytes[start_byte: start_byte + copy_len] = mixed
                    except Exception:
                        final_audio_bytes[start_byte: start_byte + copy_len] = seg_data[:copy_len]

        baseline = AudioSegment(
            data=bytes(final_audio_bytes),
            sample_width=sample_width,
            frame_rate=sample_rate,
            channels=channels,
        )

        assembled_path = os.path.join(self._temp_dir, "assembled_es.mp3")
        baseline.export(assembled_path, format="mp3", bitrate="128k")
        return assembled_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trim_silence(
        audio,
        pydub_silence_mod,
        silence_thresh_db: int = -40,
        min_silence_ms:    int = 100,
    ):
        """
        Remove leading and trailing silence from a pydub AudioSegment.
        edge-tts often pads 50-200ms of silence at the edges.
        """
        try:
            chunks = pydub_silence_mod.detect_nonsilent(
                audio,
                min_silence_len=min_silence_ms,
                silence_thresh=silence_thresh_db,
            )
            if not chunks:
                return audio
            # Keep only the span from first non-silent to last non-silent
            start_trim = chunks[0][0]
            end_trim   = chunks[-1][1]
            return audio[start_trim:end_trim]
        except Exception:
            return audio

    @staticmethod
    def _assign_speaker(
        start: float, end: float, diar: List[SpeakerSegment]
    ) -> str:
        """Return the speaker_id whose window overlaps [start, end] the most."""
        best_speaker = "speaker_1"
        best_overlap = -1.0
        for d in diar:
            overlap = min(end, d.end_time) - max(start, d.start_time)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d.speaker_id
        return best_speaker

    def _run_tts(self, text: str, speaker_id: str, out_mp3: str) -> None:
        """Synthesise *text* with edge-tts and save to *out_mp3*."""
        import edge_tts
        voice = self._SPEAKER_VOICE.get(speaker_id, self._DEFAULT_VOICE)

        async def _synth():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(out_mp3)

        asyncio.run(_synth())

    @staticmethod
    def _ffmpeg_atempo(src: str, dst: str, speed_ratio: float) -> None:
        """
        Stretch/compress *src* by *speed_ratio* using the atempo filter.
        atempo is constrained to [0.5, 2.0]; chains multiple instances for
        values outside this range.
        """
        ratio = max(0.1, min(speed_ratio, 10.0))
        filters: List[str] = []
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

    def _to_mp3(self, input_path: str) -> Tuple[str, float]:
        """Convert any supported audio/video to 128k MP3; returns (audio_path, duration_s)."""
        mp3_out = os.path.join(self._temp_dir, "source.mp3")
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

    def _export(self, audio_path: str, output_path: str) -> None:
        """Export the assembled audio to the desired output format."""
        clip = AudioFileClip(audio_path)
        ext  = os.path.splitext(output_path)[1].lower()
        codec_map = {
            ".mp3": "libmp3lame",
            ".ogg": "libvorbis",
            ".aac": "aac",
            ".m4a": "aac",
        }
        codec = codec_map.get(ext)
        if codec:
            clip.write_audiofile(output_path, codec=codec, bitrate="128k", logger=None)
        else:
            clip.write_audiofile(output_path, logger=None)
        clip.close()

    def _cleanup(self) -> None:
        if self._temp_dir and os.path.isdir(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    # ------------------------------------------------------------------
    # Legacy / compatibility shim
    # ------------------------------------------------------------------

    def translate_text(self, text: str) -> str:
        """Translate a single English string to Spanish (compatibility shim)."""
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_name = "Helsinki-NLP/opus-mt-en-es"
        tokenizer  = AutoTokenizer.from_pretrained(model_name)
        model      = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        inputs     = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        translated_tokens = model.generate(**inputs)
        return tokenizer.decode(translated_tokens[0], skip_special_tokens=True) if text else ""
