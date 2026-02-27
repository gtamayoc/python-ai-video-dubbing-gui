import os
import json
import math
import logging
import re
import copy
from typing import List, Optional, Callable
from src.core.entities import TimedSegment
from src.core.config import MERGE_GAP_S, MERGE_MAX_DUR_S, TTS_BATCH_SIZE

logger = logging.getLogger(__name__)

class TranslationService:
    def __init__(self):
        self._llm = None
        self._llm_tokenizer = None
        self._llm_backend = "none"

    def load_model(self, notify_fn: Optional[Callable] = None):
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama
            logger.info("Attempting to load LLM via llama-cpp-python (GGUF)...")
            if notify_fn:
                notify_fn("Translating", "Cargando LLM GGUF (Qwen2.5-1.5B Q4_K_M)…")
            
            system_cores = os.cpu_count() or 4
            safe_n_threads = max(1, system_cores - 2)
            
            logger.info(f"LLM Config: n_ctx=4096, n_threads={safe_n_threads}, n_gpu_layers=-1")
            self._llm = Llama.from_pretrained(
                repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                filename="*q4_k_m*",
                n_ctx=4096,
                n_threads=safe_n_threads,
                n_gpu_layers=-1, # Try GPU auto-offload first
                verbose=False,
            )
            self._llm_backend = "gguf"
            logger.info("LLM GGUF loaded successfully.")
            return
        except Exception as e:
            logger.warning(f"Failed to load LLM via llama-cpp (GGUF): {e}")
            pass

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            self._llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self._llm_backend = "transformers"
        except Exception as e:
            raise RuntimeError(f"No se pudo cargar ningún backend LLM: {e}")

    def translate_segments(
        self, segments: List[TimedSegment], notify_fn: Optional[Callable] = None
    ) -> List[TimedSegment]:
        if not segments:
            return []
            
        merged_segments = self._merge_segments(segments)
        total = len(merged_segments)
        translated: List[TimedSegment] = []

        system_prompt = (
            "Eres un traductor experto de doblaje (Inglés → Español neutro).\n"
            "Traduce los textos del JSON manteniendo la misma estructura.\n"
            "REGLAS CRÍTICAS:\n"
            "1. Devuelve EXCLUSIVAMENTE el arreglo JSON.\n"
            "2. No incluyas explicaciones ni bloques de código markdown.\n"
            "3. Mantén nombres propios e intenta que la longitud sea similar al original (isocronía).\n"
            "4. Preserva los IDs originales."
        )

        batch_idx = 0
        for batch_start in range(0, total, TTS_BATCH_SIZE):
            batch_end = min(batch_start + TTS_BATCH_SIZE, total)
            batch = merged_segments[batch_start:batch_end]
            batch_idx += 1

            if notify_fn:
                notify_fn("Translating", f"Procesando lote {batch_idx}/{math.ceil(total/TTS_BATCH_SIZE)}")

            batch_json = []
            for i, seg in enumerate(batch):
                batch_json.append({
                    "id": i,
                    "text": seg.text,
                    "duration": round(seg.end - seg.start, 2)
                })

            # Qwen ChatML formatting
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\nTraduce este JSON:\n{json.dumps(batch_json, ensure_ascii=False)}<|im_end|>\n"
                f"<|im_start|>assistant\n["
            )
            
            raw_reply = ""
            try:
                raw_reply = self._query_llm(prompt)
                # Re-add the '[' we pre-filled to ensure it's a valid list or handle it
                if not raw_reply.strip().startswith("["):
                    raw_reply = "[" + raw_reply
                
                # Robust extraction: find the last ']'
                end_pos = raw_reply.rfind(']')
                if end_pos != -1:
                    json_str = raw_reply[:end_pos+1]
                else:
                    json_str = raw_reply

                results = json.loads(json_str)
                
                # Verify we got the same amount of items
                if len(results) != len(batch):
                    logger.warning(f"Batch {batch_idx}: Recibidos {len(results)} items, esperados {len(batch)}. Usando fallback.")
                    raise ValueError("Mismatch in result count")

                for i, res in enumerate(results):
                    translated.append(TimedSegment(
                        speaker_id=batch[i].speaker_id,
                        start=batch[i].start,
                        end=batch[i].end,
                        text=res.get("text", batch[i].text)
                    ))
            except Exception as e:
                logger.error(f"Fallo en lote {batch_idx}: {e}")
                if raw_reply:
                    logger.debug(f"Respuesta cruda del LLM: {raw_reply}")
                # Fallback to original text for this batch
                for seg in batch:
                    translated.append(seg)

        return translated

    def _query_llm(self, prompt: str) -> str:
        # Settings for low-latency/deterministic output
        gen_params = {
            "max_tokens": 1024,
            "temperature": 0.0, # Greedy for structural tasks
            "stop": ["<|im_end|>", "<|endoftext|>"]
        }

        if self._llm_backend == "gguf":
            res = self._llm(prompt, **gen_params)
            return res["choices"][0]["text"]
        else:
            inputs = self._llm_tokenizer(prompt, return_tensors="pt")
            if self._llm.device.type == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = self._llm.generate(
                **inputs, 
                max_new_tokens=gen_params["max_tokens"], 
                temperature=gen_params["temperature"],
                do_sample=False,
                pad_token_id=self._llm_tokenizer.eos_token_id
            )
            # We only want the new part after the prompt
            input_len = inputs["input_ids"].shape[1]
            return self._llm_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    def _merge_segments(self, segments: List[TimedSegment]) -> List[TimedSegment]:
        if not segments: return []
        merged = []
        # Create a deep copy for merging to avoid modifying inputs
        curr = copy.deepcopy(segments[0])
        
        for next_seg in segments[1:]:
            gap = next_seg.start - curr.end
            dur = next_seg.end - curr.start
            if next_seg.speaker_id == curr.speaker_id and gap < MERGE_GAP_S and dur < MERGE_MAX_DUR_S:
                curr.text += " " + next_seg.text
                curr.end = next_seg.end
            else:
                merged.append(curr)
                curr = copy.deepcopy(next_seg)
        merged.append(curr)
        return merged
