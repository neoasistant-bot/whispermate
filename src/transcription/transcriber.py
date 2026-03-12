"""
Wrapper de faster-whisper para transcripción local.
Lazy-load del modelo al primer uso.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TranscriptionResult:
    text: str
    segments: list  # [{"start": float, "end": float, "text": str}]
    language: str
    duration_ms: int


VALID_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]


class Transcriber:
    def __init__(self, model_name: str = "small"):
        if model_name not in VALID_MODELS:
            print(f"[Transcriber] Modelo '{model_name}' no reconocido. Usando 'small'.")
            model_name = "small"
        self._model_name = model_name
        self._model = None
        self._lock = threading.Lock()
        self._loading = False

    def _load_model(self) -> bool:
        """Carga el modelo faster-whisper. Retorna True si exitoso."""
        if self._model is not None:
            return True

        try:
            from faster_whisper import WhisperModel
            print(f"[Transcriber] Cargando modelo '{self._model_name}'...")
            print("[Transcriber] (Primera vez puede tardar mientras descarga el modelo)")

            # Intentar GPU, fallback a CPU
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
            except ImportError:
                device = "cpu"
                compute_type = "int8"

            self._model = WhisperModel(
                self._model_name,
                device=device,
                compute_type=compute_type,
            )
            print(f"[Transcriber] Modelo '{self._model_name}' cargado en {device}.")
            return True

        except Exception as e:
            print(f"[Transcriber] Error cargando modelo: {e}")
            self._model = None
            return False

    def preload(self) -> bool:
        """Precarga el modelo explícitamente (útil para evitar delay en primer uso)."""
        with self._lock:
            return self._load_model()

    def change_model(self, model_name: str) -> None:
        """Cambia el modelo. Se recargará en la próxima transcripción."""
        if model_name not in VALID_MODELS:
            print(f"[Transcriber] Modelo inválido: {model_name}")
            return
        with self._lock:
            if self._model_name != model_name:
                print(f"[Transcriber] Cambiando modelo a '{model_name}'.")
                self._model_name = model_name
                self._model = None  # forzar reload

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
    ) -> Optional[TranscriptionResult]:
        """
        Transcribe un array de audio (float32, 16kHz, mono).
        Retorna TranscriptionResult o None si hay error.
        """
        if audio is None or len(audio) == 0:
            return None

        with self._lock:
            if self._model is None:
                success = self._load_model()
                if not success:
                    return None

            start_time = time.time()

            try:
                # faster-whisper acepta numpy array directamente
                segments_iter, info = self._model.transcribe(
                    audio,
                    language=language if language and language != "auto" else None,
                    beam_size=5,
                    vad_filter=False,  # usamos nuestro propio VAD
                    word_timestamps=False,
                )

                segments = []
                full_text_parts = []
                for seg in segments_iter:
                    text = seg.text.strip()
                    if text:
                        segments.append({
                            "start": seg.start,
                            "end": seg.end,
                            "text": text,
                        })
                        full_text_parts.append(text)

                full_text = " ".join(full_text_parts)
                duration_ms = int((time.time() - start_time) * 1000)
                detected_lang = info.language if info else (language or "unknown")

                return TranscriptionResult(
                    text=full_text,
                    segments=segments,
                    language=detected_lang,
                    duration_ms=duration_ms,
                )

            except Exception as e:
                print(f"[Transcriber] Error en transcripción: {e}")
                return None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
