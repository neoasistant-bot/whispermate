"""
VAD chunking con silero-vad (modo onnxruntime, sin torch).
Detecta segmentos de habla y emite chunks completos para transcripción.
"""

import threading
from typing import Callable, Optional

import numpy as np

from src.config import Config

SAMPLE_RATE = 16000


class VADChunker:
    """
    Acumula frames de audio y usa silero-vad para detectar segmentos de habla.
    Cuando se detecta fin de habla (silencio > min_silence_ms), llama on_segment.

    Usa silero-vad v6 con backend onnxruntime (sin torch).
    """

    def __init__(self, config: Config, on_segment: Callable[[np.ndarray], None]):
        self._config = config
        self._on_segment = on_segment
        self._lock = threading.Lock()

        # Buffers
        self._speech_buffer: list[np.ndarray] = []

        # Estado VAD
        self._is_speaking = False
        self._silence_frames = 0

        # Parámetros
        self._threshold = config.vad_threshold
        self._min_silence_frames = int(config.vad_min_silence_ms * SAMPLE_RATE / 1000)

        # Cargar modelo silero-vad via onnxruntime (sin torch)
        self._model = None
        self._iterator = None
        self._model_loaded = False
        self._load_model()

    def _load_model(self):
        """Carga el modelo silero-vad con backend onnx (no requiere torch)."""
        try:
            from silero_vad import load_silero_vad, VADIterator
            # onnx=True usa onnxruntime en lugar de torch
            self._model = load_silero_vad(onnx=True)
            self._iterator = VADIterator(
                self._model,
                threshold=self._threshold,
                sampling_rate=SAMPLE_RATE,
                min_silence_duration_ms=self._config.vad_min_silence_ms,
                speech_pad_ms=self._config.vad_chunk_padding_ms,
            )
            self._model_loaded = True
            print("[VAD] Modelo silero-vad cargado (onnxruntime).")
        except Exception as e:
            print(f"[VAD] No se pudo cargar silero-vad: {e}. VAD desactivado (pass-through).")
            self._model_loaded = False

    def _run_vad(self, audio_frame: np.ndarray) -> Optional[dict]:
        """
        Corre VAD en un frame con VADIterator.
        Retorna dict con 'start'/'end' en samples si hay evento, None si no.
        """
        if not self._model_loaded or self._iterator is None:
            return None
        try:
            # VADIterator espera tensor de float32
            result = self._iterator(audio_frame, return_seconds=False)
            return result
        except Exception:
            return None

    def feed(self, audio_frame: np.ndarray) -> None:
        """
        Recibe frames continuamente.
        Llama on_segment cuando detecta fin de habla.
        """
        with self._lock:
            if not self._model_loaded:
                # Sin VAD: acumular todo y emitir chunks de 5s
                self._speech_buffer.append(audio_frame.copy())
                total = sum(len(f) for f in self._speech_buffer)
                if total >= SAMPLE_RATE * 5:
                    segment = np.concatenate(self._speech_buffer)
                    self._speech_buffer = []
                    threading.Thread(target=self._on_segment, args=(segment,), daemon=True).start()
                return

            event = self._run_vad(audio_frame)
            self._speech_buffer.append(audio_frame.copy())

            if event:
                if 'start' in event:
                    self._is_speaking = True
                    print(f"[VAD] Inicio de habla")

                if 'end' in event and self._is_speaking:
                    self._is_speaking = False
                    segment = np.concatenate(self._speech_buffer)
                    self._speech_buffer = []
                    print(f"[VAD] Fin de habla. Segmento: {len(segment)/SAMPLE_RATE:.1f}s")
                    threading.Thread(target=self._on_segment, args=(segment,), daemon=True).start()

    def flush(self) -> None:
        """Fuerza el procesamiento del buffer pendiente (al detener grabación)."""
        with self._lock:
            if self._speech_buffer and self._is_speaking:
                segment = np.concatenate(self._speech_buffer)
                self._speech_buffer = []
                self._is_speaking = False
                self._silence_frames = 0
                if len(segment) > SAMPLE_RATE * 0.1:
                    print(f"[VAD] Flush: emitiendo segmento de {len(segment)/SAMPLE_RATE:.1f}s")
                    threading.Thread(target=self._on_segment, args=(segment,), daemon=True).start()
            # Reset iterator para próxima sesión
            if self._model_loaded and self._iterator is not None:
                self._iterator.reset_states()

    def reset(self) -> None:
        """Resetea el estado interno."""
        with self._lock:
            self._speech_buffer = []
            self._is_speaking = False
            self._silence_frames = 0
            if self._model_loaded and self._iterator is not None:
                self._iterator.reset_states()
