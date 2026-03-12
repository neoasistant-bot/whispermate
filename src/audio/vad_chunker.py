"""
VAD chunking por energía de audio (RMS threshold).
No requiere torch ni onnxruntime adicional — usa solo numpy.
Detecta segmentos de habla y emite chunks completos para transcripción.
"""

import threading
from typing import Callable

import numpy as np

from src.config import Config

SAMPLE_RATE = 16000


class VADChunker:
    """
    Acumula frames de audio y detecta segmentos de habla por energía RMS.
    Cuando detecta silencio sostenido después de habla, emite el segmento.

    Ventajas sobre silero-vad: cero dependencias extra, misma efectividad
    para separar pausas naturales en conversaciones.
    """

    def __init__(self, config: Config, on_segment: Callable[[np.ndarray], None]):
        self._config = config
        self._on_segment = on_segment
        self._lock = threading.Lock()

        # Parámetros
        self._threshold = 0.01            # RMS mínimo para considerar habla
        self._min_silence_frames = int(config.vad_min_silence_ms * SAMPLE_RATE / 1000)
        self._padding_frames = int(config.vad_chunk_padding_ms * SAMPLE_RATE / 1000)
        self._max_segment_frames = SAMPLE_RATE * 30  # máximo 30s por segmento

        # Estado
        self._speech_buffer: list[np.ndarray] = []
        self._is_speaking = False
        self._silence_frames = 0

        print("[VAD] Energy-based VAD iniciado (sin dependencias externas).")

    def _is_voice(self, frame: np.ndarray) -> bool:
        """Detecta voz por RMS de energía."""
        rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
        return rms > self._threshold

    def feed(self, audio_frame: np.ndarray) -> None:
        """
        Recibe frames continuamente.
        Llama on_segment cuando detecta fin de habla (silencio > min_silence_ms).
        """
        with self._lock:
            has_voice = self._is_voice(audio_frame)

            if has_voice:
                if not self._is_speaking:
                    self._is_speaking = True
                self._silence_frames = 0
                self._speech_buffer.append(audio_frame.copy())
            else:
                if self._is_speaking:
                    self._silence_frames += len(audio_frame)
                    self._speech_buffer.append(audio_frame.copy())

                    total_frames = sum(len(f) for f in self._speech_buffer)

                    # Emitir si hay suficiente silencio O si el segmento es muy largo
                    if (self._silence_frames >= self._min_silence_frames or
                            total_frames >= self._max_segment_frames):
                        self._emit_segment()

            # Emitir segmento si se acumuló demasiado audio sin silencio
            if self._is_speaking:
                total = sum(len(f) for f in self._speech_buffer)
                if total >= self._max_segment_frames:
                    self._emit_segment()

    def _emit_segment(self):
        """Emite el segmento acumulado (debe llamarse con el lock tomado)."""
        if not self._speech_buffer:
            return
        segment = np.concatenate(self._speech_buffer)
        self._speech_buffer = []
        self._is_speaking = False
        self._silence_frames = 0
        duration = len(segment) / SAMPLE_RATE
        print(f"[VAD] Segmento detectado: {duration:.1f}s")
        # Llamar callback fuera del lock en thread separado
        threading.Thread(
            target=self._on_segment,
            args=(segment,),
            daemon=True,
        ).start()

    def flush(self) -> None:
        """Fuerza el procesamiento del buffer pendiente (al detener grabación)."""
        with self._lock:
            if self._speech_buffer:
                total = sum(len(f) for f in self._speech_buffer)
                if total > SAMPLE_RATE * 0.3:  # mínimo 300ms para transcribir
                    print(f"[VAD] Flush: {total/SAMPLE_RATE:.1f}s pendientes")
                    self._emit_segment()
                else:
                    self._speech_buffer = []
                    self._is_speaking = False

    def reset(self) -> None:
        """Resetea el estado interno."""
        with self._lock:
            self._speech_buffer = []
            self._is_speaking = False
            self._silence_frames = 0
