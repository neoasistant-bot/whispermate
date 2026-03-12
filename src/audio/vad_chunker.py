"""
VAD chunking con silero-vad.
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
    """

    def __init__(self, config: Config, on_segment: Callable[[np.ndarray], None]):
        self._config = config
        self._on_segment = on_segment
        self._lock = threading.Lock()

        # Buffers
        self._audio_buffer: list[np.ndarray] = []
        self._speech_buffer: list[np.ndarray] = []

        # Estado VAD
        self._is_speaking = False
        self._silence_frames = 0

        # Parámetros
        self._threshold = config.vad_threshold
        self._min_silence_frames = int(config.vad_min_silence_ms * SAMPLE_RATE / 1000)
        self._padding_frames = int(config.vad_chunk_padding_ms * SAMPLE_RATE / 1000)

        # Cargar modelo (lazy)
        self._model = None
        self._model_loaded = False
        self._load_model()

    def _load_model(self):
        """Carga el modelo silero-vad desde torch.hub."""
        try:
            import torch
            # Suprimir output de torch.hub
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    trust_repo=True,
                )
            self._model.eval()
            self._model_loaded = True
            print("[VAD] Modelo silero-vad cargado.")
        except Exception as e:
            print(f"[VAD] No se pudo cargar silero-vad: {e}. VAD desactivado.")
            self._model_loaded = False

    def _run_vad(self, audio_frame: np.ndarray) -> float:
        """Corre VAD en un frame. Retorna probabilidad de voz (0.0-1.0)."""
        if not self._model_loaded:
            # Sin VAD: asumir que siempre hay habla
            return 1.0
        try:
            import torch
            tensor = torch.FloatTensor(audio_frame)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            with torch.no_grad():
                prob = self._model(tensor, SAMPLE_RATE).item()
            return prob
        except Exception as e:
            print(f"[VAD] Error en inferencia: {e}")
            return 0.0

    def feed(self, audio_frame: np.ndarray) -> None:
        """
        Recibe frames continuamente.
        Llama on_segment cuando detecta fin de habla.
        """
        with self._lock:
            prob = self._run_vad(audio_frame)
            is_voice = prob >= self._threshold

            if is_voice:
                if not self._is_speaking:
                    # Inicio de segmento: agregar padding previo si hay
                    self._is_speaking = True
                    print(f"[VAD] Inicio de habla (prob={prob:.2f})")
                self._silence_frames = 0
                self._speech_buffer.append(audio_frame.copy())
            else:
                if self._is_speaking:
                    self._silence_frames += len(audio_frame)
                    self._speech_buffer.append(audio_frame.copy())  # incluir silencio post

                    if self._silence_frames >= self._min_silence_frames:
                        # Fin de segmento
                        self._is_speaking = False
                        segment = np.concatenate(self._speech_buffer)
                        self._speech_buffer = []
                        self._silence_frames = 0
                        print(f"[VAD] Fin de habla. Segmento: {len(segment)/SAMPLE_RATE:.1f}s")
                        # Llamar callback fuera del lock
                        threading.Thread(
                            target=self._on_segment,
                            args=(segment,),
                            daemon=True,
                        ).start()

    def flush(self) -> None:
        """Fuerza el procesamiento del buffer pendiente (al detener grabación)."""
        with self._lock:
            if self._speech_buffer:
                segment = np.concatenate(self._speech_buffer)
                self._speech_buffer = []
                self._is_speaking = False
                self._silence_frames = 0
                if len(segment) > SAMPLE_RATE * 0.1:  # mínimo 100ms
                    print(f"[VAD] Flush: emitiendo segmento de {len(segment)/SAMPLE_RATE:.1f}s")
                    threading.Thread(
                        target=self._on_segment,
                        args=(segment,),
                        daemon=True,
                    ).start()

    def reset(self) -> None:
        """Resetea el estado interno."""
        with self._lock:
            self._audio_buffer = []
            self._speech_buffer = []
            self._is_speaking = False
            self._silence_frames = 0
