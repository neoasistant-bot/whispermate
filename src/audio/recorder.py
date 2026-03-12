"""
Captura de audio: micrófono (sounddevice) y loopback del sistema (soundcard).
"""

import threading
import warnings
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

try:
    import soundcard as sc
    SOUNDCARD_AVAILABLE = True
except ImportError:
    SOUNDCARD_AVAILABLE = False
    warnings.warn("[Recorder] soundcard no disponible. El modo reunión solo usará micrófono.")


class AudioRecorder:
    SAMPLE_RATE = 16000  # Whisper requiere 16kHz
    BLOCK_SIZE = 1024    # Frames por bloque

    def __init__(self):
        self._recording = False
        self._mic_thread: Optional[threading.Thread] = None
        self._system_thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Modo dictado: solo micrófono
    # ------------------------------------------------------------------

    def start_mic_only(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Graba solo micrófono.
        callback(audio_chunk: np.ndarray) se llama con cada bloque de audio.
        """
        with self._lock:
            if self._recording:
                return
            self._recording = True

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"[Recorder] Status mic: {status}")
            if self._recording:
                # indata shape: (frames, channels) → mono float32
                mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
                callback(mono)

        try:
            self._stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=self.BLOCK_SIZE,
                callback=audio_callback,
            )
            self._stream.start()
            print("[Recorder] Micrófono iniciado.")
        except Exception as e:
            self._recording = False
            raise RuntimeError(f"No se pudo iniciar el micrófono: {e}") from e

    # ------------------------------------------------------------------
    # Modo reunión: micrófono + loopback del sistema
    # ------------------------------------------------------------------

    def start_meeting(self, callback: Callable[[np.ndarray, str], None]) -> None:
        """
        Graba micrófono + loopback del sistema.
        callback(audio_chunk: np.ndarray, source: str) donde source es 'MIC' o 'SYSTEM'.
        """
        with self._lock:
            if self._recording:
                return
            self._recording = True

        # Micrófono via sounddevice
        def mic_callback(indata, frames, time_info, status):
            if status:
                print(f"[Recorder] Status mic: {status}")
            if self._recording:
                mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
                callback(mono, "MIC")

        try:
            self._stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=self.BLOCK_SIZE,
                callback=mic_callback,
            )
            self._stream.start()
            print("[Recorder] Micrófono iniciado (modo reunión).")
        except Exception as e:
            self._recording = False
            raise RuntimeError(f"No se pudo iniciar el micrófono: {e}") from e

        # Loopback del sistema via soundcard (en thread separado)
        if SOUNDCARD_AVAILABLE:
            self._system_thread = threading.Thread(
                target=self._loopback_worker,
                args=(callback,),
                daemon=True,
            )
            self._system_thread.start()
        else:
            print("[Recorder] Loopback no disponible. Solo micrófono activo.")

    def _loopback_worker(self, callback: Callable[[np.ndarray, str], None]) -> None:
        """Thread de captura de audio del sistema via loopback."""
        try:
            speaker = sc.default_speaker()
            if speaker is None:
                print("[Recorder] No se encontró speaker por defecto para loopback.")
                return

            with speaker.recorder(samplerate=self.SAMPLE_RATE, channels=1, blocksize=self.BLOCK_SIZE) as mic:
                print("[Recorder] Loopback del sistema iniciado.")
                while self._recording:
                    data = mic.record(numframes=self.BLOCK_SIZE)
                    if data is not None and self._recording:
                        mono = data[:, 0] if data.ndim > 1 else data
                        callback(mono.astype(np.float32), "SYSTEM")
        except Exception as e:
            print(f"[Recorder] Error en loopback: {e}. Solo micrófono activo.")

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Para la grabación."""
        with self._lock:
            self._recording = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                print(f"[Recorder] Error cerrando stream: {e}")
            self._stream = None

        if self._system_thread is not None:
            self._system_thread.join(timeout=2.0)
            self._system_thread = None

        print("[Recorder] Grabación detenida.")

    # ------------------------------------------------------------------
    # Dispositivos disponibles
    # ------------------------------------------------------------------

    def get_available_devices(self) -> list[dict]:
        """Lista dispositivos de audio disponibles."""
        devices = []
        try:
            sd_devices = sd.query_devices()
            for i, dev in enumerate(sd_devices):
                devices.append({
                    "id": i,
                    "name": dev["name"],
                    "type": "input" if dev["max_input_channels"] > 0 else "output",
                    "channels": dev["max_input_channels"] or dev["max_output_channels"],
                    "source": "sounddevice",
                })
        except Exception as e:
            print(f"[Recorder] Error listando dispositivos: {e}")

        if SOUNDCARD_AVAILABLE:
            try:
                for spk in sc.all_speakers():
                    devices.append({
                        "id": str(spk.id),
                        "name": f"[Loopback] {spk.name}",
                        "type": "loopback",
                        "channels": spk.channels,
                        "source": "soundcard",
                    })
            except Exception as e:
                print(f"[Recorder] Error listando loopback devices: {e}")

        return devices

    @property
    def is_recording(self) -> bool:
        return self._recording
