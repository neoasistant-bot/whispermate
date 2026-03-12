"""
Captura de audio: micrófono y loopback del sistema via sounddevice (WASAPI en Windows).
"""

import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd


class AudioRecorder:
    SAMPLE_RATE = 16000  # Whisper requiere 16kHz
    BLOCK_SIZE = 1024    # Frames por bloque (~64ms a 16kHz)

    def __init__(self):
        self._recording = False
        self._mic_stream: Optional[sd.InputStream] = None
        self._loopback_stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Modo dictado: solo micrófono
    # ------------------------------------------------------------------

    def start_mic_only(self, callback: Callable[[np.ndarray], None]) -> None:
        """Graba solo micrófono. callback(audio_chunk) con cada bloque."""
        with self._lock:
            if self._recording:
                return
            self._recording = True

        def audio_callback(indata, frames, time_info, status):
            if self._recording:
                mono = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()
                callback(mono.astype(np.float32))

        try:
            self._mic_stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=self.BLOCK_SIZE,
                callback=audio_callback,
            )
            self._mic_stream.start()
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
        callback(audio_chunk, source) donde source es 'MIC' o 'SYSTEM'.
        """
        with self._lock:
            if self._recording:
                return
            self._recording = True

        # --- Micrófono ---
        def mic_callback(indata, frames, time_info, status):
            if self._recording:
                mono = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()
                callback(mono.astype(np.float32), "MIC")

        try:
            self._mic_stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=self.BLOCK_SIZE,
                callback=mic_callback,
            )
            self._mic_stream.start()
            print("[Recorder] Micrófono iniciado (modo reunión).")
        except Exception as e:
            self._recording = False
            raise RuntimeError(f"No se pudo iniciar el micrófono: {e}") from e

        # --- Loopback del sistema ---
        # En Windows: busca dispositivo WASAPI loopback (Stereo Mix o similar)
        # En otros SO: intenta el dispositivo de loopback disponible
        loopback_device = self._find_loopback_device()
        if loopback_device is not None:
            def loopback_callback(indata, frames, time_info, status):
                if self._recording:
                    mono = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()
                    callback(mono.astype(np.float32), "SYSTEM")
            try:
                # Obtener sample rate nativo del dispositivo loopback
                dev_info = sd.query_devices(loopback_device)
                native_sr = int(dev_info['default_samplerate'])
                channels = min(2, dev_info['max_input_channels'])

                self._loopback_stream = sd.InputStream(
                    device=loopback_device,
                    samplerate=native_sr if native_sr > 0 else self.SAMPLE_RATE,
                    channels=channels,
                    dtype="float32",
                    blocksize=self.BLOCK_SIZE,
                    callback=loopback_callback,
                )
                self._loopback_stream.start()
                print(f"[Recorder] Loopback iniciado: {dev_info['name']}")
            except Exception as e:
                print(f"[Recorder] No se pudo iniciar loopback: {e}. Solo micrófono.")
                self._loopback_stream = None
        else:
            print("[Recorder] No se encontró dispositivo loopback. Solo micrófono activo.")
            print("[Recorder] Tip Windows: activá 'Stereo Mix' en Configuración de Sonido > Dispositivos de grabación.")

    def _find_loopback_device(self) -> Optional[int]:
        """
        Busca un dispositivo de loopback disponible.
        En Windows busca 'Stereo Mix', 'What U Hear' o similar.
        En Linux busca dispositivos 'monitor'.
        """
        try:
            devices = sd.query_devices()
            # Keywords que indican dispositivos de loopback
            loopback_keywords = [
                'stereo mix', 'what u hear', 'wave out mix',
                'monitor', 'loopback', 'mix', 'mezcla estéreo',
            ]
            for i, dev in enumerate(devices):
                name_lower = dev['name'].lower()
                if dev['max_input_channels'] > 0:
                    for kw in loopback_keywords:
                        if kw in name_lower:
                            return i
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Para toda grabación activa."""
        with self._lock:
            self._recording = False

        for stream_attr in ('_mic_stream', '_loopback_stream'):
            stream = getattr(self, stream_attr, None)
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception as e:
                    print(f"[Recorder] Error cerrando stream: {e}")
                setattr(self, stream_attr, None)

        print("[Recorder] Grabación detenida.")

    # ------------------------------------------------------------------
    # Dispositivos disponibles
    # ------------------------------------------------------------------

    def get_available_devices(self) -> list[dict]:
        """Lista dispositivos de audio de entrada disponibles."""
        devices = []
        try:
            for i, dev in enumerate(sd.query_devices()):
                if dev['max_input_channels'] > 0:
                    devices.append({
                        "id": i,
                        "name": dev['name'],
                        "channels": dev['max_input_channels'],
                        "sample_rate": dev['default_samplerate'],
                    })
        except Exception as e:
            print(f"[Recorder] Error listando dispositivos: {e}")
        return devices

    @property
    def is_recording(self) -> bool:
        return self._recording
