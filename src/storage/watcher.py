"""
Watch folder: detecta archivos de audio y los transcribe automáticamente.
"""

import threading
import time
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEventHandler, FileCreatedEvent
from watchdog.observers import Observer

from src.config import Config

AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a", ".mp4", ".webm", ".flac"}


class _AudioHandler(FileSystemEventHandler):
    def __init__(self, on_new_file):
        super().__init__()
        self._on_new_file = on_new_file
        self._processing: set = set()
        self._lock = threading.Lock()

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() in AUDIO_EXTENSIONS:
            with self._lock:
                if str(path) in self._processing:
                    return
                self._processing.add(str(path))
            # Esperar un poco para que el archivo termine de escribirse
            threading.Thread(
                target=self._delayed_process,
                args=(path,),
                daemon=True,
            ).start()

    def _delayed_process(self, path: Path):
        time.sleep(1.5)  # esperar a que el archivo esté completo
        try:
            self._on_new_file(path)
        finally:
            with self._lock:
                self._processing.discard(str(path))


class FolderWatcher:
    def __init__(self, config: Config, transcriber, notify_callback=None):
        """
        config: Config instance
        transcriber: Transcriber instance
        notify_callback: función(title, message) para mostrar notificaciones del sistema
        """
        self._config = config
        self._transcriber = transcriber
        self._notify = notify_callback or self._default_notify
        self._observer: Optional[Observer] = None
        self._running = False

    def _default_notify(self, title: str, message: str):
        print(f"[Watcher] {title}: {message}")

    def start(self) -> bool:
        """Inicia el observer. Retorna True si exitoso."""
        if self._running:
            return True

        watch_dir = self._config.watch_folder
        try:
            watch_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"[Watcher] No se pudo crear watch folder: {e}")
            return False

        handler = _AudioHandler(on_new_file=self._on_new_audio_file)
        self._observer = Observer()
        self._observer.schedule(handler, str(watch_dir), recursive=False)

        try:
            self._observer.start()
            self._running = True
            print(f"[Watcher] Observando: {watch_dir}")
            return True
        except Exception as e:
            print(f"[Watcher] Error iniciando observer: {e}")
            return False

    def stop(self):
        """Detiene el observer."""
        if self._observer and self._running:
            self._observer.stop()
            self._observer.join(timeout=3.0)
            self._observer = None
            self._running = False
            print("[Watcher] Detenido.")

    def _on_new_audio_file(self, audio_path: Path):
        """Callback cuando aparece un nuevo archivo de audio."""
        print(f"[Watcher] Nuevo archivo detectado: {audio_path.name}")
        self._notify("WhisperMate", f"Transcribiendo: {audio_path.name}...")

        output_path = audio_path.with_suffix(".md")

        try:
            import numpy as np
            audio = self._load_audio(audio_path)
            if audio is None:
                self._notify("WhisperMate", f"❌ Error: no se pudo cargar {audio_path.name}")
                return

            result = self._transcriber.transcribe(
                audio,
                language=self._config.language,
            )

            if result is None:
                self._notify("WhisperMate", f"❌ Error en transcripción: {audio_path.name}")
                return

            # Escribir el .md
            self._write_md(audio_path, output_path, result)
            self._notify("WhisperMate", f"✅ Listo: {output_path.name}")
            print(f"[Watcher] Transcripción guardada: {output_path}")

        except Exception as e:
            print(f"[Watcher] Error procesando {audio_path.name}: {e}")
            self._notify("WhisperMate", f"❌ Error: {audio_path.name} — {e}")

    def _load_audio(self, path: Path):
        """Carga un archivo de audio como numpy array a 16kHz mono."""
        try:
            # Intentar con soundfile primero (para .wav)
            if path.suffix.lower() == ".wav":
                import soundfile as sf
                import numpy as np
                audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != 16000:
                    audio = self._resample(audio, sr, 16000)
                return audio

            # Para otros formatos, intentar con librosa o ffmpeg
            try:
                import librosa
                audio, sr = librosa.load(str(path), sr=16000, mono=True)
                return audio
            except ImportError:
                pass

            # Fallback: usar ffmpeg
            return self._load_with_ffmpeg(path)

        except Exception as e:
            print(f"[Watcher] Error cargando audio: {e}")
            return None

    def _load_with_ffmpeg(self, path: Path):
        """Carga audio usando ffmpeg como proceso externo."""
        import subprocess
        import numpy as np

        cmd = [
            "ffmpeg", "-i", str(path),
            "-ar", "16000",
            "-ac", "1",
            "-f", "f32le",
            "-",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=300,
            )
            if result.returncode == 0:
                audio = np.frombuffer(result.stdout, dtype=np.float32)
                return audio
            else:
                print(f"[Watcher] ffmpeg error: {result.stderr.decode()}")
                return None
        except FileNotFoundError:
            print("[Watcher] ffmpeg no encontrado. Instalar para soportar MP3/M4A/etc.")
            return None
        except subprocess.TimeoutExpired:
            print("[Watcher] ffmpeg timeout.")
            return None

    def _resample(self, audio, orig_sr: int, target_sr: int):
        """Resamplea audio a target_sr."""
        import numpy as np
        try:
            import scipy.signal as signal
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            return signal.resample(audio, new_length).astype(np.float32)
        except ImportError:
            # Resample simple sin scipy
            indices = np.linspace(0, len(audio) - 1, int(len(audio) * target_sr / orig_sr))
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def _write_md(self, audio_path: Path, output_path: Path, result) -> None:
        """Escribe el resultado de transcripción en un archivo .md."""
        from datetime import datetime
        now = datetime.now()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Transcripción — {audio_path.name}\n\n")
            f.write(f"**Fecha:** {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Idioma detectado:** {result.language}\n")
            f.write(f"**Modelo:** {self._transcriber.model_name}\n\n")
            f.write("---\n\n")

            if result.segments:
                for seg in result.segments:
                    start = seg.get("start", 0)
                    minutes = int(start // 60)
                    seconds = start % 60
                    ts = f"{minutes:02d}:{seconds:05.2f}"
                    f.write(f"**[{ts}]:** {seg['text']}\n\n")
            else:
                f.write(result.text + "\n")

            duration_s = result.duration_ms / 1000
            f.write(f"\n---\n\n_Procesado en {duration_s:.1f}s_\n")

    @property
    def is_running(self) -> bool:
        return self._running
