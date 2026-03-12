"""
Manejo de archivos .md para transcripciones.
"""

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import Config

MONTH_NAMES_ES = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
}

MODE_NAMES_ES = {
    "dictation": "Dictado",
    "meeting": "Reunión",
}


class FileManager:
    def __init__(self, config: Config):
        self._config = config
        self._current_file: Optional[Path] = None
        self._session_start: Optional[datetime] = None
        self._segment_count = 0
        self._lock = threading.Lock()

    def start_session(self, mode: str) -> Path:
        """
        Crea un nuevo archivo .md para la sesión actual.
        Retorna el path del archivo creado.
        """
        with self._lock:
            now = datetime.now()
            self._session_start = now
            self._segment_count = 0

            # Asegurar que existe el directorio de salida
            output_dir = self._config.output_folder
            output_dir.mkdir(parents=True, exist_ok=True)

            # Nombre del archivo
            filename = now.strftime(f"%Y-%m-%d_%H-%M-%S_{mode}.md")
            filepath = output_dir / filename

            # Header del .md
            day = now.day
            month_name = MONTH_NAMES_ES.get(now.month, str(now.month))
            year = now.year
            time_str = now.strftime("%H:%M")
            mode_display = MODE_NAMES_ES.get(mode, mode.capitalize())
            model_name = self._config.model

            header = f"""# Transcripción — {day} de {month_name} de {year}, {time_str}
**Modo:** {mode_display}
**Modelo:** {model_name}

---

"""
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(header)

            self._current_file = filepath
            print(f"[FileManager] Sesión iniciada: {filepath}")
            return filepath

    def append_segment(
        self,
        speaker: Optional[str],
        timestamp: str,
        text: str,
    ) -> None:
        """
        Agrega un segmento al .md abierto.
        
        Formato con speaker:  **[14:32:05] MIC:** texto transcripto
        Formato sin speaker:  **[14:32:05]:** texto transcripto
        """
        with self._lock:
            if self._current_file is None:
                print("[FileManager] No hay sesión activa. Ignorando segmento.")
                return

            if speaker:
                line = f"**[{timestamp}] {speaker}:** {text}\n\n"
            else:
                line = f"**[{timestamp}]:** {text}\n\n"

            try:
                with open(self._current_file, "a", encoding="utf-8") as f:
                    f.write(line)
                self._segment_count += 1
            except OSError as e:
                print(f"[FileManager] Error escribiendo segmento: {e}")

    def end_session(self) -> Optional[Path]:
        """
        Cierra el archivo, agrega footer con duración total.
        Retorna el path del archivo cerrado.
        """
        with self._lock:
            if self._current_file is None:
                return None

            filepath = self._current_file
            self._current_file = None

            if self._session_start:
                now = datetime.now()
                duration = now - self._session_start
                total_seconds = int(duration.total_seconds())
                minutes = total_seconds // 60
                seconds = total_seconds % 60

                footer = f"\n---\n\n_Sesión finalizada: {now.strftime('%H:%M:%S')} · Duración: {minutes}m {seconds}s · Segmentos: {self._segment_count}_\n"

                try:
                    with open(filepath, "a", encoding="utf-8") as f:
                        f.write(footer)
                except OSError as e:
                    print(f"[FileManager] Error escribiendo footer: {e}")

            self._session_start = None
            self._segment_count = 0
            print(f"[FileManager] Sesión cerrada: {filepath}")
            return filepath

    @property
    def current_file(self) -> Optional[Path]:
        return self._current_file

    @property
    def segment_count(self) -> int:
        return self._segment_count

    @property
    def is_active(self) -> bool:
        return self._current_file is not None
