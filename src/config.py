"""
Manejo de configuración de WhisperMate.
Carga config.json desde el directorio raíz del proyecto.
"""

import json
import os
import shutil
from pathlib import Path

# Directorio raíz del proyecto (donde está main.py)
ROOT_DIR = Path(__file__).parent.parent

DEFAULTS = {
    "model": "small",
    "language": "es",
    "output_folder": "~/Transcripciones",
    "watch_folder": "~/Transcripciones/inbox",
    "watch_folder_enabled": False,
    "shortcuts": {
        "dictation": "ctrl+shift+r",
        "meeting": "ctrl+shift+m",
    },
    "vad_threshold": 0.5,
    "vad_min_silence_ms": 800,
    "vad_chunk_padding_ms": 200,
}


class Config:
    def __init__(self):
        self._config_path = ROOT_DIR / "config.json"
        self._data = {}
        self._load()

    def _load(self):
        if not self._config_path.exists():
            example = ROOT_DIR / "config.json.example"
            if example.exists():
                shutil.copy(example, self._config_path)
            else:
                self._data = dict(DEFAULTS)
                self.save()
                return

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            # Merge con defaults para campos faltantes
            self._data = {**DEFAULTS, **loaded}
            # Merge nested shortcuts
            if "shortcuts" in loaded:
                self._data["shortcuts"] = {**DEFAULTS["shortcuts"], **loaded["shortcuts"]}
        except (json.JSONDecodeError, OSError) as e:
            print(f"[Config] Error cargando config.json: {e}. Usando defaults.")
            self._data = dict(DEFAULTS)

    def save(self):
        try:
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except OSError as e:
            print(f"[Config] Error guardando config.json: {e}")

    # --- Accessors ---

    @property
    def model(self) -> str:
        return self._data.get("model", DEFAULTS["model"])

    @model.setter
    def model(self, value: str):
        self._data["model"] = value

    @property
    def language(self) -> str:
        return self._data.get("language", DEFAULTS["language"])

    @language.setter
    def language(self, value: str):
        self._data["language"] = value

    @property
    def output_folder(self) -> Path:
        return Path(self._data.get("output_folder", DEFAULTS["output_folder"])).expanduser()

    @output_folder.setter
    def output_folder(self, value: str):
        self._data["output_folder"] = value

    @property
    def watch_folder(self) -> Path:
        return Path(self._data.get("watch_folder", DEFAULTS["watch_folder"])).expanduser()

    @watch_folder.setter
    def watch_folder(self, value: str):
        self._data["watch_folder"] = value

    @property
    def watch_folder_enabled(self) -> bool:
        return self._data.get("watch_folder_enabled", False)

    @watch_folder_enabled.setter
    def watch_folder_enabled(self, value: bool):
        self._data["watch_folder_enabled"] = value

    @property
    def shortcuts(self) -> dict:
        return self._data.get("shortcuts", DEFAULTS["shortcuts"])

    @shortcuts.setter
    def shortcuts(self, value: dict):
        self._data["shortcuts"] = value

    @property
    def vad_threshold(self) -> float:
        return float(self._data.get("vad_threshold", DEFAULTS["vad_threshold"]))

    @property
    def vad_min_silence_ms(self) -> int:
        return int(self._data.get("vad_min_silence_ms", DEFAULTS["vad_min_silence_ms"]))

    @property
    def vad_chunk_padding_ms(self) -> int:
        return int(self._data.get("vad_chunk_padding_ms", DEFAULTS["vad_chunk_padding_ms"]))

    def get_raw(self) -> dict:
        return dict(self._data)
