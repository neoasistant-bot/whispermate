"""
Ventana de configuración de WhisperMate.
"""

from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QComboBox, QLineEdit, QPushButton,
    QFileDialog, QCheckBox, QGroupBox, QDialogButtonBox,
    QMessageBox,
)
from PyQt6.QtCore import Qt

from src.config import Config

MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
LANGUAGES = [
    ("Español", "es"),
    ("English", "en"),
    ("Auto-detect", "auto"),
    ("Français", "fr"),
    ("Deutsch", "de"),
    ("Italiano", "it"),
    ("Português", "pt"),
    ("日本語", "ja"),
    ("中文", "zh"),
]


class SettingsWindow(QDialog):
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self._config = config
        self._result_saved = False

        self.setWindowTitle("WhisperMate — Configuración")
        self.setMinimumWidth(480)
        self.setModal(True)

        self._build_ui()
        self._load_values()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # === Transcripción ===
        transcription_group = QGroupBox("Transcripción")
        transcription_layout = QFormLayout(transcription_group)

        self._model_combo = QComboBox()
        self._model_combo.addItems(MODELS)
        transcription_layout.addRow("Modelo Whisper:", self._model_combo)

        self._lang_combo = QComboBox()
        for name, _ in LANGUAGES:
            self._lang_combo.addItem(name)
        transcription_layout.addRow("Idioma:", self._lang_combo)

        layout.addWidget(transcription_group)

        # === Carpetas ===
        folders_group = QGroupBox("Carpetas")
        folders_layout = QFormLayout(folders_group)

        # Output folder
        output_row = QHBoxLayout()
        self._output_folder_edit = QLineEdit()
        self._output_folder_edit.setPlaceholderText("~/Transcripciones")
        output_browse_btn = QPushButton("…")
        output_browse_btn.setFixedWidth(30)
        output_browse_btn.clicked.connect(self._browse_output_folder)
        output_row.addWidget(self._output_folder_edit)
        output_row.addWidget(output_browse_btn)
        folders_layout.addRow("Carpeta de salida:", output_row)

        # Watch folder
        watch_row = QHBoxLayout()
        self._watch_folder_edit = QLineEdit()
        self._watch_folder_edit.setPlaceholderText("~/Transcripciones/inbox")
        watch_browse_btn = QPushButton("…")
        watch_browse_btn.setFixedWidth(30)
        watch_browse_btn.clicked.connect(self._browse_watch_folder)
        watch_row.addWidget(self._watch_folder_edit)
        watch_row.addWidget(watch_browse_btn)
        folders_layout.addRow("Watch folder:", watch_row)

        self._watch_enabled_cb = QCheckBox("Activar watch folder")
        folders_layout.addRow("", self._watch_enabled_cb)

        layout.addWidget(folders_group)

        # === Shortcuts ===
        shortcuts_group = QGroupBox("Atajos de teclado")
        shortcuts_layout = QFormLayout(shortcuts_group)

        self._shortcut_dictation_edit = QLineEdit()
        self._shortcut_dictation_edit.setPlaceholderText("ctrl+shift+r")
        shortcuts_layout.addRow("🎤 Dictado:", self._shortcut_dictation_edit)

        self._shortcut_meeting_edit = QLineEdit()
        self._shortcut_meeting_edit.setPlaceholderText("ctrl+shift+m")
        shortcuts_layout.addRow("👥 Reunión:", self._shortcut_meeting_edit)

        layout.addWidget(shortcuts_group)

        # === Botones ===
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._save)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _load_values(self):
        """Carga los valores actuales de config en los widgets."""
        # Modelo
        model = self._config.model
        idx = MODELS.index(model) if model in MODELS else 0
        self._model_combo.setCurrentIndex(idx)

        # Idioma
        lang = self._config.language
        lang_codes = [code for _, code in LANGUAGES]
        lang_idx = lang_codes.index(lang) if lang in lang_codes else 0
        self._lang_combo.setCurrentIndex(lang_idx)

        # Carpetas
        self._output_folder_edit.setText(str(self._config.output_folder))
        self._watch_folder_edit.setText(str(self._config.watch_folder))
        self._watch_enabled_cb.setChecked(self._config.watch_folder_enabled)

        # Shortcuts
        shortcuts = self._config.shortcuts
        self._shortcut_dictation_edit.setText(shortcuts.get("dictation", "ctrl+shift+r"))
        self._shortcut_meeting_edit.setText(shortcuts.get("meeting", "ctrl+shift+m"))

    def _browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Seleccionar carpeta de salida",
            str(self._config.output_folder),
        )
        if folder:
            self._output_folder_edit.setText(folder)

    def _browse_watch_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Seleccionar watch folder",
            str(self._config.watch_folder),
        )
        if folder:
            self._watch_folder_edit.setText(folder)

    def _save(self):
        """Guarda la configuración."""
        # Modelo
        self._config.model = self._model_combo.currentText()

        # Idioma
        lang_idx = self._lang_combo.currentIndex()
        self._config.language = LANGUAGES[lang_idx][1]

        # Carpetas
        output = self._output_folder_edit.text().strip()
        if output:
            self._config.output_folder = output

        watch = self._watch_folder_edit.text().strip()
        if watch:
            self._config.watch_folder = watch

        self._config.watch_folder_enabled = self._watch_enabled_cb.isChecked()

        # Shortcuts
        self._config.shortcuts = {
            "dictation": self._shortcut_dictation_edit.text().strip() or "ctrl+shift+r",
            "meeting": self._shortcut_meeting_edit.text().strip() or "ctrl+shift+m",
        }

        self._config.save()
        self._result_saved = True
        self.accept()

    @property
    def saved(self) -> bool:
        return self._result_saved
