"""
System tray de WhisperMate.
Maneja el ícono, menú, estados y coordinación de todos los componentes.
"""

import os
import subprocess
import sys
import threading
from datetime import datetime
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon, QColor, QPixmap, QPainter
from PyQt6.QtWidgets import QSystemTrayIcon, QMenu, QApplication, QMessageBox

from src.config import Config
from src.audio.recorder import AudioRecorder
from src.audio.vad_chunker import VADChunker
from src.transcription.transcriber import Transcriber
from src.storage.file_manager import FileManager
from src.storage.watcher import FolderWatcher


# Estados
STATE_IDLE = "idle"
STATE_DICTATING = "dictating"
STATE_MEETING = "meeting"
STATE_TRANSCRIBING = "transcribing"


class _Signals(QObject):
    """Señales Qt para comunicación entre threads."""
    status_changed = pyqtSignal(str)
    notification = pyqtSignal(str, str)   # title, message
    transcription_done = pyqtSignal(str, str)  # mode, filepath
    error = pyqtSignal(str)


def _make_icon(color: str, size: int = 22) -> QIcon:
    """Crea un ícono de color sólido simple (círculo)."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(color))
    painter.setPen(QColor(color).darker(120))
    margin = 2
    painter.drawEllipse(margin, margin, size - 2 * margin, size - 2 * margin)
    painter.end()
    return QIcon(pixmap)


# Los íconos se crean en TrayApp.__init__() después de que QApplication existe.
ICONS: dict = {}


class TrayApp(QSystemTrayIcon):
    def __init__(self, config: Config):
        super().__init__()
        self._config = config
        self._state = STATE_IDLE
        self._signals = _Signals()

        # Crear íconos ahora que QApplication ya existe
        ICONS.update({
            STATE_IDLE: _make_icon("#888888"),
            STATE_DICTATING: _make_icon("#22cc55"),
            STATE_MEETING: _make_icon("#cc2222"),
            STATE_TRANSCRIBING: _make_icon("#f5a623"),
        })

        # Componentes
        self._recorder = AudioRecorder()
        self._transcriber = Transcriber(model_name=config.model)
        self._file_manager = FileManager(config)
        self._vad_chunker: Optional[VADChunker] = None
        self._watcher: Optional[FolderWatcher] = None

        # Buffer de audio para dictado
        self._dictation_buffer: list[np.ndarray] = []
        self._dictation_lock = threading.Lock()

        # Contador de transcripciones pendientes en modo reunión
        self._pending_segments = 0
        self._pending_lock = threading.Lock()

        # Conectar señales
        self._signals.status_changed.connect(self._on_status_changed)
        self._signals.notification.connect(self._show_notification)
        self._signals.transcription_done.connect(self._on_transcription_done)
        self._signals.error.connect(self._on_error)

        # Construir UI
        self._build_menu()
        self.setIcon(ICONS[STATE_IDLE])
        self.setToolTip("WhisperMate — Listo")

        # Double click → settings
        self.activated.connect(self._on_activated)

        # Registrar hotkeys
        self._register_hotkeys()

        # Iniciar watch folder si está habilitado
        if config.watch_folder_enabled:
            self._start_watcher()

    # ------------------------------------------------------------------
    # Menú
    # ------------------------------------------------------------------

    def _build_menu(self):
        menu = QMenu()

        self._status_action = menu.addAction("Estado: Listo")
        self._status_action.setEnabled(False)
        menu.addSeparator()

        self._dictation_action = menu.addAction("🎤  Iniciar dictado  (Ctrl+Shift+R)")
        self._dictation_action.triggered.connect(self.toggle_dictation)

        self._meeting_action = menu.addAction("👥  Iniciar reunión  (Ctrl+Shift+M)")
        self._meeting_action.triggered.connect(self.toggle_meeting)

        menu.addSeparator()

        open_folder_action = menu.addAction("📂  Abrir carpeta de salida")
        open_folder_action.triggered.connect(self._open_output_folder)

        settings_action = menu.addAction("⚙️   Configuración")
        settings_action.triggered.connect(self._open_settings)

        menu.addSeparator()

        quit_action = menu.addAction("✕  Salir")
        quit_action.triggered.connect(self._quit)

        self.setContextMenu(menu)

    def _update_menu_state(self):
        """Actualiza labels del menú según el estado actual."""
        if self._state == STATE_IDLE:
            self._status_action.setText("Estado: Listo")
            self._dictation_action.setText("🎤  Iniciar dictado  (Ctrl+Shift+R)")
            self._meeting_action.setText("👥  Iniciar reunión  (Ctrl+Shift+M)")
            self._dictation_action.setEnabled(True)
            self._meeting_action.setEnabled(True)

        elif self._state == STATE_DICTATING:
            self._status_action.setText("Estado: 🔴 Grabando dictado...")
            self._dictation_action.setText("⏹  Detener dictado  (Ctrl+Shift+R)")
            self._meeting_action.setEnabled(False)

        elif self._state == STATE_MEETING:
            self._status_action.setText("Estado: 🔴 Grabando reunión...")
            self._meeting_action.setText("⏹  Detener reunión  (Ctrl+Shift+M)")
            self._dictation_action.setEnabled(False)

        elif self._state == STATE_TRANSCRIBING:
            self._status_action.setText("Estado: ⏳ Transcribiendo...")
            self._dictation_action.setEnabled(False)
            self._meeting_action.setEnabled(False)

    # ------------------------------------------------------------------
    # Hotkeys
    # ------------------------------------------------------------------

    def _register_hotkeys(self):
        """Registra los hotkeys globales con la librería keyboard."""
        try:
            import keyboard
            shortcuts = self._config.shortcuts
            keyboard.add_hotkey(shortcuts.get("dictation", "ctrl+shift+r"), self.toggle_dictation)
            keyboard.add_hotkey(shortcuts.get("meeting", "ctrl+shift+m"), self.toggle_meeting)
            print(f"[Tray] Hotkeys registrados: {shortcuts}")
        except Exception as e:
            print(f"[Tray] No se pudieron registrar hotkeys: {e}")
            print("[Tray] Tip: en Linux puede requerir permisos de root o grupo 'input'.")

    def _unregister_hotkeys(self):
        try:
            import keyboard
            keyboard.unhook_all_hotkeys()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Toggle dictado
    # ------------------------------------------------------------------

    def toggle_dictation(self):
        if self._state == STATE_IDLE:
            self._start_dictation()
        elif self._state == STATE_DICTATING:
            self._stop_dictation()

    def _start_dictation(self):
        self._dictation_buffer = []
        try:
            self._recorder.start_mic_only(callback=self._dictation_audio_callback)
            self._set_state(STATE_DICTATING)
        except RuntimeError as e:
            self._signals.error.emit(str(e))

    def _dictation_audio_callback(self, audio_chunk: np.ndarray):
        with self._dictation_lock:
            self._dictation_buffer.append(audio_chunk.copy())

    def _stop_dictation(self):
        self._recorder.stop()
        self._set_state(STATE_TRANSCRIBING)

        with self._dictation_lock:
            if not self._dictation_buffer:
                self._set_state(STATE_IDLE)
                return
            audio = np.concatenate(self._dictation_buffer)
            self._dictation_buffer = []

        threading.Thread(
            target=self._process_dictation,
            args=(audio,),
            daemon=True,
        ).start()

    def _process_dictation(self, audio: np.ndarray):
        result = self._transcriber.transcribe(audio, language=self._config.language)
        if result is None or not result.text.strip():
            self._signals.error.emit("No se detectó texto en la grabación.")
            self._signals.status_changed.emit(STATE_IDLE)
            return

        filepath = self._file_manager.start_session("dictation")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._file_manager.append_segment(None, timestamp, result.text)
        final_path = self._file_manager.end_session()

        self._signals.transcription_done.emit("dictation", str(final_path))
        self._signals.status_changed.emit(STATE_IDLE)

    # ------------------------------------------------------------------
    # Toggle reunión
    # ------------------------------------------------------------------

    def toggle_meeting(self):
        if self._state == STATE_IDLE:
            self._start_meeting()
        elif self._state == STATE_MEETING:
            self._stop_meeting()

    def _start_meeting(self):
        self._file_manager.start_session("meeting")
        self._pending_segments = 0          # contador de transcripciones en curso
        self._pending_lock = threading.Lock()
        self._vad_chunker = VADChunker(
            config=self._config,
            on_segment=self._on_meeting_segment,
        )
        try:
            self._recorder.start_meeting(callback=self._meeting_audio_callback)
            self._set_state(STATE_MEETING)
        except RuntimeError as e:
            self._file_manager.end_session()
            self._signals.error.emit(str(e))

    def _meeting_audio_callback(self, audio_chunk: np.ndarray, source: str):
        if self._vad_chunker:
            self._vad_chunker.feed(audio_chunk)

    def _on_meeting_segment(self, segment: np.ndarray):
        """Callback del VAD: transcribir y agregar al .md en tiempo real."""
        # Incrementar contador de transcripciones pendientes
        with self._pending_lock:
            self._pending_segments += 1
        try:
            result = self._transcriber.transcribe(segment, language=self._config.language)
            if result and result.text.strip():
                timestamp = datetime.now().strftime("%H:%M:%S")
                self._file_manager.append_segment(None, timestamp, result.text)
                print(f"[Meeting] Segmento transcripto: {result.text[:60]}...")
        finally:
            # Siempre decrementar, incluso si hay error
            with self._pending_lock:
                self._pending_segments -= 1

    def _stop_meeting(self):
        self._recorder.stop()

        if self._vad_chunker:
            self._vad_chunker.flush()
            self._vad_chunker = None

        # Esperar que todas las transcripciones pendientes terminen (máx 60s)
        print("[Meeting] Esperando transcripciones pendientes...")
        self._set_state(STATE_TRANSCRIBING)

        def _wait_and_close():
            import time
            deadline = time.time() + 60
            while time.time() < deadline:
                with self._pending_lock:
                    if self._pending_segments <= 0:
                        break
                time.sleep(0.2)

            segment_count = self._file_manager.segment_count
            final_path = self._file_manager.end_session()
            msg = f"Reunión transcripta: {segment_count} segmento(s) — {final_path.name}"
            self._signals.notification.emit("WhisperMate", msg)
            self._signals.transcription_done.emit("meeting", str(final_path))
            self._signals.status_changed.emit(STATE_IDLE)

        threading.Thread(target=_wait_and_close, daemon=True).start()

    # ------------------------------------------------------------------
    # Watch folder
    # ------------------------------------------------------------------

    def _start_watcher(self):
        self._watcher = FolderWatcher(
            config=self._config,
            transcriber=self._transcriber,
            notify_callback=lambda t, m: self._signals.notification.emit(t, m),
        )
        self._watcher.start()

    def _stop_watcher(self):
        if self._watcher:
            self._watcher.stop()
            self._watcher = None

    # ------------------------------------------------------------------
    # Señales Qt (deben ejecutarse en el UI thread)
    # ------------------------------------------------------------------

    def _set_state(self, state: str):
        """Thread-safe: emite señal para cambiar estado en UI thread."""
        self._signals.status_changed.emit(state)

    def _on_status_changed(self, state: str):
        self._state = state
        self.setIcon(ICONS.get(state, ICONS[STATE_IDLE]))
        self._update_menu_state()

        tooltips = {
            STATE_IDLE: "WhisperMate — Listo",
            STATE_DICTATING: "WhisperMate — Grabando dictado...",
            STATE_MEETING: "WhisperMate — Grabando reunión...",
            STATE_TRANSCRIBING: "WhisperMate — Transcribiendo...",
        }
        self.setToolTip(tooltips.get(state, "WhisperMate"))

    def _show_notification(self, title: str, message: str):
        self.showMessage(title, message, QSystemTrayIcon.MessageIcon.Information, 4000)

    def _on_transcription_done(self, mode: str, filepath: str):
        from pathlib import Path
        filename = Path(filepath).name
        self._show_notification("WhisperMate", f"✅ Transcripción guardada: {filename}")

    def _on_error(self, message: str):
        self._show_notification("WhisperMate — Error", f"❌ {message}")
        if self._state != STATE_IDLE:
            self._on_status_changed(STATE_IDLE)

    # ------------------------------------------------------------------
    # Acciones del menú
    # ------------------------------------------------------------------

    def _open_output_folder(self):
        folder = str(self._config.output_folder)
        try:
            folder_path = self._config.output_folder
            folder_path.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(folder)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as e:
            self._signals.error.emit(f"No se pudo abrir la carpeta: {e}")

    def _open_settings(self):
        from src.ui.settings_window import SettingsWindow
        was_idle = self._state == STATE_IDLE
        dialog = SettingsWindow(self._config)
        if dialog.exec() and dialog.saved:
            # Recargar transcriber si cambió el modelo
            if self._transcriber.model_name != self._config.model:
                self._transcriber.change_model(self._config.model)

            # Re-registrar hotkeys
            self._unregister_hotkeys()
            self._register_hotkeys()

            # Watch folder
            if self._config.watch_folder_enabled and not (self._watcher and self._watcher.is_running):
                self._start_watcher()
            elif not self._config.watch_folder_enabled and self._watcher:
                self._stop_watcher()

    def _on_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._open_settings()

    # ------------------------------------------------------------------
    # Quit
    # ------------------------------------------------------------------

    def _quit(self):
        # Limpiar
        if self._recorder.is_recording:
            self._recorder.stop()
        if self._watcher:
            self._watcher.stop()
        self._unregister_hotkeys()
        QApplication.quit()
