# WhisperMate 🎙️

App de system tray minimalista para transcripción de audio local. Sin nube, sin suscripciones. Máximo poder, mínima UI.

## ¿Qué hace?

- **Modo Dictado** (`Ctrl+Shift+R`): graba tu micrófono y genera un `.md` al terminar
- **Modo Reunión** (`Ctrl+Shift+M`): graba mic + audio del sistema en tiempo real con detección de voz (VAD)
- **Watch Folder**: deja archivos de audio en una carpeta y se transcriben automáticamente
- Todo el procesamiento es **100% local** con [faster-whisper](https://github.com/SYSTRAN/faster-whisper)

---

## Requisitos

- **Python 3.10+**
- **ffmpeg** (para transcribir MP3, M4A, MP4, WebM desde watch folder)
  - Windows: `winget install ffmpeg` o descargar de https://ffmpeg.org
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

---

## Instalación

```bash
# 1. Clonar / descargar el proyecto
cd whispermate

# 2. (Recomendado) Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Copiar config
cp config.json.example config.json
# Editar config.json a gusto (o usar la UI de settings)

# 5. Ejecutar
python main.py
```

---

## Modelos disponibles

| Modelo | Tamaño | Velocidad | Precisión |
|--------|--------|-----------|-----------|
| `tiny` | ~75 MB | ⚡⚡⚡⚡ | ★★☆☆☆ |
| `base` | ~145 MB | ⚡⚡⚡ | ★★★☆☆ |
| `small` | ~465 MB | ⚡⚡ | ★★★★☆ |
| `medium` | ~1.5 GB | ⚡ | ★★★★☆ |
| `large-v2` | ~3 GB | 🐢 | ★★★★★ |
| `large-v3` | ~3 GB | 🐢 | ★★★★★ |

El modelo se descarga automáticamente la primera vez que se usa (~HuggingFace Hub).

**Recomendación:** `small` para uso diario. `large-v3` si tenés GPU y priorizás precisión.

---

## Uso

### System Tray

Al ejecutar `python main.py`, aparece un ícono en la barra del sistema:

- **Gris** → Listo
- **Verde** → Grabando dictado
- **Rojo** → Grabando reunión
- **Naranja** → Transcribiendo

Clic derecho → menú con todas las opciones.
Doble clic → abre configuración.

### Shortcuts

| Acción | Default |
|--------|---------|
| Toggle dictado | `Ctrl+Shift+R` |
| Toggle reunión | `Ctrl+Shift+M` |

Configurables desde la ventana de settings.

### Watch Folder

1. Activar en configuración → "Activar watch folder"
2. Configurar la carpeta a observar
3. Copiar/mover archivos de audio ahí → se transcriben automáticamente

Formatos soportados: `.mp3`, `.wav`, `.ogg`, `.m4a`, `.mp4`, `.webm`, `.flac`

---

## Archivos de salida

Los `.md` se guardan en la carpeta configurada (default: `~/Transcripciones`) con el formato:

```
2026-03-12_14-30-00_dictation.md
2026-03-12_15-00-00_meeting.md
```

Ejemplo de contenido:

```markdown
# Transcripción — 12 de marzo de 2026, 14:30
**Modo:** Dictado
**Modelo:** small

---

**[14:30:15]:** texto transcripto aquí...

---

_Sesión finalizada: 14:30:45 · Duración: 0m 30s · Segmentos: 1_
```

---

## Configuración (`config.json`)

```json
{
  "model": "small",
  "language": "es",
  "output_folder": "~/Transcripciones",
  "watch_folder": "~/Transcripciones/inbox",
  "watch_folder_enabled": false,
  "shortcuts": {
    "dictation": "ctrl+shift+r",
    "meeting": "ctrl+shift+m"
  },
  "vad_threshold": 0.5,
  "vad_min_silence_ms": 800,
  "vad_chunk_padding_ms": 200
}
```

### Parámetros VAD

- `vad_threshold`: 0.0-1.0. Más alto = menos sensible al ruido de fondo (default: 0.5)
- `vad_min_silence_ms`: ms de silencio para cortar un segmento (default: 800ms)
- `vad_chunk_padding_ms`: padding al inicio/fin de cada segmento (default: 200ms)

---

## Notas por plataforma

### Windows

- El loopback del sistema (captura de audio que suena por los parlantes) funciona nativamente con `soundcard`.
- Los hotkeys globales funcionan sin permisos especiales.
- Si PyQt6 no muestra el ícono en el tray, instalar: `pip install PyQt6-Qt6 PyQt6-sip`.

### macOS

- **Loopback**: macOS bloquea la captura de audio del sistema por defecto. Opciones:
  - Instalar [BlackHole](https://github.com/ExistentialAudio/BlackHole) (driver virtual gratuito)
  - Instalar [Loopback](https://rogueamoeba.com/loopback/) (pago)
  - Sin loopback, el modo reunión solo captura el micrófono.
- Los hotkeys globales pueden requerir permisos de **Accesibilidad** en Preferencias del Sistema.
- Si el ícono no aparece: puede necesitar que la app sea un bundle `.app` en versiones recientes.

### Linux

- **Loopback**: depende del sistema de audio:
  - **PulseAudio**: crear un sink virtual: `pactl load-module module-null-sink sink_name=whispermate`
  - **PipeWire**: suele funcionar directamente con `soundcard`.
- **Hotkeys globales**: la librería `keyboard` requiere acceso a `/dev/input`. Opciones:
  - Ejecutar con `sudo` (no recomendado para producción)
  - Agregar tu usuario al grupo `input`: `sudo usermod -a -G input $USER` (requiere logout)
  - Usar `evdev` como alternativa
- **Tray**: requiere un DE compatible (GNOME puede necesitar extensión `AppIndicator`).

---

## Troubleshooting

**"No se pudo cargar silero-vad"**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install silero-vad
```

**"No se pudo iniciar el micrófono"**
- Verificar que el micrófono no esté en uso por otra aplicación
- Revisar permisos del sistema

**Transcripción lenta**
- Usar un modelo más pequeño (`tiny` o `base`)
- Si tenés GPU NVIDIA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

**El ícono no aparece en GNOME**
```bash
sudo apt install gnome-shell-extension-appindicator
# Activar la extensión en GNOME Extensions
```

---

## Licencia

MIT — hacé lo que quieras con esto.
