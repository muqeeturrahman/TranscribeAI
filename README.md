# Offline Whisper Transcriber

Transcribe `.m4a`, `.mp3`, `.wav`, and `.aac` audio to text, SRT, VTT, and
JSON — entirely offline, using the open-source
[OpenAI Whisper](https://github.com/openai/whisper) model. No API key, no
cloud calls, no per-minute cost.

Works on Windows 10/11 with Python 3.11+.

---

## Features

- Runs Whisper **locally** — the audio never leaves your machine.
- Choice of three models: `turbo` (fast), `medium` (balanced), `large-v3` (best).
- **Automatic FFmpeg detection** with clear install guidance if it's missing.
- Automatic **model download** on first run (cached under `~/.cache/whisper`).
- **Auto language detection**, or force a language with `--language`.
- Handles **long recordings** (multi-hour lectures, interviews).
- Uses your **GPU (CUDA)** automatically if available, otherwise CPU.
- Emits four output formats: `transcript.txt`, `transcript.srt`,
  `transcript.vtt`, `transcript.json`.

---

## Installation

### 1. Install Python 3.11+

Download from <https://www.python.org/downloads/windows/>. During install,
tick **"Add Python to PATH"**.

Verify:

```powershell
python --version
```

### 2. Install FFmpeg (required)

Whisper uses FFmpeg to decode audio. Pick **one**:

```powershell
# Windows Package Manager (recommended)
winget install --id=Gyan.FFmpeg -e
```

```powershell
# Chocolatey
choco install ffmpeg
```

Or download a build from <https://www.gyan.dev/ffmpeg/builds/>, unzip to
`C:\ffmpeg`, and add `C:\ffmpeg\bin` to your PATH.

Verify (open a **new** terminal):

```powershell
ffmpeg -version
```

### 3. Install Python dependencies

```powershell
# From the project folder
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU users:** the default `torch` from PyPI is CPU-only. For NVIDIA GPUs
> install the matching CUDA build from <https://pytorch.org/get-started/locally/>
> **before** `pip install -r requirements.txt`. Example (CUDA 12.1):
>
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Usage

Basic:

```powershell
python transcribe.py "C:\Audio\Theses.m4a" --model turbo
```

Force a language:

```powershell
python transcribe.py lecture.mp3 --model large-v3 --language en
```

Custom output directory + verbose logs:

```powershell
python transcribe.py interview.wav --output_dir .\out --verbose
```

### Arguments

| Argument       | Required | Default              | Description                                                  |
|----------------|----------|----------------------|--------------------------------------------------------------|
| `audio`        | yes      | —                    | Path to `.m4a`/`.mp3`/`.wav`/`.aac` file.                    |
| `--model`      | no       | `turbo`              | `turbo`, `medium`, or `large-v3`.                            |
| `--language`   | no       | auto-detect          | ISO code (`en`, `de`, `ar`, `fr`, ...).                      |
| `--output_dir` | no       | audio file's folder  | Where transcript files are written.                          |
| `--verbose`    | no       | off                  | DEBUG-level logging.                                          |

### Outputs

Written to `--output_dir` (or next to the audio file by default):

- `transcript.txt`  — plain text
- `transcript.srt`  — SubRip subtitles (for video players)
- `transcript.vtt`  — WebVTT subtitles (for HTML5 `<video>`)
- `transcript.json` — full Whisper result: segments, timings, language, tokens

The transcript is also printed to the console when finished.

---

## Model comparison

| Model      | Size on disk | Speed (CPU) | Accuracy    | VRAM (GPU) |
|------------|-------------:|-------------|-------------|-----------:|
| `turbo`    | ~1.5 GB      | Fastest     | Very good   | ~6 GB      |
| `medium`   | ~1.5 GB      | Moderate    | Better      | ~5 GB      |
| `large-v3` | ~3 GB        | Slowest     | Best        | ~10 GB     |

Models are downloaded once and cached under `%USERPROFILE%\.cache\whisper\`.

---

## Troubleshooting

**`FFmpeg not found`**
Install FFmpeg (see above), then open a **new** terminal so the updated PATH
is picked up.

**`CUDA out of memory`**
Use a smaller model: `--model turbo`. Or force CPU by uninstalling the CUDA
build of Torch (`pip uninstall torch` then reinstall the CPU wheel).

**Model download stalls / fails**
The first run downloads the model from OpenAI's CDN. On slow/flaky networks
this can time out. Re-run — downloads resume from cache.

**`UserWarning: FP16 is not supported on CPU; using FP32 instead`**
Harmless — the script already forces FP32 on CPU. Warning comes from a
transitive dep and can be ignored.

**Odd characters in the console (Windows)**
Windows terminals sometimes mangle UTF-8. The output files are always UTF-8;
if the console looks garbled, open `transcript.txt` in Notepad/VS Code.

**Extremely long files (2h+)**
Supported — Whisper streams internally in 30-second windows. Expect roughly
real-time transcription on CPU with `turbo`; much faster on GPU.

**Wrong language detected**
Pass `--language` explicitly (e.g. `--language de`).

---

## Building a standalone Windows `.exe` (PyInstaller)

Users without Python installed can run a bundled `.exe`. Note the caveats
below — this bundle is large because Whisper's ML deps are large.

### 1. Install PyInstaller

```powershell
pip install pyinstaller
```

### 2. Locate Whisper's `assets` folder

Whisper ships a `mel_filters.npz` file it loads at runtime. PyInstaller needs
to be told about it:

```powershell
python -c "import whisper, os; print(os.path.join(os.path.dirname(whisper.__file__), 'assets'))"
```

Copy that path — you'll paste it into the command below.

### 3. Build

```powershell
pyinstaller `
  --onefile `
  --name transcribe `
  --collect-all whisper `
  --collect-all torch `
  --collect-all tiktoken `
  --add-data "<PASTE_ASSETS_PATH_HERE>;whisper/assets" `
  transcribe.py
```

The single-file executable will be at `dist\transcribe.exe`. Ship it
alongside the FFmpeg install instructions — **PyInstaller does not bundle
FFmpeg**, since FFmpeg is a separate `ffmpeg.exe` binary invoked via
subprocess.

### 4. Ship FFmpeg with your app (optional)

To make the bundle fully self-contained:

1. Download `ffmpeg.exe` from <https://www.gyan.dev/ffmpeg/builds/>.
2. Place it next to `transcribe.exe`.
3. Add this line near the top of `transcribe.py` before building:

   ```python
   os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
   ```

   That prepends the folder holding the `.exe` (and thus `ffmpeg.exe`) to PATH
   so Whisper finds it.

### 5. Run

```powershell
.\dist\transcribe.exe "C:\Audio\Theses.m4a" --model turbo
```

### Caveats

- **Bundle size:** ~2–3 GB with Torch. Consider `--onedir` instead of
  `--onefile` for faster startup and smaller download-on-demand chunks.
- **First-run download:** the Whisper model itself is not bundled and will
  download on first use. To bundle it, pre-download to
  `%USERPROFILE%\.cache\whisper\`, then add that folder via `--add-data`.
- **Antivirus:** `--onefile` binaries are occasionally flagged as false
  positives. `--onedir` avoids this.
- **GPU support:** if you built with a CUDA Torch wheel, the `.exe` will only
  work on machines with a matching CUDA runtime. For broad distribution,
  build against the CPU-only Torch wheel.

---

## License

The Whisper model and code are MIT-licensed by OpenAI. This wrapper is
provided as-is for educational and personal use.
