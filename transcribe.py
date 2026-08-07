"""
Offline audio transcription using OpenAI Whisper (open-source).

Transcribes .m4a, .mp3, .wav, and .aac files to text, SRT, VTT, and JSON
formats using a locally-run Whisper model. No API calls, no internet
required after the model is downloaded.

Usage:
    python transcribe.py "C:\\Audio\\Theses.m4a" --model turbo
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Whisper is imported lazily inside run_transcription() so that --help and
# preflight checks (e.g. missing FFmpeg) still work if the heavy ML deps
# fail to import.

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".m4a", ".mp3", ".wav", ".aac")
AVAILABLE_MODELS: tuple[str, ...] = (
    "tiny", "base", "small", "medium", "turbo", "large-v3"
)
DEFAULT_MODEL: str = "turbo"

FFMPEG_INSTALL_HELP = """\
FFmpeg is required by Whisper to decode audio files but was not found on PATH.

Install it on Windows using one of the following methods:

  1) winget (recommended, Windows 10/11):
       winget install --id=Gyan.FFmpeg -e

  2) Chocolatey:
       choco install ffmpeg

  3) Manual install:
       - Download a build from https://www.gyan.dev/ffmpeg/builds/
       - Unzip somewhere permanent, e.g. C:\\ffmpeg
       - Add C:\\ffmpeg\\bin to your PATH (System Properties -> Environment Variables)
       - Open a NEW terminal and run: ffmpeg -version

After installing, open a new PowerShell/Terminal window and re-run this script.
"""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def configure_logging(verbose: bool) -> logging.Logger:
    """Configure the root logger for the app and return a module logger."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Whisper is chatty at INFO; keep it aligned with our verbosity.
    logging.getLogger("whisper").setLevel(level)
    return logging.getLogger("transcribe")


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
def check_ffmpeg_installed() -> bool:
    """Return True if the `ffmpeg` executable is discoverable on PATH."""
    if shutil.which("ffmpeg") is None:
        return False
    # Also verify it actually runs — a stale PATH entry can leave a broken exe.
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=10,
        )
        return True
    except (subprocess.SubprocessError, OSError):
        return False


def validate_audio_file(path: Path) -> Path:
    """Resolve and validate the input audio file. Raises on any problem."""
    if not path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{path.suffix}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    if path.stat().st_size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    return path.resolve()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@dataclass
class LoadedModel:
    """Wrapper carrying the Whisper model plus the device it was loaded on."""
    model: Any
    device: str
    name: str


def pick_device() -> str:
    """Return 'cuda' if a working CUDA GPU is available, else 'cpu'."""
    try:
        import torch  # local import so preflight errors are cleaner
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001 — torch may be missing or misconfigured
        pass
    return "cpu"


def load_whisper_model(model_name: str, logger: logging.Logger) -> LoadedModel:
    """Load (and download on first use) the requested Whisper model."""
    import whisper  # heavy import — kept out of module top-level

    device = pick_device()
    logger.info("Loading Whisper model '%s' on %s...", model_name, device.upper())
    logger.info("(First run will download the model — this can take several minutes.)")
    t0 = time.perf_counter()
    model = whisper.load_model(model_name, device=device)
    logger.info("Model ready in %.1fs.", time.perf_counter() - t0)
    return LoadedModel(model=model, device=device, name=model_name)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
def run_transcription(
    loaded: LoadedModel,
    audio_path: Path,
    language: str | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Run Whisper on the given audio file and return the raw result dict."""
    logger.info("Transcribing: %s", audio_path)
    if language:
        logger.info("Language: %s (user-specified)", language)
    else:
        logger.info("Language: auto-detect")

    # fp16 only makes sense on CUDA; force fp32 on CPU to avoid warnings.
    use_fp16 = loaded.device == "cuda"

    t0 = time.perf_counter()
    # `verbose=True` makes Whisper print each decoded segment as it goes,
    # which is our progress indicator for long files (>1h). Whisper streams
    # segments internally, so there is no separate progress bar to hook.
    result = loaded.model.transcribe(
        str(audio_path),
        language=language,           # None -> Whisper auto-detects
        task="transcribe",
        verbose=True,
        fp16=use_fp16,
        condition_on_previous_text=True,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Transcription finished in %.1fs.", elapsed)
    return result


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def _format_timestamp(seconds: float, *, comma: bool) -> str:
    """Format seconds as HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT)."""
    if seconds < 0:
        seconds = 0.0
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:  # rounding overflow
        millis = 0
        secs += 1
    sep = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def write_txt(segments: Iterable[dict[str, Any]], full_text: str, dest: Path) -> None:
    """Write plain-text transcript. Uses full text if available, else segments."""
    text = full_text.strip() if full_text else "\n".join(
        s["text"].strip() for s in segments
    )
    dest.write_text(text + "\n", encoding="utf-8")


def write_srt(segments: Iterable[dict[str, Any]], dest: Path) -> None:
    """Write SubRip (.srt) subtitles."""
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        start = _format_timestamp(seg["start"], comma=True)
        end = _format_timestamp(seg["end"], comma=True)
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"].strip())
        lines.append("")  # blank line between cues
    dest.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(segments: Iterable[dict[str, Any]], dest: Path) -> None:
    """Write WebVTT (.vtt) subtitles."""
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        start = _format_timestamp(seg["start"], comma=False)
        end = _format_timestamp(seg["end"], comma=False)
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"].strip())
        lines.append("")
    dest.write_text("\n".join(lines), encoding="utf-8")


def write_json(result: dict[str, Any], dest: Path) -> None:
    """Write the full Whisper result as pretty-printed JSON."""
    # Whisper returns numpy floats in some fields — json can't serialize those
    # directly, so we round-trip through a default=str fallback.
    dest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def save_all_outputs(
    result: dict[str, Any],
    output_dir: Path,
    logger: logging.Logger,
) -> dict[str, Path]:
    """Write txt/srt/vtt/json to output_dir. Returns map of format -> path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    segments = result.get("segments", []) or []
    full_text = result.get("text", "") or ""

    paths = {
        "txt":  output_dir / "transcript.txt",
        "srt":  output_dir / "transcript.srt",
        "vtt":  output_dir / "transcript.vtt",
        "json": output_dir / "transcript.json",
    }

    write_txt(segments, full_text, paths["txt"])
    write_srt(segments, paths["srt"])
    write_vtt(segments, paths["vtt"])
    write_json(result, paths["json"])

    for fmt, p in paths.items():
        logger.info("Wrote %s: %s", fmt.upper(), p)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Offline audio transcription with OpenAI Whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python transcribe.py \"C:\\Audio\\Theses.m4a\" --model turbo\n"
            "  python transcribe.py lecture.mp3 --model large-v3 --language en\n"
            "  python transcribe.py interview.wav --output_dir .\\out --verbose\n"
        ),
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="Path to the input audio file (.m4a, .mp3, .wav, .aac).",
    )
    parser.add_argument(
        "--model",
        choices=AVAILABLE_MODELS,
        default=DEFAULT_MODEL,
        help=f"Whisper model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="ISO language code (e.g. 'en', 'de', 'ar'). Omit to auto-detect.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write transcript files. Defaults to the audio's folder.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logger = configure_logging(args.verbose)

    # 1. Preflight: FFmpeg is a hard dependency of Whisper's audio loader.
    if not check_ffmpeg_installed():
        logger.error("FFmpeg not found.")
        print("\n" + FFMPEG_INSTALL_HELP, file=sys.stderr)
        return 2

    # 2. Validate input file.
    try:
        audio_path = validate_audio_file(args.audio)
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return 1

    # 3. Resolve output location — default to the audio file's directory.
    output_dir = (args.output_dir or audio_path.parent).resolve()

    # 4. Load model + transcribe. Wrap heavy ML section in a broad handler
    #    so users get a useful message instead of a traceback dump.
    try:
        loaded = load_whisper_model(args.model, logger)
        result = run_transcription(loaded, audio_path, args.language, logger)
    except ImportError as e:
        logger.error(
            "Missing dependency: %s. Run: pip install -r requirements.txt", e
        )
        return 3
    except FileNotFoundError as e:
        # Usually means FFmpeg vanished mid-run, or Whisper couldn't find audio.
        logger.error("File error during transcription: %s", e)
        return 4
    except RuntimeError as e:
        # Torch/CUDA OOM or model load failures come through as RuntimeError.
        logger.error("Runtime error: %s", e)
        if "out of memory" in str(e).lower():
            logger.error(
                "Try a smaller model (--model turbo) or run on CPU by "
                "uninstalling the CUDA build of torch."
            )
        return 5
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130

    # 5. Persist outputs.
    save_all_outputs(result, output_dir, logger)

    # 6. Echo the transcript to the console for immediate feedback.
    detected_lang = result.get("language", "?")
    print("\n" + "=" * 72)
    print(f"TRANSCRIPT  (language: {detected_lang})")
    print("=" * 72)
    print((result.get("text") or "").strip())
    print("=" * 72 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
