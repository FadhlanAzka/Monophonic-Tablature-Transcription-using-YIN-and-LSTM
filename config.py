from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        print(f"[WARN] {name} harus berupa integer. Menggunakan default: {default}")
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def find_fluidsynth_bin() -> Optional[Path]:
    env_value = os.getenv("YINTAB_FLUIDSYNTH_BIN")
    if env_value:
        return Path(env_value).expanduser()

    found = shutil.which("fluidsynth")
    if found:
        return Path(found)

    legacy_windows_path = Path(
        r"C:\Program Files\fluidsynth-v2.5.1-win10-x64-cpp11\bin\fluidsynth.exe"
    )
    if legacy_windows_path.exists():
        return legacy_windows_path

    return None


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path = BASE_DIR
    static_dir: Path = _env_path("YINTAB_STATIC_DIR", BASE_DIR / "static")
    results_dir: Path = _env_path("YINTAB_RESULTS_DIR", BASE_DIR / "static" / "results")
    token_index_csv: Path = _env_path(
        "YINTAB_TOKEN_INDEX_CSV",
        BASE_DIR / "tokens" / "token index v2.csv",
    )
    model_path: Path = _env_path("YINTAB_MODEL_PATH", BASE_DIR / "models" / "optuna.jit")
    default_device: str = os.getenv("YINTAB_DEVICE", "auto")
    soundfont_path: Path = _env_path(
        "YINTAB_SOUNDFONT_PATH",
        BASE_DIR / "assets" / "FluidR3_GM.sf2",
    )
    fluidsynth_bin: Optional[Path] = find_fluidsynth_bin()
    render_sample_rate: int = _env_int("YINTAB_RENDER_SAMPLE_RATE", 44100)
    flask_host: str = os.getenv("YINTAB_FLASK_HOST", "0.0.0.0")
    flask_port: int = _env_int("YINTAB_FLASK_PORT", 5000)
    flask_debug: bool = _env_bool("YINTAB_FLASK_DEBUG", True)


def load_config() -> AppConfig:
    return AppConfig()


def validate_startup_config(cfg: AppConfig) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not cfg.token_index_csv.exists():
        errors.append(f"Token index CSV tidak ditemukan: {cfg.token_index_csv}")
    if not cfg.model_path.exists():
        errors.append(f"Model file tidak ditemukan: {cfg.model_path}")
    if not cfg.soundfont_path.exists():
        warnings.append(
            f"SoundFont tidak ditemukan: {cfg.soundfont_path}; render MIDI ke WAV akan dilewati."
        )
    if cfg.fluidsynth_bin is None:
        warnings.append(
            "FluidSynth tidak ditemukan. Set YINTAB_FLUIDSYNTH_BIN atau tambahkan fluidsynth ke PATH "
            "untuk mengaktifkan render MIDI ke WAV."
        )
    elif not cfg.fluidsynth_bin.exists():
        warnings.append(
            f"FluidSynth tidak ditemukan di {cfg.fluidsynth_bin}; render MIDI ke WAV akan dilewati."
        )

    return errors, warnings
