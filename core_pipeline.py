from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

import settings
from audio_loader import load_audio
from analysis.beat_tracking import compute_beats
from analysis.onset_detection import compute_onsets
from tab_midi import csv_notes_to_midi_file
from utils.sanitize import ensure_finite

from app import (
    NOTE_OUTPUT_COLUMNS,
    _build_dataframe_from_stable,
    _collapse_to_sustains,
    _detect_device,
    _load_lstm_model,
    _load_token_index_df,
    _run_lstm_mapping,
    _save_basic_visualization,
    normalize_note_to_sharp,
    render_ascii_tab,
    run_pipeline14,
)

try:
    from viz.visualizer import save_visualization
except Exception:
    from visualizer import save_visualization


RenderMidiWavFn = Callable[[Path, Path], Optional[Path]]
ShowVisualizationFn = Callable[..., None]


@dataclass(frozen=True)
class PipelineConfig:
    wav_path: Path
    output_dir: Path
    token_index_csv: Path
    model_path: Path
    device: str = "auto"
    create_stem_subdir: bool = False
    show_player: bool = False
    make_midi: bool = False
    render_midi_wav: bool = False
    pitch_mode: Optional[str] = None
    pitch_confidence_threshold: Optional[float] = None
    use_tab_optimizer: Optional[bool] = None


@dataclass(frozen=True)
class PipelineArtifacts:
    out_dir: Path
    csv: Optional[Path] = None
    tab: Optional[Path] = None
    viz: Optional[Path] = None
    midi: Optional[Path] = None
    midi_wav: Optional[Path] = None

    def as_dict(self) -> Dict[str, Optional[Path]]:
        return {
            "out_dir": self.out_dir,
            "csv": self.csv,
            "tab": self.tab,
            "viz": self.viz,
            "midi": self.midi,
            "midi_wav": self.midi_wav,
        }


def _resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return _detect_device()
    if device_str == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("[WARN] CUDA diminta tetapi tidak tersedia, fallback ke CPU.")
                return "cpu"
        except Exception:
            print("[WARN] CUDA diminta tetapi torch tidak tersedia, fallback ke CPU.")
            return "cpu"
    return device_str


def _write_empty_csv(path: Path) -> Path:
    pd.DataFrame(columns=NOTE_OUTPUT_COLUMNS).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )
    return path


def _maybe_make_midi(
    out_csv: Path,
    out_dir: Path,
    wav_stem: str,
    enabled: bool,
    render_enabled: bool,
    render_midi_wav_fn: Optional[RenderMidiWavFn],
) -> tuple[Optional[Path], Optional[Path]]:
    if not enabled:
        return None, None

    out_midi: Optional[Path] = None
    out_midi_wav: Optional[Path] = None
    try:
        out_midi = out_dir / f"{wav_stem}_lstm_mapped.mid"
        csv_notes_to_midi_file(
            csv_path=out_csv,
            out_mid_path=out_midi,
            notes_per_second=2.0,
            midi_col="midi",
            rest_policy="keep",
        )
        print(f"[OK] MIDI disimpan: {out_midi}")
    except Exception as e:
        print(f"[WARN] Gagal membuat MIDI dari CSV: {e}")
        out_midi = None

    if render_enabled and out_midi is not None and render_midi_wav_fn is not None:
        out_midi_wav = render_midi_wav_fn(
            out_midi,
            out_dir / f"{wav_stem}_lstm_mapped_render.wav",
        )

    return out_midi, out_midi_wav


def run_transcription_pipeline(
    config: PipelineConfig,
    render_midi_wav_fn: Optional[RenderMidiWavFn] = None,
    show_visualization_fn: Optional[ShowVisualizationFn] = None,
) -> PipelineArtifacts:
    wav_path = Path(config.wav_path)
    out_dir = Path(config.output_dir)
    if config.create_stem_subdir:
        out_dir = out_dir / wav_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    token_index_df = _load_token_index_df(Path(config.token_index_csv))

    device = _resolve_device(config.device)
    print(f"[INFO] Device: {device}")

    model = _load_lstm_model(Path(config.model_path), device=device)
    print(f"[INFO] Model loaded from: {config.model_path}")

    print(f"[INFO] Memuat audio: {wav_path}")
    y, sr = load_audio(str(wav_path))
    y = ensure_finite(y)

    try:
        tempo, beat_times = compute_beats(y, sr)
    except Exception as e:
        print(f"[WARN] compute_beats gagal: {e}")
        tempo, beat_times = None, np.asarray([])

    try:
        onset_times = compute_onsets(y, sr)
    except Exception as e:
        print(f"[WARN] compute_onsets gagal: {e}")
        onset_times = np.asarray([])

    print("[INFO] Menjalankan Pipeline #14...")
    stable_notes, stable_f0, stable_midi, sample_times, pitch_diag = run_pipeline14(
        y,
        sr,
        pitch_mode=config.pitch_mode or settings.PITCH_DETECTION_MODE,
        confidence_threshold=(
            settings.PITCH_CONFIDENCE_THRESHOLD
            if config.pitch_confidence_threshold is None
            else float(config.pitch_confidence_threshold)
        ),
        return_diagnostics=True,
    )

    duration = len(y) / sr
    notes_vis = np.asarray(
        [normalize_note_to_sharp(n) if isinstance(n, str) else n for n in stable_notes],
        dtype=object,
    )

    if config.show_player and show_visualization_fn is not None:
        try:
            show_visualization_fn(
                audio_path=str(wav_path),
                duration=duration,
                sr=sr,
                tempo=tempo,
                beat_times=beat_times,
                onset_times=onset_times,
                times=sample_times,
                f0=stable_f0,
                notes=notes_vis,
            )
        except Exception as e:
            print(f"[WARN] show_visualization gagal: {e}")
    elif config.show_player:
        print("[INFO] show_player=True tetapi show_visualization_fn tidak tersedia.")

    wav_stem = wav_path.stem
    viz_path = out_dir / f"{wav_stem}_viz.png"
    try:
        save_visualization(out_dir)(
            audio_path=str(wav_path),
            duration=duration,
            sr=sr,
            tempo=tempo,
            beat_times=beat_times,
            onset_times=onset_times,
            times=sample_times,
            f0=stable_f0,
            notes=notes_vis,
        )
        print(f"[OK] Visualization disimpan ke folder: {out_dir}")
    except Exception as e:
        _save_basic_visualization(
            out_png=viz_path,
            audio_path=wav_path,
            sr=sr,
            times=sample_times,
            f0_for_plot=stable_f0,
            notes_for_plot=notes_vis,
            beat_times=np.asarray(beat_times) if beat_times is not None else np.asarray([]),
            onset_times=np.asarray(onset_times),
        )
        print(f"[WARN] Modul viz gagal, pakai fallback sederhana. Disimpan: {viz_path} ({e})")

    out_csv = out_dir / f"{wav_stem}_lstm_mapped.csv"
    df0 = _build_dataframe_from_stable(
        stable_f0,
        stable_notes,
        stable_midi,
        sample_times,
        pitch_confidence=pitch_diag.get("pitch_confidence"),
        pitch_mode=pitch_diag.get("pitch_mode", "yin"),
        octave_corrected=pitch_diag.get("octave_corrected"),
    )

    if len(df0) == 0:
        print(f"[SKIP] {wav_path.name}: no valid notes")
        _write_empty_csv(out_csv)
        out_midi, out_midi_wav = _maybe_make_midi(
            out_csv,
            out_dir,
            wav_stem,
            config.make_midi,
            config.render_midi_wav,
            render_midi_wav_fn,
        )
        return PipelineArtifacts(out_dir, out_csv, None, viz_path, out_midi, out_midi_wav)

    df = _collapse_to_sustains(df0, onset_times=onset_times)
    if len(df) == 0:
        print(f"[SKIP] {wav_path.name}: no sustained notes setelah RLE")
        _write_empty_csv(out_csv)
        out_midi, out_midi_wav = _maybe_make_midi(
            out_csv,
            out_dir,
            wav_stem,
            config.make_midi,
            config.render_midi_wav,
            render_midi_wav_fn,
        )
        return PipelineArtifacts(out_dir, out_csv, None, viz_path, out_midi, out_midi_wav)

    print("[INFO] Menjalankan LSTM Mapper...")
    df_out = _run_lstm_mapping(
        df_notes=df,
        model=model,
        token_index_df=token_index_df,
        device=device,
        use_pitch_mask=True,
        use_sequence_optimizer=(
            settings.TAB_OPTIMIZER_ENABLED
            if config.use_tab_optimizer is None
            else bool(config.use_tab_optimizer)
        ),
    )

    cols = [c for c in NOTE_OUTPUT_COLUMNS if c in df_out.columns]
    df_out[cols].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV LSTM-mapped disimpan: {out_csv}")

    out_midi, out_midi_wav = _maybe_make_midi(
        out_csv,
        out_dir,
        wav_stem,
        config.make_midi,
        config.render_midi_wav,
        render_midi_wav_fn,
    )

    out_tab: Optional[Path] = None
    if {"string", "fret"}.issubset(df_out.columns):
        try:
            tab_txt = render_ascii_tab(
                df_out["string"].tolist(),
                df_out["fret"].tolist(),
                width_pad=1,
            )
            out_tab = out_dir / f"{wav_stem}_lstm_tab.txt"
            out_tab.write_text(tab_txt, encoding="utf-8")
            print(f"[OK] ASCII tablature LSTM disimpan: {out_tab}")
        except Exception as e:
            print(f"[WARN] Gagal membuat ASCII tablature LSTM: {e}")
    else:
        print("[WARN] Kolom 'string'/'fret' tidak ada - TAB tidak dibuat.")

    print("[DONE] Inference LSTM Mapper selesai.")
    return PipelineArtifacts(out_dir, out_csv, out_tab, viz_path, out_midi, out_midi_wav)
