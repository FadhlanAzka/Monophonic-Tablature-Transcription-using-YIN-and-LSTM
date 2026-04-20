# yin_visualize.py
# Tkinter (select wav after bandpass) -> Tkinter (select save location) ->
# Plot BEFORE (waveform + spectrogram) ->
# YIN ->
# Plot AFTER YIN (f0 curve + spectrogram overlay) ->
# Save CSV (time_sec, f0_hz)

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display


# =========================
# Config (ubah kalau perlu)
# =========================
N_FFT = 2048
HOP_LENGTH = 512
FRAME_LENGTH = 2048  # untuk YIN
YIN_FMIN = 80.0
YIN_FMAX = 1200.0


# =========================
# Tkinter pickers
# =========================
def pick_wav_file() -> str:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Pilih WAV (hasil Bandpass)",
        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
    )
    root.destroy()
    return path


def pick_output_dir() -> str:
    root = tk.Tk()
    root.withdraw()
    out_dir = filedialog.askdirectory(title="Pilih folder output")
    root.destroy()
    return out_dir


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# =========================
# Plot helpers
# =========================
def plot_waveform(y: np.ndarray, sr: int, title: str, save_path: Path) -> None:
    plt.figure(figsize=(12, 3.5))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


def plot_spectrogram_db(
    y: np.ndarray,
    sr: int,
    title: str,
    save_path: Path,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    ref_value: float | None = None,
) -> float:
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(S)

    if ref_value is None:
        ref_value = float(np.max(mag)) if mag.size else 1.0
        if ref_value <= 0:
            ref_value = 1.0

    db = librosa.amplitude_to_db(mag, ref=ref_value)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()

    return ref_value


def plot_f0(times: np.ndarray, f0: np.ndarray, title: str, save_path: Path) -> None:
    plt.figure(figsize=(12, 3.5))
    plt.plot(times, f0, marker="o", markersize=2, linewidth=1)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


def plot_spec_with_f0_overlay(
    y: np.ndarray,
    sr: int,
    times: np.ndarray,
    f0: np.ndarray,
    title: str,
    save_path: Path,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    ref_value: float | None = None,
) -> None:
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(S)

    if ref_value is None:
        ref_value = float(np.max(mag)) if mag.size else 1.0
        if ref_value <= 0:
            ref_value = 1.0

    db = librosa.amplitude_to_db(mag, ref=ref_value)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")

    # Overlay f0 (abaikan yang invalid)
    mask = np.isfinite(f0) & (f0 > 0)
    plt.plot(times[mask], f0[mask], linewidth=2)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


# =========================
# YIN core
# =========================
def do_yin(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    f0 = librosa.yin(
        y=y,
        fmin=YIN_FMIN,
        fmax=YIN_FMAX,
        sr=sr,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )
    times = librosa.times_like(f0, sr=sr, hop_length=HOP_LENGTH)

    # Bersihkan nilai non-sense kecil / negatif (jaga-jaga)
    f0 = np.asarray(f0, dtype=float)
    f0[(~np.isfinite(f0)) | (f0 <= 0)] = np.nan

    return times, f0


# =========================
# Main
# =========================
def main():
    wav_path = pick_wav_file()
    if not wav_path:
        return

    out_dir = pick_output_dir()
    if not out_dir:
        return

    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    stem = Path(wav_path).stem
    run_dir = out_dir / f"{stem}_yin_viz"
    ensure_dir(run_dir)

    # Load audio
    y, sr = librosa.load(wav_path, sr=None, mono=True)

    # BEFORE: waveform + spectrogram
    plot_waveform(y, sr, "YIN Input - Waveform (After Bandpass)", run_dir / "01_input_waveform.png")
    ref_db = plot_spectrogram_db(
        y, sr,
        "YIN Input - Spectrogram (dB) (After Bandpass)",
        run_dir / "01_input_spectrogram.png",
        ref_value=None
    )

    # YIN
    times, f0 = do_yin(y, sr)

    # AFTER: f0 curve
    plot_f0(
        times, f0,
        f"YIN Output - Pitch f0 (Hz) (fmin={YIN_FMIN}, fmax={YIN_FMAX})",
        run_dir / "02_yin_f0_curve.png"
    )

    # AFTER: spectrogram + overlay f0
    plot_spec_with_f0_overlay(
        y, sr, times, f0,
        "YIN Output - Spectrogram (dB) with f0 overlay",
        run_dir / "03_yin_overlay_on_spectrogram.png",
        ref_value=ref_db
    )

    # Save CSV
    csv_path = run_dir / "yin_f0.csv"
    np.savetxt(
        csv_path,
        np.column_stack([times, f0]),
        delimiter=",",
        header="time_sec,f0_hz",
        comments=""
    )

    messagebox.showinfo(
        "Selesai",
        "YIN visualization selesai.\n"
        f"Output tersimpan di:\n{run_dir}\n\n"
        "- PNG waveform + spectrogram\n"
        "- PNG f0 curve + overlay\n"
        "- yin_f0.csv"
    )


if __name__ == "__main__":
    main()
