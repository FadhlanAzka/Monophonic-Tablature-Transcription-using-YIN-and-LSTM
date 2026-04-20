# hpss_visualize.py
# Tkinter (select wav) -> Tkinter (select save location) ->
# Plot BEFORE (waveform + spectrogram) ->
# HPSS ->
# Plot AFTER Harmonic (waveform + spectrogram) ->
# Plot AFTER Percussive (waveform + spectrogram) ->
# Save WAV outputs

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display
import soundfile as sf


# =========================
# Config
# =========================
N_FFT = 2048
HOP_LENGTH = 512


# =========================
# Tkinter pickers
# =========================
def pick_wav_file() -> str:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Pilih file WAV",
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
    S_mag: np.ndarray,
    sr: int,
    title: str,
    save_path: Path,
    hop_length: int = HOP_LENGTH,
    ref_value: float | None = None,
) -> float:
    """
    Plot spectrogram (dB) dari magnitude STFT.
    Return ref_value agar scaling dB konsisten antar plot (before vs after).
    """
    if ref_value is None:
        ref_value = float(np.max(S_mag)) if S_mag.size else 1.0
        if ref_value <= 0:
            ref_value = 1.0

    S_db = librosa.amplitude_to_db(S_mag, ref=ref_value)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()

    return ref_value


# =========================
# HPSS core
# =========================
def do_hpss(y: np.ndarray, sr: int):
    """
    Return:
      - y_harm, y_perc: audio results
      - S, H, P: STFT complex (original, harmonic, percussive)
    """
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    H, P = librosa.decompose.hpss(S)  # complex-valued separation
    y_harm = librosa.istft(H, hop_length=HOP_LENGTH, length=len(y))
    y_perc = librosa.istft(P, hop_length=HOP_LENGTH, length=len(y))
    return y_harm, y_perc, S, H, P


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
    run_dir = out_dir / f"{stem}_hpss_viz"
    ensure_dir(run_dir)

    # Load audio
    y, sr = librosa.load(wav_path, sr=None, mono=True)

    # =========================
    # BEFORE
    # =========================
    plot_waveform(y, sr, "BEFORE HPSS - Waveform (Original)", run_dir / "01_before_waveform.png")

    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_mag = np.abs(S)
    ref_db = plot_spectrogram_db(
        S_mag, sr,
        "BEFORE HPSS - Spectrogram (dB) (Original)",
        run_dir / "01_before_spectrogram.png",
        ref_value=None  # ambil ref dari original agar konsisten
    )

    # =========================
    # HPSS
    # =========================
    y_harm, y_perc, S, H, P = do_hpss(y, sr)

    # =========================
    # AFTER - Harmonic
    # =========================
    plot_waveform(y_harm, sr, "AFTER HPSS - Waveform (Harmonic)", run_dir / "02_after_harm_waveform.png")
    plot_spectrogram_db(
        np.abs(H), sr,
        "AFTER HPSS - Spectrogram (dB) (Harmonic)",
        run_dir / "02_after_harm_spectrogram.png",
        ref_value=ref_db
    )
    sf.write(run_dir / "02_after_harm.wav", y_harm, sr)

    # =========================
    # AFTER - Percussive
    # =========================
    plot_waveform(y_perc, sr, "AFTER HPSS - Waveform (Percussive)", run_dir / "03_after_perc_waveform.png")
    plot_spectrogram_db(
        np.abs(P), sr,
        "AFTER HPSS - Spectrogram (dB) (Percussive)",
        run_dir / "03_after_perc_spectrogram.png",
        ref_value=ref_db
    )
    sf.write(run_dir / "03_after_perc.wav", y_perc, sr)

    messagebox.showinfo(
        "Selesai",
        "HPSS visualization selesai.\n"
        f"Output tersimpan di:\n{run_dir}\n\n"
        "- PNG waveform + spectrogram (before/harmonic/percussive)\n"
        "- WAV hasil harmonic & percussive"
    )


if __name__ == "__main__":
    main()
