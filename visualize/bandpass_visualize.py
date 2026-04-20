# bandpass_visualize.py
# Tkinter (select harmonic wav) -> Tkinter (select save location) ->
# Plot BEFORE (waveform + spectrogram) ->
# Bandpass ->
# Plot AFTER (waveform + spectrogram) ->
# Save filtered wav (+ optional freq response plot)

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, sosfreqz


# =========================
# Config (ubah kalau perlu)
# =========================
N_FFT = 2048
HOP_LENGTH = 512

BANDPASS_LOWCUT = 80.0
BANDPASS_HIGHCUT = 1200.0
BANDPASS_ORDER = 6

PLOT_FREQ_RESPONSE = True  # set False kalau tidak perlu


# =========================
# Tkinter pickers
# =========================
def pick_wav_file() -> str:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Pilih WAV Harmonic (hasil HPSS)",
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
    """
    Plot spectrogram (dB) dari audio.
    Return ref_value agar scaling dB konsisten before vs after.
    """
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


def plot_freq_response(sos: np.ndarray, sr: int, title: str, save_path: Path) -> None:
    # Frequency response of SOS filter
    w, h = sosfreqz(sos, worN=4096, fs=sr)
    mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))

    plt.figure(figsize=(12, 3.5))
    plt.plot(w, mag_db)
    plt.axvline(BANDPASS_LOWCUT, linestyle="--")
    plt.axvline(BANDPASS_HIGHCUT, linestyle="--")
    plt.ylim([-90, 5])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.grid(True, which="both", linestyle=":")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


# =========================
# Bandpass core
# =========================
def design_bandpass(sr: int, lowcut: float, highcut: float, order: int) -> np.ndarray:
    nyq = sr / 2.0

    # Safety clamp supaya tidak invalid
    low = max(1.0, float(lowcut))
    high = float(highcut)

    if high >= nyq:
        high = nyq * 0.99  # clamp sedikit di bawah Nyquist
    if low >= high:
        # fallback aman
        low = max(1.0, high * 0.5)

    sos = butter(order, [low, high], btype="bandpass", fs=sr, output="sos")
    return sos


def apply_bandpass(y: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return sosfiltfilt(sos, y)


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
    run_dir = out_dir / f"{stem}_bandpass_viz"
    ensure_dir(run_dir)

    # Load audio (diasumsikan harmonic/monophonic)
    y, sr = librosa.load(wav_path, sr=None, mono=True)

    # =========================
    # BEFORE
    # =========================
    plot_waveform(y, sr, "BEFORE Bandpass - Waveform (Input Harmonic)", run_dir / "01_before_waveform.png")
    ref_db = plot_spectrogram_db(
        y, sr,
        "BEFORE Bandpass - Spectrogram (dB) (Input Harmonic)",
        run_dir / "01_before_spectrogram.png",
        ref_value=None
    )

    # =========================
    # Bandpass
    # =========================
    sos = design_bandpass(sr, BANDPASS_LOWCUT, BANDPASS_HIGHCUT, BANDPASS_ORDER)

    if PLOT_FREQ_RESPONSE:
        plot_freq_response(
            sos, sr,
            f"Bandpass Frequency Response (order={BANDPASS_ORDER}, {BANDPASS_LOWCUT}-{BANDPASS_HIGHCUT} Hz)",
            run_dir / "02_filter_frequency_response.png"
        )

    y_bp = apply_bandpass(y, sos)

    # =========================
    # AFTER
    # =========================
    plot_waveform(
        y_bp, sr,
        f"AFTER Bandpass - Waveform ({BANDPASS_LOWCUT}-{BANDPASS_HIGHCUT} Hz)",
        run_dir / "03_after_waveform.png"
    )
    plot_spectrogram_db(
        y_bp, sr,
        f"AFTER Bandpass - Spectrogram (dB) ({BANDPASS_LOWCUT}-{BANDPASS_HIGHCUT} Hz)",
        run_dir / "03_after_spectrogram.png",
        ref_value=ref_db
    )

    # Save WAV output
    out_wav = run_dir / "after_bandpass.wav"
    sf.write(out_wav, y_bp, sr)

    messagebox.showinfo(
        "Selesai",
        "Bandpass visualization selesai.\n"
        f"Output tersimpan di:\n{run_dir}\n\n"
        "- PNG before/after (waveform + spectrogram)\n"
        "- (opsional) frequency response plot\n"
        "- after_bandpass.wav"
    )


if __name__ == "__main__":
    main()
