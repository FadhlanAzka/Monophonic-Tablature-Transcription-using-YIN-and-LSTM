# rms_noise_visualize.py
# Tkinter (select wav after bandpass) ->
# Tkinter (select yin_f0.csv) ->
# Tkinter (select save location) ->
# RMS Noise Gate ->
# Plot RMS(dB) + threshold ->
# Plot f0 before gate ->
# Plot f0 after gate ->
# Save CSV (time_sec, f0_before_hz, f0_after_rms_hz)

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display


# =========================
# Config (samakan dengan YIN-only)
# =========================
HOP_LENGTH = 512
FRAME_LENGTH = 2048

# Gate: buang frame yg RMS-nya < (maxRMS - RMS_GATE_DB)
RMS_GATE_DB = 35.0


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


def pick_csv_file() -> str:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Pilih CSV YIN (yin_f0.csv)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
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
def plot_rms_db(times: np.ndarray, rms_db: np.ndarray, thr_db: float, title: str, save_path: Path) -> None:
    plt.figure(figsize=(12, 3.5))
    plt.plot(times, rms_db, linewidth=1)
    plt.axhline(thr_db, linestyle="--")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("RMS (dB, ref=max)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


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


# =========================
# IO helpers
# =========================
def load_yin_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Expect header: time_sec,f0_hz
    Return times, f0
    """
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    if data.ndim == 1 and data.size == 0:
        raise ValueError("CSV kosong atau format tidak valid.")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 2:
        raise ValueError("CSV harus punya minimal 2 kolom: time_sec,f0_hz")

    times = data[:, 0].astype(float)
    f0 = data[:, 1].astype(float)
    f0[(~np.isfinite(f0)) | (f0 <= 0)] = np.nan
    return times, f0


# =========================
# RMS Noise Gate core
# =========================
def compute_rms_db(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
    RMS per frame -> dB relative max.
    Return (times_rms, rms_db)
    """
    rms = librosa.feature.rms(
        y=y,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        center=True
    )[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    times_rms = librosa.times_like(rms, sr=sr, hop_length=HOP_LENGTH)
    return times_rms, rms_db


def gate_f0_with_rms(
    times_f0: np.ndarray,
    f0: np.ndarray,
    times_rms: np.ndarray,
    rms_db: np.ndarray,
    gate_db: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Keep frame when rms_db >= -gate_db (ref=max).
    Jika grid waktu beda, rms_db diinterpolasi ke times_f0.
    Return (rms_db_on_f0_grid, f0_gated, thr_db)
    """
    thr_db = -float(gate_db)

    # Interpolate RMS(dB) ke grid time f0 (lebih aman kalau panjang beda)
    if len(times_rms) < 2:
        rms_on_f0 = np.full_like(times_f0, np.nan, dtype=float)
    else:
        rms_on_f0 = np.interp(times_f0, times_rms, rms_db, left=np.nan, right=np.nan)

    keep = np.isfinite(rms_on_f0) & (rms_on_f0 >= thr_db)

    f0_gated = f0.copy()
    f0_gated[~keep] = np.nan

    return rms_on_f0, f0_gated, thr_db


# =========================
# Main
# =========================
def main():
    wav_path = pick_wav_file()
    if not wav_path:
        return

    csv_path = pick_csv_file()
    if not csv_path:
        return

    out_dir = pick_output_dir()
    if not out_dir:
        return

    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    stem_wav = Path(wav_path).stem
    stem_csv = Path(csv_path).stem
    run_dir = out_dir / f"{stem_wav}_rms_viz"
    ensure_dir(run_dir)

    # Load inputs
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    times_f0, f0 = load_yin_csv(csv_path)

    # Compute RMS(dB)
    times_rms, rms_db = compute_rms_db(y, sr)

    # Gate f0
    rms_on_f0, f0_gated, thr_db = gate_f0_with_rms(
        times_f0, f0,
        times_rms, rms_db,
        gate_db=RMS_GATE_DB
    )

    # Plots
    plot_rms_db(
        times_f0, rms_on_f0, thr_db,
        f"RMS (dB, ref=max) on f0 grid + threshold ({thr_db:.1f} dB)",
        run_dir / "01_rms_db_threshold.png"
    )

    plot_f0(
        times_f0, f0,
        "Pitch f0 (Hz) - BEFORE RMS Noise Gate",
        run_dir / "02_f0_before_rms.png"
    )

    plot_f0(
        times_f0, f0_gated,
        f"Pitch f0 (Hz) - AFTER RMS Noise Gate (remove < {thr_db:.1f} dB)",
        run_dir / "03_f0_after_rms.png"
    )

    # Save CSV
    out_csv = run_dir / "yin_f0_after_rms.csv"
    np.savetxt(
        out_csv,
        np.column_stack([times_f0, f0, f0_gated]),
        delimiter=",",
        header="time_sec,f0_before_hz,f0_after_rms_hz",
        comments=""
    )

    messagebox.showinfo(
        "Selesai",
        "RMS Noise Gate visualization selesai.\n"
        f"Output tersimpan di:\n{run_dir}\n\n"
        "- 01_rms_db_threshold.png\n"
        "- 02_f0_before_rms.png\n"
        "- 03_f0_after_rms.png\n"
        "- yin_f0_after_rms.csv"
    )


if __name__ == "__main__":
    main()
