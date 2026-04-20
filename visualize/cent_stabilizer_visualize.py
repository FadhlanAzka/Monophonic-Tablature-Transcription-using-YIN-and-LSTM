# cent_stabilizer_visualize.py
# Tkinter (select f0_block.csv) -> Tkinter (select save location) ->
# Cent Stabilizer ->
# Plot before ->
# Plot after ->
# Save CSV (time_sec, f0_block_hz, f0_cent_stable_hz)

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
CENT_SNAP_TOL = 35.0  # cents


# =========================
# Tkinter pickers
# =========================
def pick_csv_file() -> str:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Pilih CSV Block Sampling (f0_block.csv)",
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
# Plot helper
# =========================
def plot_f0(times: np.ndarray, f0: np.ndarray, title: str, save_path: Path) -> None:
    plt.figure(figsize=(12, 3.5))
    plt.plot(times, f0, marker="o", markersize=3, linewidth=1)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


# =========================
# Pitch helpers
# =========================
def hz_to_midi(hz: np.ndarray) -> np.ndarray:
    hz = np.asarray(hz, dtype=float)
    midi = np.full_like(hz, np.nan, dtype=float)
    mask = np.isfinite(hz) & (hz > 0)
    midi[mask] = 69.0 + 12.0 * np.log2(hz[mask] / 440.0)
    return midi


def midi_to_hz(midi: np.ndarray) -> np.ndarray:
    midi = np.asarray(midi, dtype=float)
    hz = np.full_like(midi, np.nan, dtype=float)
    mask = np.isfinite(midi)
    hz[mask] = 440.0 * (2.0 ** ((midi[mask] - 69.0) / 12.0))
    return hz


# =========================
# IO helper
# =========================
def load_block_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Expect header: time_sec,f0_block_hz
    """
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)

    if data.ndim == 1 and data.size == 0:
        raise ValueError("CSV kosong atau format tidak valid.")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 2:
        raise ValueError("CSV harus punya minimal 2 kolom: time_sec,f0_block_hz")

    times = data[:, 0].astype(float)
    f0_block = data[:, 1].astype(float)
    f0_block[(~np.isfinite(f0_block)) | (f0_block <= 0)] = np.nan
    return times, f0_block


# =========================
# Cent stabilizer core
# =========================
def cent_stabilize(f0_block: np.ndarray, cent_tol: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (f0_cent_stable, cents_dev)
    cents_dev = (midi - round(midi)) * 100
    """
    midi = hz_to_midi(f0_block)
    midi_round = np.round(midi)

    cents_dev = (midi - midi_round) * 100.0

    midi_out = midi.copy()
    snap_mask = np.isfinite(midi) & (np.abs(cents_dev) <= cent_tol)
    midi_out[snap_mask] = midi_round[snap_mask]

    f0_out = midi_to_hz(midi_out)
    return f0_out, cents_dev


# =========================
# Main
# =========================
def main():
    csv_path = pick_csv_file()
    if not csv_path:
        return

    out_dir = pick_output_dir()
    if not out_dir:
        return

    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    stem = Path(csv_path).stem
    run_dir = out_dir / f"{stem}_cent_viz"
    ensure_dir(run_dir)

    # Load
    times, f0_block = load_block_csv(csv_path)

    # Plot before
    plot_f0(
        times, f0_block,
        "Pitch f0 (Hz) - BEFORE Cent Stabilizer (Block-level)",
        run_dir / "01_before_cent_stabilizer.png"
    )

    # Cent stabilize
    f0_cent, cents_dev = cent_stabilize(f0_block, CENT_SNAP_TOL)

    # Plot after
    plot_f0(
        times, f0_cent,
        f"Pitch f0 (Hz) - AFTER Cent Stabilizer (tol={CENT_SNAP_TOL} cent)",
        run_dir / "02_after_cent_stabilizer.png"
    )

    # Save CSV
    out_csv = run_dir / "f0_cent_stable.csv"
    np.savetxt(
        out_csv,
        np.column_stack([times, f0_block, f0_cent, cents_dev]),
        delimiter=",",
        header="time_sec,f0_block_hz,f0_cent_stable_hz,cents_dev",
        comments=""
    )

    messagebox.showinfo(
        "Selesai",
        "Cent Stabilizer visualization selesai.\n"
        f"Output tersimpan di:\n{run_dir}\n\n"
        "- 01_before_cent_stabilizer.png\n"
        "- 02_after_cent_stabilizer.png\n"
        "- f0_cent_stable.csv"
    )


if __name__ == "__main__":
    main()
