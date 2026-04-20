# block_sampling_visualize.py
# Tkinter (select yin_f0_after_rms.csv) -> Tkinter (select save location) ->
# Block Sampling ->
# Plot before (after RMS) ->
# Plot after (block-level) ->
# Save CSV (time_sec, f0_block_hz)

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
BLOCK_SEC = 0.5  # sesuai desain kamu


# =========================
# Tkinter pickers
# =========================
def pick_csv_file() -> str:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Pilih CSV RMS output (yin_f0_after_rms.csv)",
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
    plt.plot(times, f0, marker="o", markersize=2, linewidth=1)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


# =========================
# IO helper
# =========================
def load_rms_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Expect header: time_sec,f0_before_hz,f0_after_rms_hz
    Return times, f0_after_rms
    """
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)

    if data.ndim == 1 and data.size == 0:
        raise ValueError("CSV kosong atau format tidak valid.")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 3:
        raise ValueError("CSV harus punya minimal 3 kolom: time_sec,f0_before_hz,f0_after_rms_hz")

    times = data[:, 0].astype(float)
    f0_after_rms = data[:, 2].astype(float)
    f0_after_rms[(~np.isfinite(f0_after_rms)) | (f0_after_rms <= 0)] = np.nan
    return times, f0_after_rms


# =========================
# Block sampling core
# =========================
def block_sampling(times: np.ndarray, f0: np.ndarray, block_sec: float) -> tuple[np.ndarray, np.ndarray]:
    if len(times) == 0:
        return times, f0

    t0 = float(times[0])
    block_ids = np.floor((times - t0) / block_sec).astype(int)
    n_blocks = int(block_ids.max()) + 1 if block_ids.size else 0

    t_out = np.zeros(n_blocks, dtype=float)
    f_out = np.full(n_blocks, np.nan, dtype=float)

    for b in range(n_blocks):
        idx = np.where(block_ids == b)[0]
        if idx.size == 0:
            continue

        t_out[b] = float(np.mean(times[idx]))

        vals = f0[idx]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            f_out[b] = float(np.median(vals))
        else:
            f_out[b] = np.nan

    return t_out, f_out


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
    run_dir = out_dir / f"{stem}_block_viz"
    ensure_dir(run_dir)

    # Load
    times, f0_after_rms = load_rms_csv(csv_path)

    # Plot before block sampling
    plot_f0(
        times, f0_after_rms,
        "Pitch f0 (Hz) - BEFORE Block Sampling (After RMS)",
        run_dir / "01_before_block_sampling.png"
    )

    # Block sampling
    t_blk, f0_blk = block_sampling(times, f0_after_rms, BLOCK_SEC)

    # Plot after
    plot_f0(
        t_blk, f0_blk,
        f"Pitch f0 (Hz) - AFTER Block Sampling ({BLOCK_SEC}s)",
        run_dir / "02_after_block_sampling.png"
    )

    # Save CSV
    out_csv = run_dir / "f0_block.csv"
    np.savetxt(
        out_csv,
        np.column_stack([t_blk, f0_blk]),
        delimiter=",",
        header="time_sec,f0_block_hz",
        comments=""
    )

    messagebox.showinfo(
        "Selesai",
        "Block Sampling visualization selesai.\n"
        f"Output tersimpan di:\n{run_dir}\n\n"
        "- 01_before_block_sampling.png\n"
        "- 02_after_block_sampling.png\n"
        "- f0_block.csv"
    )


if __name__ == "__main__":
    main()
