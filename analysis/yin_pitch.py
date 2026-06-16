# analysis/yin_pitch.py
import numpy as np
import librosa
from settings import FRAME_LENGTH, HOP_LENGTH, FMIN, FMAX

# >>> Added: sharp-only converter
from evaluation.music_theory import midi_to_note_sharp


def compute_yin(y, sr):
    """
    Compute YIN-based F0 estimation.
    Returns:
        f0: np.ndarray of fundamental frequency (Hz)
        times: np.ndarray of timestamps (sec)
        notes: np.ndarray of sharp-only note labels (e.g., F#3)
    """
    f0 = librosa.yin(
        y=y,
        fmin=FMIN,
        fmax=FMAX,
        sr=sr,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )

    times = librosa.frames_to_time(
        np.arange(len(f0)), sr=sr, hop_length=HOP_LENGTH
    )

    midi = librosa.hz_to_midi(f0)

    # >>> Force sharp-only labels with ASCII '#'
    notes = np.array(
        ["N/A" if np.isnan(m) else midi_to_note_sharp(m) for m in midi],
        dtype=object
    )

    return f0, times, notes


def compute_pyin(y, sr):
    """
    Compute pYIN-based F0 estimation.
    Returns:
        f0: np.ndarray of fundamental frequency (Hz)
        times: np.ndarray of timestamps (sec)
        notes: np.ndarray of sharp-only note labels
        voiced_flag: np.ndarray of voiced/unvoiced booleans
        voiced_prob: np.ndarray of voiced probabilities
    """
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y=y,
        fmin=FMIN,
        fmax=FMAX,
        sr=sr,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )

    times = librosa.frames_to_time(
        np.arange(len(f0)), sr=sr, hop_length=HOP_LENGTH
    )

    midi = librosa.hz_to_midi(f0)
    notes = np.array(
        ["N/A" if np.isnan(m) else midi_to_note_sharp(m) for m in midi],
        dtype=object,
    )

    return f0, times, notes, voiced_flag.astype(bool), voiced_prob.astype(float)


def compute_pitch_track(y, sr, mode: str = "yin"):
    """
    Compute pitch using the selected mode and return diagnostics.

    Default mode "yin" preserves the historical behavior. Confidence for YIN is
    a finite-pitch proxy; pYIN uses librosa's voiced probability.
    """
    mode_norm = (mode or "yin").strip().lower()
    if mode_norm == "pyin":
        f0, times, notes, voiced_flag, voiced_prob = compute_pyin(y, sr)
        confidence = np.nan_to_num(voiced_prob, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "mode": "pyin",
            "f0": f0,
            "times": times,
            "notes": notes,
            "voiced_flag": voiced_flag,
            "confidence": confidence,
        }

    f0, times, notes = compute_yin(y, sr)
    voiced_flag = np.isfinite(f0)
    confidence = voiced_flag.astype(float)
    return {
        "mode": "yin",
        "f0": f0,
        "times": times,
        "notes": notes,
        "voiced_flag": voiced_flag,
        "confidence": confidence,
    }
