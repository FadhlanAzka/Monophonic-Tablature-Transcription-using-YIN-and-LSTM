from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _safe_numeric_array(values) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def _token_lookup(token_index_df: pd.DataFrame, token_ids: np.ndarray, column: str) -> np.ndarray:
    if column not in token_index_df.columns:
        return np.full(len(token_ids), np.nan)

    df = token_index_df.copy()
    if "token_idx" in df.columns:
        df = df.set_index("token_idx", drop=False)

    mapped = pd.Series(token_ids).map(df[column])
    return _safe_numeric_array(mapped)


def _levenshtein_distance(a: Iterable[int], b: Iterable[int]) -> int:
    a = list(a)
    b = list(b)
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        curr = [i]
        for j, y in enumerate(b, start=1):
            cost = 0 if x == y else 1
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = curr
    return prev[-1]


def _accuracy_where_valid(true_vals: np.ndarray, pred_vals: np.ndarray) -> Optional[float]:
    valid = np.isfinite(true_vals) & np.isfinite(pred_vals)
    if not valid.any():
        return None
    return float(np.mean(true_vals[valid] == pred_vals[valid]))


def _mean_abs_error_where_valid(true_vals: np.ndarray, pred_vals: np.ndarray) -> Optional[float]:
    valid = np.isfinite(true_vals) & np.isfinite(pred_vals)
    if not valid.any():
        return None
    return float(np.mean(np.abs(true_vals[valid] - pred_vals[valid])))


def compute_token_musical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    token_index_df: pd.DataFrame,
    onset_tolerance_sec: float = 0.05,
) -> Dict[str, Optional[float]]:
    """
    Compute music-facing metrics from token predictions.

    These metrics evaluate the string/fret/note meaning of predicted token_idx
    values. They do not replace event-level onset/offset evaluation; the current
    model-evaluation scripts operate on token windows, so event timing metrics
    are reported as unavailable.
    """
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)

    true_midi = _token_lookup(token_index_df, y_true, "midi")
    pred_midi = _token_lookup(token_index_df, y_pred, "midi")
    true_string = _token_lookup(token_index_df, y_true, "string")
    pred_string = _token_lookup(token_index_df, y_pred, "string")
    true_fret = _token_lookup(token_index_df, y_true, "fret")
    pred_fret = _token_lookup(token_index_df, y_pred, "fret")

    valid_sf = (
        np.isfinite(true_string)
        & np.isfinite(pred_string)
        & np.isfinite(true_fret)
        & np.isfinite(pred_fret)
    )
    string_fret_joint = None
    if valid_sf.any():
        string_fret_joint = float(
            np.mean(
                (true_string[valid_sf] == pred_string[valid_sf])
                & (true_fret[valid_sf] == pred_fret[valid_sf])
            )
        )

    valid_pitch = np.isfinite(true_midi) & np.isfinite(pred_midi)
    mean_abs_pitch_cent = None
    pitch_within_50_cent = None
    if valid_pitch.any():
        pitch_cent_error = np.abs(true_midi[valid_pitch] - pred_midi[valid_pitch]) * 100.0
        mean_abs_pitch_cent = float(np.mean(pitch_cent_error))
        pitch_within_50_cent = float(np.mean(pitch_cent_error <= 50.0))

    edit_distance = _levenshtein_distance(y_true, y_pred)
    edit_similarity = 1.0
    denom = max(len(y_true), len(y_pred))
    if denom > 0:
        edit_similarity = float(1.0 - (edit_distance / denom))

    return {
        "musical_metrics_scope": "token_index_mapping",
        "note_accuracy": _accuracy_where_valid(true_midi, pred_midi),
        "midi_accuracy": _accuracy_where_valid(true_midi, pred_midi),
        "pitch_accuracy_50_cent": pitch_within_50_cent,
        "mean_abs_pitch_error_cent": mean_abs_pitch_cent,
        "string_accuracy": _accuracy_where_valid(true_string, pred_string),
        "fret_accuracy": _accuracy_where_valid(true_fret, pred_fret),
        "string_fret_joint_accuracy": string_fret_joint,
        "mean_abs_string_error": _mean_abs_error_where_valid(true_string, pred_string),
        "mean_abs_fret_error": _mean_abs_error_where_valid(true_fret, pred_fret),
        "tablature_edit_distance": int(edit_distance),
        "tablature_edit_similarity": edit_similarity,
        "onset_tolerance_sec": float(onset_tolerance_sec),
        "onset_precision": None,
        "onset_recall": None,
        "onset_f1": None,
        "onset_metrics_note": (
            "Unavailable in token-window evaluation because predicted event onsets are not produced."
        ),
    }
