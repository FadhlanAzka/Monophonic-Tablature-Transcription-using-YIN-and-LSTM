from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

import tkinter as tk
from tkinter import filedialog

from audio_loader import load_audio
from preprocessing.hpss_processing import apply_hpss
from analysis.yin_pitch import compute_pitch_track
from analysis.yin_postproc import postprocess_yin
from analysis.noise_filter import apply_rms_noise_filter
from analysis.block_sampler import block_sample_pitch
from analysis.pitch_stabilizer import stabilize_pitch
from analysis.beat_tracking import compute_beats
from analysis.onset_detection import compute_onsets
from utils.sanitize import ensure_finite
from settings import (
    HOP_LENGTH,
    PITCH_DETECTION_MODE,
    PITCH_CONFIDENCE_THRESHOLD,
    TAB_OPTIMIZER_ENABLED,
    TAB_COST_FRET_DISTANCE,
    TAB_COST_STRING_CHANGE,
    TAB_COST_LARGE_SHIFT,
    TAB_LARGE_SHIFT_THRESHOLD,
    TAB_COST_OPEN_STRING,
    TAB_COST_POSITION,
    TAB_PREFERRED_FRET,
)

try:
    from evaluation.music_theory import midi_to_note_sharp, normalize_note_to_sharp
except Exception:
    SHARP_NAMES = [
        "C", "C#", "D", "D#", "E", "F",
        "F#", "G", "G#", "A", "A#", "B"
    ]

    def midi_to_note_sharp(midi: float) -> str:
        if not np.isfinite(midi):
            return ""
        m = int(round(midi))
        octave = (m // 12) - 1
        name = SHARP_NAMES[m % 12]
        return f"{name}{octave}"

    def normalize_note_to_sharp(x: str) -> str:
        return x

MIN_RUN_LEN = 2
MIN_NOTE_DURATION_SEC = 0.08
MIN_REST_DURATION_SEC = 0.08
ONSET_SPLIT_TOLERANCE_SEC = 0.035
NOTE_OUTPUT_COLUMNS = [
    "start_time",
    "end_time",
    "duration",
    "frame_count",
    "segment_source",
    "onset_aligned",
    "pitch_mode",
    "pitch_confidence",
    "octave_corrected",
    "hz",
    "note",
    "midi",
    "string",
    "fret",
    "token_idx",
    "mapping_decoder",
]

def _build_dataframe_from_stable(
    stable_f0: np.ndarray,
    stable_notes: np.ndarray,
    stable_midi: np.ndarray,
    sample_times: Optional[np.ndarray] = None,
    include_rests: bool = False,
    pitch_confidence: Optional[np.ndarray] = None,
    pitch_mode: str = PITCH_DETECTION_MODE,
    octave_corrected: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    notes = np.array(
        [normalize_note_to_sharp(n) if isinstance(n, str) else "" for n in stable_notes],
        dtype=object,
    )
    n_rows = len(stable_f0)
    if sample_times is None:
        starts = np.arange(n_rows, dtype=float)
    else:
        starts = np.asarray(sample_times, dtype=float)
        if len(starts) != n_rows:
            if len(starts) == 0:
                starts = np.arange(n_rows, dtype=float) * 0.05
            else:
                starts = np.resize(starts, n_rows).astype(float)

    if n_rows > 1:
        diffs = np.diff(starts)
        positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        default_step = float(np.median(positive_diffs)) if len(positive_diffs) else 0.05
        ends = np.empty(n_rows, dtype=float)
        ends[:-1] = starts[1:]
        ends[-1] = starts[-1] + default_step
    elif n_rows == 1:
        ends = starts + 0.05
    else:
        ends = np.asarray([], dtype=float)

    durations = np.maximum(0.0, ends - starts)
    if pitch_confidence is None or len(pitch_confidence) != n_rows:
        confidence = np.where(np.isfinite(stable_midi), 1.0, 0.0)
    else:
        confidence = np.asarray(pitch_confidence, dtype=float)
        confidence = np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)

    if octave_corrected is None or len(octave_corrected) != n_rows:
        octave_flags = np.zeros(n_rows, dtype=bool)
    else:
        octave_flags = np.asarray(octave_corrected, dtype=bool)

    df = pd.DataFrame({
        "start_time": starts,
        "end_time": ends,
        "duration": durations,
        "frame_count": np.ones(n_rows, dtype=int),
        "segment_source": "frame",
        "onset_aligned": False,
        "pitch_mode": str(pitch_mode),
        "pitch_confidence": confidence,
        "octave_corrected": octave_flags,
        "hz": stable_f0.astype(float),
        "note": notes,
        "midi": stable_midi.astype(float),
    })
    if include_rests:
        invalid = ~(np.isfinite(df["midi"]) & (df["note"] != "N/A"))
        df.loc[invalid, "note"] = "REST"
        df.loc[invalid, "hz"] = np.nan
        df.loc[invalid, "midi"] = np.nan
        return df.reset_index(drop=True)

    mask = np.isfinite(df["midi"]) & (df["note"] != "N/A")
    return df[mask].reset_index(drop=True)


def _collapse_to_sustains(
    df: pd.DataFrame,
    min_run: int = MIN_RUN_LEN,
    onset_times: Optional[np.ndarray] = None,
    onset_tolerance: float = ONSET_SPLIT_TOLERANCE_SEC,
    min_note_duration: float = MIN_NOTE_DURATION_SEC,
    min_rest_duration: float = MIN_REST_DURATION_SEC,
) -> pd.DataFrame:
    if len(df) == 0:
        return df

    midi_int = pd.to_numeric(df["midi"], errors="coerce").round().astype("Int64")
    pitch_boundary = midi_int.ne(midi_int.shift(1))
    has_timing = {"start_time", "end_time", "duration"}.issubset(df.columns)

    onset_boundary = pd.Series(False, index=df.index)
    onset_aligned = pd.Series(False, index=df.index)
    if has_timing and onset_times is not None:
        onset_arr = np.asarray(onset_times, dtype=float)
        onset_arr = onset_arr[np.isfinite(onset_arr)]
        if len(onset_arr):
            starts = pd.to_numeric(df["start_time"], errors="coerce").to_numpy(dtype=float)
            ends = pd.to_numeric(df["end_time"], errors="coerce").to_numpy(dtype=float)
            for i, (start, end) in enumerate(zip(starts, ends)):
                if not np.isfinite(start) or not np.isfinite(end):
                    continue
                near = onset_arr[
                    (onset_arr >= start - float(onset_tolerance))
                    & (onset_arr < end + float(onset_tolerance))
                ]
                if len(near):
                    onset_aligned.iloc[i] = True
                    if i > 0 and not bool(pitch_boundary.iloc[i]):
                        prev_start = starts[i - 1] if i - 1 < len(starts) else np.nan
                        if np.isfinite(prev_start) and np.min(np.abs(near - prev_start)) > float(onset_tolerance):
                            onset_boundary.iloc[i] = True

    boundary = pitch_boundary | onset_boundary
    run_id = boundary.cumsum()

    df_tmp = df.copy()
    df_tmp["run_id"] = run_id.values
    df_tmp["onset_aligned"] = onset_aligned.values
    df_tmp["segment_source"] = np.where(onset_boundary.values, "onset", "pitch")

    lens = df_tmp.groupby("run_id", sort=False).size()
    run_has_onset = df_tmp.groupby("run_id", sort=False)["onset_aligned"].any()
    valid_ids = lens[(lens >= int(min_run)) | run_has_onset].index
    if len(valid_ids) == 0:
        return pd.DataFrame(columns=["start_time", "end_time", "duration", "frame_count", "segment_source", "onset_aligned", "pitch_mode", "pitch_confidence", "octave_corrected", "hz", "note", "midi"])

    df_valid = df_tmp[df_tmp["run_id"].isin(valid_ids)]

    df_work = df_valid.assign(
        hz_num=pd.to_numeric(df_valid["hz"], errors="coerce"),
        midi_num=pd.to_numeric(df_valid["midi"], errors="coerce"),
    )

    agg_map = {
        "hz": ("hz_num", lambda s: float(np.nanmedian(s))),
        "midi": ("midi_num", lambda s: float(np.nanmedian(s))),
        "frame_count": ("midi_num", "size"),
        "segment_source": ("segment_source", lambda s: "onset" if (s == "onset").any() else "pitch"),
        "onset_aligned": ("onset_aligned", "any"),
    }
    if "pitch_mode" in df_valid.columns:
        agg_map["pitch_mode"] = ("pitch_mode", lambda s: str(s.iloc[0]))
    if "pitch_confidence" in df_valid.columns:
        df_work = df_work.assign(
            pitch_confidence_num=pd.to_numeric(df_valid["pitch_confidence"], errors="coerce")
        )
        agg_map["pitch_confidence"] = ("pitch_confidence_num", lambda s: float(np.nanmean(s)))
    if "octave_corrected" in df_valid.columns:
        agg_map["octave_corrected"] = ("octave_corrected", "any")
    if has_timing:
        df_work = df_work.assign(
            start_time_num=pd.to_numeric(df_valid["start_time"], errors="coerce"),
            end_time_num=pd.to_numeric(df_valid["end_time"], errors="coerce"),
            duration_num=pd.to_numeric(df_valid["duration"], errors="coerce"),
        )
        agg_map.update({
            "start_time": ("start_time_num", "min"),
            "end_time": ("end_time_num", "max"),
            "duration": ("duration_num", "sum"),
        })

    agg = df_work.groupby("run_id", sort=False).agg(**agg_map).reset_index(drop=True)

    agg["midi"] = agg["midi"].round()
    agg["note"] = agg["midi"].apply(midi_to_note_sharp)
    if has_timing:
        agg["duration"] = np.maximum(0.0, agg["end_time"] - agg["start_time"])
        min_duration = np.where(agg["note"].eq("REST"), float(min_rest_duration), float(min_note_duration))
        duration_mask = (agg["duration"] >= min_duration) | agg["onset_aligned"].astype(bool)
        agg = agg[duration_mask].reset_index(drop=True)
        cols = ["start_time", "end_time", "duration", "frame_count", "segment_source", "onset_aligned", "pitch_mode", "pitch_confidence", "octave_corrected", "hz", "note", "midi"]
        return agg[[c for c in cols if c in agg.columns]]
    cols = ["frame_count", "segment_source", "onset_aligned", "pitch_mode", "pitch_confidence", "octave_corrected", "hz", "note", "midi"]
    return agg[[c for c in cols if c in agg.columns]]


def _save_basic_visualization(
    out_png: Path,
    audio_path: Path,
    sr: int,
    times: np.ndarray,
    f0_for_plot: np.ndarray,
    notes_for_plot: np.ndarray,
    beat_times: np.ndarray,
    onset_times: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4), dpi=120)
    ax = plt.gca()

    ax.plot(times, f0_for_plot, linewidth=1.25)
    ax.scatter(times, f0_for_plot, s=8)

    if beat_times is not None and len(beat_times):
        for t in beat_times:
            ax.axvline(t, alpha=0.15)
    if onset_times is not None and len(onset_times):
        for t in onset_times:
            ax.axvline(t, alpha=0.15)

    ax.set_title(f"YINTab Visualization — {audio_path.name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("F0 (Hz)")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def _block_sample_values(times: np.ndarray, values: np.ndarray, sample_times: np.ndarray, default=np.nan) -> np.ndarray:
    if values is None or len(values) == 0 or len(sample_times) == 0:
        return np.full(len(sample_times), default)

    times = np.asarray(times, dtype=float)
    values = np.asarray(values)
    if len(times) != len(values):
        values = np.resize(values, len(times))

    if len(sample_times) > 1:
        block_len = float(np.median(np.diff(sample_times)))
        if not np.isfinite(block_len) or block_len <= 0:
            block_len = 0.05
    else:
        block_len = 0.05

    out = []
    for t0 in sample_times:
        t1 = t0 + block_len
        idx = np.where((times >= t0) & (times < t1))[0]
        if len(idx) == 0:
            out.append(default)
            continue
        block_vals = values[idx]
        if block_vals.dtype == bool:
            out.append(bool(np.any(block_vals)))
            continue
        block_vals = block_vals.astype(float, copy=False)
        block_vals = block_vals[np.isfinite(block_vals)]
        out.append(float(np.nanmean(block_vals)) if len(block_vals) else default)
    return np.asarray(out)


# =========================
#  Core Pipeline #14
# =========================
def run_pipeline14(
    y: np.ndarray,
    sr: int,
    pitch_mode: str = PITCH_DETECTION_MODE,
    confidence_threshold: float = PITCH_CONFIDENCE_THRESHOLD,
    return_diagnostics: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        hpss_out = apply_hpss(y, sr)
    except TypeError:
        hpss_out = apply_hpss(y)
    y_harm = hpss_out[0] if isinstance(hpss_out, (list, tuple)) else hpss_out

    pitch = compute_pitch_track(y_harm, sr, mode=pitch_mode)
    f0_raw = pitch["f0"].copy()
    times = pitch["times"]
    confidence = np.asarray(pitch["confidence"], dtype=float)
    if confidence_threshold > 0:
        f0_raw[confidence < float(confidence_threshold)] = np.nan

    f0_nf = apply_rms_noise_filter(y_harm, sr, f0_raw, hop_length=HOP_LENGTH)

    f0_corr = postprocess_yin(
        f0_nf,
        y_harm,
        sr,
        n_fft=4096,
        hop_length=HOP_LENGTH,
        median_kernel=5,
        max_harm=4,
        continuity_weight=0.8,
    )

    sample_times, sample_f0 = block_sample_pitch(
        times,
        f0_corr,
        block_ms=50,
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    sample_confidence = _block_sample_values(times, confidence, sample_times, default=0.0)
    sample_raw_f0 = _block_sample_values(times, f0_nf, sample_times, default=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        octave_delta = np.abs(np.log2(sample_f0 / sample_raw_f0))
    octave_corrected = np.isfinite(octave_delta) & (octave_delta >= 0.5)

    stable_notes, stable_f0, stable_midi = stabilize_pitch(sample_times, sample_f0)
    if return_diagnostics:
        diagnostics = {
            "pitch_mode": pitch["mode"],
            "pitch_confidence": sample_confidence,
            "octave_corrected": octave_corrected,
        }
        return stable_notes, stable_f0, stable_midi, sample_times, diagnostics
    return stable_notes, stable_f0, stable_midi, sample_times

def pick_wav_file() -> Optional[Path]:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Pilih file WAV",
        filetypes=[("WAV file", "*.wav")],
    )
    root.destroy()
    return Path(path) if path else None


def pick_output_folder() -> Optional[Path]:
    root = tk.Tk()
    root.withdraw()
    base = filedialog.askdirectory(
        title="Pilih folder output (akan dibuat subfolder nama WAV)",
    )
    root.destroy()
    return Path(base) if base else None


def pick_token_index_csv() -> Optional[Path]:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Pilih file token index.csv",
        filetypes=[("CSV file", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(path) if path else None


def render_ascii_tab(strings, frets, width_pad: int = 1) -> str:
    string_labels = {1: "e", 2: "B", 3: "G", 4: "D", 5: "A", 6: "E"}
    lines = {s: [] for s in string_labels}
    width_pad = max(1, int(width_pad))

    if len(strings) == 0 or len(frets) == 0:
        return "\n".join(f"{name}||" for _, name in sorted(string_labels.items()))

    if len(strings) != len(frets):
        raise ValueError(
            f"Panjang strings ({len(strings)}) dan frets ({len(frets)}) tidak sama."
        )

    for s_val, f_val in zip(strings, frets):
        for s in string_labels:
            lines[s].append("-")

        try:
            s_idx = int(s_val)
            fret = int(f_val)
        except Exception:
            s_idx, fret = None, None

        if (
            s_idx in string_labels
            and fret is not None
            and np.isfinite(fret)
            and fret >= 0
        ):
            mark = str(int(fret))
            lines[s_idx][-1] = mark[0]

            if len(mark) > 1:
                for extra_digit in mark[1:]:
                    for ss in string_labels:
                        lines[ss].append("-")
                    lines[s_idx][-1] = extra_digit

        for s in string_labels:
            for _ in range(width_pad):
                lines[s].append("-")

    '''out_lines = []
    for s in sorted(string_labels.keys()):
        name = string_labels[s]
        out_lines.append(f"{name}|{''.join(lines[s])}|")
    return "\n".join(out_lines)'''
    out_lines = []
    for s in sorted(string_labels.keys(), reverse=True):
        name = string_labels[s]
        out_lines.append(f"{name}|{''.join(lines[s])}|")
    return "\n".join(out_lines)

def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _pick_lstm_model() -> Optional[Path]:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Pilih file model LSTM (best.pt / best.ts.pt / best.jit)",
        filetypes=[("PyTorch model", "*.pt *.jit"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(path) if path else None

def _load_token_index_df(csv_path: Path) -> pd.DataFrame:
    df_tok = pd.read_csv(csv_path)

    if "token_idx" not in df_tok.columns:
        raise ValueError(
            f"token index CSV '{csv_path}' tidak memiliki kolom 'token_idx'."
        )

    if "string" not in df_tok.columns or "fret" not in df_tok.columns:
        raise ValueError(
            f"token index CSV '{csv_path}' harus memiliki kolom 'string' dan 'fret'."
        )

    if "midi" not in df_tok.columns:
        print(
            "[WARN] Kolom 'midi' tidak ditemukan di token index; "
        )

    return df_tok.set_index("token_idx")


def _load_lstm_model(model_path: Path, device: str):
    path_str = str(model_path)
    lower = path_str.lower()

    if lower.endswith(".ts.pt") or lower.endswith(".jit"):
        model = torch.jit.load(path_str, map_location=device)
        model.eval()
        return model

    obj = torch.load(path_str, map_location=device)

    if hasattr(obj, "forward"):
        model = obj
    elif isinstance(obj, dict) and "model_state_dict" in obj:
        raise RuntimeError(
            "File ini tampaknya checkpoint (dict dengan 'model_state_dict'). "
            "Untuk inference, gunakan file .pt (full model) atau .ts.pt / .jit (TorchScript)."
        )
    else:
        raise RuntimeError(f"Tidak bisa interpret file model: {model_path}")

    model.eval()
    return model

def _build_midi_to_mask(
    token_index_df: pd.DataFrame,
    num_classes: int,
    device: str,
) -> Dict[int, torch.BoolTensor]:
    if "midi" not in token_index_df.columns:
        return {}

    midi_to_mask: Dict[int, torch.BoolTensor] = {}

    for token_idx, row in token_index_df.iterrows():
        try:
            midi_val = int(row["midi"])
        except Exception:
            continue

        if midi_val not in midi_to_mask:
            mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
            midi_to_mask[midi_val] = mask

        idx = int(token_idx)
        if 0 <= idx < num_classes:
            midi_to_mask[midi_val][idx] = True

    return midi_to_mask


def _apply_pitch_mask_for_sequence(
    logits: torch.Tensor,          
    midi_seq: np.ndarray,          
    midi_to_mask: Dict[int, torch.BoolTensor],
) -> torch.Tensor:
    if not midi_to_mask:
        return logits

    if logits.dim() == 2:
        logits = logits.unsqueeze(0)

    B, T, C = logits.shape
    if T != len(midi_seq):
        print(
            f"[WARN] Panjang midi_seq ({len(midi_seq)}) "
            f"≠ panjang logits ({T}); pitch mask mungkin tidak konsisten."
        )

    logit_mask_value = -1e9

    for t in range(min(T, len(midi_seq))):
        midi_val = int(midi_seq[t])
        mask_1d = midi_to_mask.get(midi_val, None)
        if mask_1d is None:
            continue

        logits[:, t, ~mask_1d] = logit_mask_value

    return logits


def _transition_playability_cost(
    prev_string: float,
    prev_fret: float,
    curr_string: float,
    curr_fret: float,
) -> float:
    if not all(np.isfinite([prev_string, prev_fret, curr_string, curr_fret])):
        return 0.0

    fret_delta = abs(float(curr_fret) - float(prev_fret))
    string_delta = abs(float(curr_string) - float(prev_string))
    large_shift = max(0.0, fret_delta - float(TAB_LARGE_SHIFT_THRESHOLD))

    return (
        float(TAB_COST_FRET_DISTANCE) * fret_delta
        + float(TAB_COST_STRING_CHANGE) * string_delta
        + float(TAB_COST_LARGE_SHIFT) * large_shift
    )


def _state_playability_cost(string_val: float, fret_val: float) -> float:
    if not np.isfinite(fret_val):
        return 0.0

    open_string_cost = float(TAB_COST_OPEN_STRING) if int(round(float(fret_val))) == 0 else 0.0
    position_cost = float(TAB_COST_POSITION) * abs(float(fret_val) - float(TAB_PREFERRED_FRET))
    return open_string_cost + position_cost


def _viterbi_decode_tablature(
    logits: torch.Tensor,
    token_index_df: pd.DataFrame,
) -> Optional[np.ndarray]:
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    if logits.dim() != 2:
        return None

    scores = torch.log_softmax(logits, dim=-1).detach().cpu().numpy()
    T, C = scores.shape
    if T == 0 or C == 0:
        return np.asarray([], dtype=int)

    idx = pd.Index(range(C), name=token_index_df.index.name)
    tok = token_index_df.reindex(idx)
    string_col = tok["string"] if "string" in tok.columns else pd.Series(np.nan, index=tok.index)
    fret_col = tok["fret"] if "fret" in tok.columns else pd.Series(np.nan, index=tok.index)
    strings = pd.to_numeric(string_col, errors="coerce").to_numpy(dtype=float)
    frets = pd.to_numeric(fret_col, errors="coerce").to_numpy(dtype=float)
    has_position = np.isfinite(strings) & np.isfinite(frets)
    if not has_position.any():
        return None

    candidates: List[np.ndarray] = []
    for t in range(T):
        valid = np.where(np.isfinite(scores[t]) & (scores[t] > -1e8) & has_position)[0]
        if len(valid) == 0:
            valid = np.asarray([int(np.nanargmax(scores[t]))], dtype=int)
        candidates.append(valid)

    dp: List[np.ndarray] = []
    back: List[np.ndarray] = []

    first = candidates[0]
    first_scores = np.asarray([
        scores[0, tok_idx] - _state_playability_cost(strings[tok_idx], frets[tok_idx])
        for tok_idx in first
    ], dtype=float)
    dp.append(first_scores)
    back.append(np.full(len(first), -1, dtype=int))

    for t in range(1, T):
        curr = candidates[t]
        prev = candidates[t - 1]
        curr_scores = np.full(len(curr), -np.inf, dtype=float)
        curr_back = np.full(len(curr), -1, dtype=int)

        for j, curr_idx in enumerate(curr):
            state_cost = _state_playability_cost(strings[curr_idx], frets[curr_idx])
            best_score = -np.inf
            best_prev_pos = -1
            for i, prev_idx in enumerate(prev):
                trans_cost = _transition_playability_cost(
                    strings[prev_idx],
                    frets[prev_idx],
                    strings[curr_idx],
                    frets[curr_idx],
                )
                score = dp[t - 1][i] + scores[t, curr_idx] - state_cost - trans_cost
                if score > best_score:
                    best_score = score
                    best_prev_pos = i
            curr_scores[j] = best_score
            curr_back[j] = best_prev_pos

        dp.append(curr_scores)
        back.append(curr_back)

    pred_positions = np.zeros(T, dtype=int)
    pred_positions[-1] = int(np.nanargmax(dp[-1]))
    for t in range(T - 1, 0, -1):
        pred_positions[t - 1] = back[t][pred_positions[t]]

    preds = np.asarray([candidates[t][pred_positions[t]] for t in range(T)], dtype=int)
    return preds


def _run_lstm_mapping(
    df_notes: pd.DataFrame,
    model,
    token_index_df: pd.DataFrame,
    device: str,
    use_pitch_mask: bool = True,
    use_sequence_optimizer: bool = TAB_OPTIMIZER_ENABLED,
) -> pd.DataFrame:
    if len(df_notes) == 0:
        return df_notes.copy()

    df_notes = df_notes.reset_index(drop=True)
    midi_seq = df_notes["midi"].round().astype(int).to_numpy(dtype="int64")

    with torch.no_grad():
        midi_tensor = torch.from_numpy(midi_seq)[None, :]
        midi_tensor = midi_tensor.to(device)

        logits = model(midi_tensor)

        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        B, T, C = logits.shape

        if use_pitch_mask and ("midi" in token_index_df.columns):
            midi_to_mask = _build_midi_to_mask(
                token_index_df=token_index_df,
                num_classes=C,
                device=logits.device,
            )
            if not midi_to_mask:
                print(
                    "[WARN] midi_to_mask kosong; kolom 'midi' mungkin bermasalah. "
                    "Logits dipakai tanpa pitch mask."
                )
            else:
                logits = _apply_pitch_mask_for_sequence(
                    logits=logits,
                    midi_seq=midi_seq,
                    midi_to_mask=midi_to_mask,
                )
        else:
            if use_pitch_mask:
                print(
                    "[WARN] Kolom 'midi' tidak tersedia di token_index_df; "
                )

        decoder_name = "argmax"
        if use_sequence_optimizer:
            opt_preds = _viterbi_decode_tablature(logits, token_index_df)
            if opt_preds is not None and len(opt_preds) == T:
                preds = opt_preds
                decoder_name = "viterbi"
            else:
                preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
                decoder_name = "argmax_fallback"
        else:
            preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

    if len(preds) != len(df_notes):
        raise RuntimeError(
            f"Panjang prediksi ({len(preds)}) tidak sama "
            f"dengan panjang df_notes ({len(df_notes)})."
        )

    df_out = df_notes.copy()
    df_out["token_idx"] = preds.astype(int)
    df_out["mapping_decoder"] = decoder_name

    df_out["string"] = df_out["token_idx"].map(token_index_df["string"])
    df_out["fret"] = df_out["token_idx"].map(token_index_df["fret"])

    return df_out


def run_app() -> None:
    
    wav_path = pick_wav_file()
    if not wav_path:
        print("Dibatalkan: tidak ada WAV dipilih.")
        return

    base_out = pick_output_folder()
    if not base_out:
        print("Dibatalkan: tidak ada folder output dipilih.")
        return

    out_dir = base_out / wav_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    tok_csv = pick_token_index_csv()
    if not tok_csv:
        print("Dibatalkan: tidak ada token index.csv dipilih.")
        return

    token_index_df = _load_token_index_df(tok_csv)

    model_path = _pick_lstm_model()
    if not model_path:
        print("Dibatalkan: tidak ada model LSTM dipilih.")
        return

    device = _detect_device()
    print(f"[INFO] Device: {device}")
    model = _load_lstm_model(model_path, device=device)
    print(f"[INFO] Model loaded from: {model_path}")

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

    print(f"[INFO] Menjalankan Pipeline #14 ({PITCH_DETECTION_MODE})...")
    stable_notes, stable_f0, stable_midi, sample_times, pitch_diag = run_pipeline14(
        y,
        sr,
        return_diagnostics=True,
    )

    duration = len(y) / sr
    f0_vis = stable_f0
    notes_vis = np.asarray(
        [normalize_note_to_sharp(n) if isinstance(n, str) else n for n in stable_notes],
        dtype=object,
    )

    try:
        from viz.visualizer import show_visualization, save_visualization
    except Exception:
        from visualizer import show_visualization, save_visualization

    try:
        show_visualization(
            audio_path=str(wav_path),
            duration=duration,
            sr=sr,
            tempo=tempo,
            beat_times=beat_times,
            onset_times=onset_times,
            times=sample_times,
            f0=f0_vis,
            notes=notes_vis,
        )
    except Exception as e:
        print(f"[WARN] show_visualization gagal: {e}")

    try:
        save_visualization(out_dir)(
            audio_path=str(wav_path),
            duration=duration,
            sr=sr,
            tempo=tempo,
            beat_times=beat_times,
            onset_times=onset_times,
            times=sample_times,
            f0=f0_vis,
            notes=notes_vis,
        )
        print(f"[OK] Visualization disimpan ke folder: {out_dir}")
    except Exception as e:
        out_png = out_dir / f"{wav_path.stem}_viz.png"
        _save_basic_visualization(
            out_png=out_png,
            audio_path=wav_path,
            sr=sr,
            times=sample_times,
            f0_for_plot=f0_vis,
            notes_for_plot=notes_vis,
            beat_times=np.asarray(beat_times) if beat_times is not None else np.asarray([]),
            onset_times=np.asarray(onset_times),
        )
        print(f"[WARN] Modul viz gagal, pakai fallback sederhana. Disimpan: {out_png} ({e})")

    df0 = _build_dataframe_from_stable(
        stable_f0,
        stable_notes,
        stable_midi,
        sample_times,
        pitch_confidence=pitch_diag.get("pitch_confidence"),
        pitch_mode=pitch_diag.get("pitch_mode", PITCH_DETECTION_MODE),
        octave_corrected=pitch_diag.get("octave_corrected"),
    )
    if len(df0) == 0:
        print(f"[SKIP] {wav_path.name}: no valid notes")
        out_csv = out_dir / f"{wav_path.stem}_lstm_mapped.csv"
        pd.DataFrame(
            columns=NOTE_OUTPUT_COLUMNS,
        ).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] Empty CSV disimpan ke: {out_csv}")
        return

    df = _collapse_to_sustains(df0, onset_times=onset_times)
    if len(df) == 0:
        print(f"[SKIP] {wav_path.name}: no sustained notes after RLE")
        out_csv = out_dir / f"{wav_path.stem}_lstm_mapped.csv"
        pd.DataFrame(
            columns=NOTE_OUTPUT_COLUMNS,
        ).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] Empty CSV disimpan ke: {out_csv}")
        return

    print("[INFO] Menjalankan LSTM Mapper...")
    df_out = _run_lstm_mapping(
        df_notes=df,
        model=model,
        token_index_df=token_index_df,
        device=device,
        use_pitch_mask=True,
    )

    cols = NOTE_OUTPUT_COLUMNS
    cols = [c for c in cols if c in df_out.columns]
    out_csv = out_dir / f"{wav_path.stem}_lstm_mapped.csv"
    df_out[cols].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV LSTM-mapped disimpan: {out_csv}")

    if {"string", "fret"}.issubset(df_out.columns):
        try:
            strings = df_out["string"].tolist()
            frets = df_out["fret"].tolist()
            tab_txt = render_ascii_tab(strings, frets, width_pad=1)
            out_tab = out_dir / f"{wav_path.stem}_lstm_tab.txt"
            with open(out_tab, "w", encoding="utf-8") as f:
                f.write(tab_txt)
            print(f"[OK] ASCII tablature LSTM disimpan: {out_tab}")
        except Exception as e:
            print(f"[WARN] Gagal membuat ASCII tablature LSTM: {e}")
    else:
        print(
            "[WARN] Kolom 'string' dan/atau 'fret' tidak ditemukan — ASCII tab tidak dibuat."
        )

    print("[DONE] Inference LSTM Mapper selesai.")


def main():
    run_app()


if __name__ == "__main__":
    main()
