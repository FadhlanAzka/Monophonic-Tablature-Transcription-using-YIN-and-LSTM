from pathlib import Path

import numpy as np
import pandas as pd
import torch

from app import (
    _build_dataframe_from_stable,
    _build_midi_to_mask,
    _collapse_to_sustains,
    _load_token_index_df,
    render_ascii_tab,
)


def test_load_token_index_df_and_pitch_mask(tmp_path: Path):
    csv_path = tmp_path / "tokens.csv"
    pd.DataFrame(
        {
            "token_idx": [0, 1, 2],
            "midi": [60, 60, 62],
            "string": [2, 3, 2],
            "fret": [1, 5, 3],
        }
    ).to_csv(csv_path, index=False)

    token_df = _load_token_index_df(csv_path)
    mask = _build_midi_to_mask(token_df, num_classes=3, device="cpu")

    assert token_df.loc[0, "string"] == 2
    assert torch.equal(mask[60], torch.tensor([True, True, False]))
    assert torch.equal(mask[62], torch.tensor([False, False, True]))


def test_collapse_to_sustains_preserves_timing_and_onset_split():
    df0 = _build_dataframe_from_stable(
        stable_f0=np.array([261.63, 261.63, 261.63, 261.63]),
        stable_notes=np.array(["C4", "C4", "C4", "C4"], dtype=object),
        stable_midi=np.array([60.0, 60.0, 60.0, 60.0]),
        sample_times=np.array([0.0, 0.05, 0.10, 0.15]),
    )

    collapsed = _collapse_to_sustains(df0, onset_times=np.array([0.10]), min_run=2)

    assert len(collapsed) == 2
    assert list(collapsed["segment_source"]) == ["pitch", "onset"]
    assert np.isclose(collapsed.iloc[0]["start_time"], 0.0)
    assert np.isclose(collapsed.iloc[1]["start_time"], 0.10)


def test_ascii_tab_rendering_contains_string_labels_and_frets():
    tab = render_ascii_tab(strings=[6, 5, 4], frets=[0, 2, 3])

    assert "E|" in tab
    assert "A|" in tab
    assert "D|" in tab
    assert "0" in tab
    assert "2" in tab
    assert "3" in tab
