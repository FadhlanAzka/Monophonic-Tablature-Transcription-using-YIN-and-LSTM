from pathlib import Path

import numpy as np
import pandas as pd

import core_pipeline
from core_pipeline import PipelineConfig, run_transcription_pipeline


class DummyModel:
    pass


def test_run_transcription_pipeline_smoke(monkeypatch, tmp_path: Path):
    wav_path = tmp_path / "input.wav"
    wav_path.write_bytes(b"not a real wav because load_audio is patched")

    token_csv = tmp_path / "tokens.csv"
    pd.DataFrame(
        {
            "hz": [261.63],
            "note": ["C4"],
            "midi": [60],
            "string": [2],
            "fret": [1],
            "token_idx": [0],
        }
    ).to_csv(token_csv, index=False)

    model_path = tmp_path / "model.jit"
    model_path.write_bytes(b"placeholder")

    def fake_load_audio(_):
        return np.zeros(22050, dtype=np.float32), 22050

    def fake_pipeline(_y, _sr, return_diagnostics=False, **_kwargs):
        diag = {
            "pitch_confidence": np.array([1.0, 1.0]),
            "pitch_mode": "yin",
            "octave_corrected": np.array([False, False]),
        }
        result = (
            np.array(["C4", "C4"], dtype=object),
            np.array([261.63, 261.63]),
            np.array([60.0, 60.0]),
            np.array([0.0, 0.05]),
        )
        return (*result, diag) if return_diagnostics else result

    def fake_mapping(df_notes, *_args, **_kwargs):
        out = df_notes.copy()
        out["token_idx"] = 0
        out["string"] = 2
        out["fret"] = 1
        out["mapping_decoder"] = "test"
        return out

    def fake_save_visualization(out_dir):
        def _save(**_kwargs):
            (Path(out_dir) / "input_viz.png").write_bytes(b"png")
        return _save

    monkeypatch.setattr(core_pipeline, "load_audio", fake_load_audio)
    monkeypatch.setattr(core_pipeline, "compute_beats", lambda *_args, **_kwargs: (0.0, np.asarray([])))
    monkeypatch.setattr(core_pipeline, "compute_onsets", lambda *_args, **_kwargs: np.asarray([]))
    monkeypatch.setattr(core_pipeline, "run_pipeline14", fake_pipeline)
    monkeypatch.setattr(core_pipeline, "_load_lstm_model", lambda *_args, **_kwargs: DummyModel())
    monkeypatch.setattr(core_pipeline, "_run_lstm_mapping", fake_mapping)
    monkeypatch.setattr(core_pipeline, "save_visualization", fake_save_visualization)

    artifacts = run_transcription_pipeline(
        PipelineConfig(
            wav_path=wav_path,
            output_dir=tmp_path / "out",
            token_index_csv=token_csv,
            model_path=model_path,
            make_midi=True,
        )
    )

    assert artifacts.csv is not None and artifacts.csv.exists()
    assert artifacts.tab is not None and artifacts.tab.exists()
    assert artifacts.midi is not None and artifacts.midi.exists()
    assert artifacts.viz is not None and artifacts.viz.exists()
