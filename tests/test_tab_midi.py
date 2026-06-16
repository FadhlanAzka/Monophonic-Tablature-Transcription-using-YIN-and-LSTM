import csv
from pathlib import Path

from tab_midi import csv_notes_to_midi_file, csv_to_midi_sequence, csv_to_timed_midi_events


def test_legacy_csv_to_midi_sequence(tmp_path: Path):
    csv_path = tmp_path / "legacy.csv"
    csv_path.write_text("midi\n60\n\n62\n", encoding="utf-8")

    assert csv_to_midi_sequence(csv_path) == [60, None, 62]


def test_timed_csv_to_midi_events_and_file(tmp_path: Path):
    csv_path = tmp_path / "timed.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_time", "end_time", "duration", "midi"])
        writer.writerow([0.0, 0.25, 0.25, 60])
        writer.writerow([0.25, 0.75, 0.5, 62])

    events = csv_to_timed_midi_events(csv_path)
    assert events == [(60, 0.0, 0.25), (62, 0.25, 0.5)]

    out_mid = tmp_path / "out.mid"
    csv_notes_to_midi_file(csv_path, out_mid)
    assert out_mid.exists()
    assert out_mid.read_bytes().startswith(b"MThd")
