import math

from evaluation.music_theory import midi_to_note_sharp, normalize_note_to_sharp, note_to_midi


def test_normalize_note_to_sharp_handles_flats_and_unicode():
    assert normalize_note_to_sharp("Gb2") == "F#2"
    assert normalize_note_to_sharp("F♯2") == "F#2"


def test_note_to_midi_and_back():
    assert math.isclose(note_to_midi("E2"), 40.0)
    assert midi_to_note_sharp(61) == "C#4"
