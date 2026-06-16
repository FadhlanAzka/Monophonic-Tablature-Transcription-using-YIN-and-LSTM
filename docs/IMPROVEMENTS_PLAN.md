# Improvements Plan

## Purpose

This document defines the planned improvements for the monophonic tablature transcription project. It is intended to be used as a reference for future implementation prompts and task planning.

Each improvement should preserve the existing project behavior unless the task explicitly changes it. The preferred approach is incremental implementation with focused verification after each stage.

## Baseline

The current system already supports:

- WAV input.
- YIN-based monophonic pitch detection.
- HPSS harmonic preprocessing.
- RMS noise filtering.
- Octave correction and pitch stabilization.
- LSTM mapping from MIDI notes to string-fret tokens.
- Pitch mask based on the token index MIDI column.
- CSV, ASCII tab, MIDI, rendered WAV, and visualization outputs.
- Flask web UI and Tkinter desktop/local workflows.
- Validation and test evaluation scripts.

Main limitations:

- Timing information is mostly lost after note collapse.
- MIDI output uses a fixed notes-per-second value.
- Segmentation relies heavily on fixed blocks and repeated MIDI collapse.
- Several paths and runtime parameters are hard-coded.
- Core pipeline and UI code are tightly coupled.
- Evaluation is mostly token-classification based.
- Dependency and setup documentation are incomplete.

## Implementation Principles

- Keep changes small and verifiable.
- Preserve existing output filenames unless a task explicitly requires a new format.
- Prefer adding backward-compatible columns over replacing existing CSV columns.
- Avoid changing model interfaces unless model retraining is part of the task.
- Keep core pipeline code independent from Flask and Tkinter.
- Add tests for behavior that can regress silently.
- Document configuration defaults near the code that consumes them.

## Priority Roadmap

### Phase 1: Reproducibility and Configuration

Goal: make the project easier to install, run, and reproduce.

Tasks:

- Add `requirements.txt` or `pyproject.toml`.
- Document supported Python version.
- Document required external tools: FFmpeg, FluidSynth, optional Demucs.
- Move hard-coded constants into a central configuration module or config file.
- Replace hard-coded FluidSynth executable path with configurable lookup.
- Add clear startup checks for model, token index, SoundFont, and FluidSynth.
- Update `README.md` with install and run instructions.

Acceptance criteria:

- A new user can install dependencies from one documented command.
- `python app_web.py` can be run after following setup steps.
- Missing optional render dependencies produce warnings, not unexplained failures.
- The default model and token index paths are documented.

Suggested prompt:

```text
Using docs/PROJECT_REQUIREMENTS.md and docs/IMPROVEMENTS_PLAN.md as references, implement Phase 1: Reproducibility and Configuration. Add dependency documentation, centralize runtime configuration, replace hard-coded FluidSynth lookup with a configurable helper, and update README with setup/run instructions. Preserve existing app behavior.
```

### Phase 2: Timing-Preserving Note Representation

Goal: preserve start time, end time, and duration for each detected note.

Tasks:

- Extend the note dataframe to include `start_time`, `end_time`, `duration`, and possibly `frame_count`.
- Modify sustain collapse to aggregate timing as well as pitch.
- Keep existing columns: `hz`, `note`, `midi`, `string`, `fret`, `token_idx`.
- Update CSV writing to include timing columns.
- Update MIDI generation to use note durations when available.
- Keep fixed-step MIDI behavior as fallback for legacy CSV files.

Acceptance criteria:

- Output CSV includes timing columns.
- MIDI duration reflects detected note duration.
- Legacy CSV without timing columns can still be converted to MIDI.
- ASCII tab generation remains compatible.

Suggested prompt:

```text
Using docs/PROJECT_REQUIREMENTS.md and docs/IMPROVEMENTS_PLAN.md as references, implement Phase 2: Timing-Preserving Note Representation. Add start/end/duration columns to the transcription output, update sustain collapse, and update MIDI generation to use duration-aware events while preserving legacy CSV support.
```

### Phase 3: Better Note Segmentation

Goal: improve note boundaries beyond fixed 50 ms blocks and repeated MIDI collapse.

Tasks:

- Integrate onset detection into note boundary decisions.
- Split sustained pitch runs when strong onsets occur inside the same MIDI note.
- Preserve rests or silence regions instead of only dropping invalid frames.
- Add configurable thresholds for minimum note duration, minimum rest duration, and onset sensitivity.
- Add diagnostic columns such as `segment_source` or `onset_aligned` if useful.

Acceptance criteria:

- Repeated notes at the same pitch can be represented as separate events when onset evidence exists.
- Silence/rest regions can be represented or intentionally skipped based on configuration.
- Segmentation parameters are configurable.
- Existing simple melodies still produce sensible output.

Suggested prompt:

```text
Using docs/PROJECT_REQUIREMENTS.md and docs/IMPROVEMENTS_PLAN.md as references, implement Phase 3: Better Note Segmentation. Use onset times and pitch changes to build note events with timing, support repeated same-pitch notes, and keep the output backward compatible.
```

### Phase 4: Improved Pitch Detection Confidence

Goal: make pitch output more reliable and diagnosable.

Tasks:

- Add voiced/unvoiced confidence or a proxy confidence score.
- Evaluate using `librosa.pyin` as an optional mode if suitable.
- Add configurable selection between YIN and pYIN modes.
- Track octave-correction decisions for debugging.
- Save optional diagnostic columns or visualization layers.

Acceptance criteria:

- The pipeline can run with the existing YIN mode.
- Optional pYIN mode can be enabled without breaking default behavior.
- Low-confidence frames are handled consistently.
- Visualization or CSV diagnostics help identify pitch errors.

Suggested prompt:

```text
Using docs/PROJECT_REQUIREMENTS.md and docs/IMPROVEMENTS_PLAN.md as references, implement Phase 4: Improved Pitch Detection Confidence. Add configurable pitch detection modes, confidence handling, and diagnostics while keeping the current YIN pipeline as the default.
```

### Phase 5: Sequence-Aware Tablature Optimization

Goal: make predicted string-fret sequences more playable.

Tasks:

- Keep LSTM logits as the primary learned signal.
- Add post-processing with sequence costs for:
  - fret distance,
  - string changes,
  - large hand shifts,
  - open-string preference or penalty,
  - configurable position range preference.
- Use dynamic programming or Viterbi-style decoding over valid token candidates.
- Preserve pitch mask constraints.
- Expose cost weights in configuration.

Acceptance criteria:

- Every predicted token remains pitch-valid.
- Output avoids unrealistic jumps when alternatives exist.
- The optimizer can be disabled to reproduce current argmax behavior.
- Cost weights are documented.

Suggested prompt:

```text
Using docs/PROJECT_REQUIREMENTS.md and docs/IMPROVEMENTS_PLAN.md as references, implement Phase 5: Sequence-Aware Tablature Optimization. Add optional Viterbi/dynamic-programming decoding over LSTM logits with configurable playability costs, while preserving current argmax mapping as a fallback.
```

### Phase 6: Evaluation Improvements

Goal: evaluate musical quality, not only token classification quality.

Tasks:

- Add note-level evaluation with onset tolerance.
- Add pitch accuracy in cents.
- Add string accuracy and fret accuracy separately.
- Add tablature edit distance or sequence similarity metric.
- Add per-stage evaluation where possible:
  - pitch detection,
  - segmentation,
  - string-fret mapping.
- Save metrics in JSON and optionally plot summaries.

Acceptance criteria:

- Evaluation reports include token metrics and musical metrics.
- Metrics are documented in the output JSON.
- Existing evaluation scripts still work or have a clear migration path.

Suggested prompt:

```text
Using docs/PROJECT_REQUIREMENTS.md and docs/IMPROVEMENTS_PLAN.md as references, implement Phase 6: Evaluation Improvements. Add note-level, pitch-level, and tablature-specific metrics while preserving existing token-classification metrics.
```

### Phase 7: Architecture Refactor

Goal: reduce duplication and make the pipeline easier to maintain.

Tasks:

- Create a core pipeline module independent from UI frameworks.
- Move shared inference code out of `app.py`, `app_gui.py`, and `app_web.py`.
- Define clear data structures for:
  - pipeline config,
  - transcription result,
  - output artifact paths.
- Keep web and desktop applications as thin wrappers.
- Avoid broad behavior changes during refactor.

Acceptance criteria:

- Web and desktop flows call the same core inference function.
- Duplicated inference logic is reduced.
- Existing output behavior remains compatible.
- The refactor has focused tests or smoke checks.

Suggested prompt:

```text
Using docs/PROJECT_REQUIREMENTS.md and docs/IMPROVEMENTS_PLAN.md as references, implement Phase 7: Architecture Refactor. Extract a UI-independent core transcription pipeline and update the Flask and Tkinter entry points to call it without changing user-visible behavior.
```

### Phase 8: Testing and Quality Gates

Goal: reduce regression risk.

Tasks:

- Add unit tests for:
  - note normalization,
  - MIDI conversion,
  - token index loading,
  - pitch mask construction,
  - sustain collapse,
  - ASCII tab rendering.
- Add integration smoke test for one small synthetic or fixture WAV.
- Add static checks where practical.
- Add a simple command section in README for running tests.

Acceptance criteria:

- Tests can be run with one documented command.
- Core pure functions are covered.
- Integration test verifies expected artifacts are created.

Suggested prompt:

```text
Using docs/PROJECT_REQUIREMENTS.md and docs/IMPROVEMENTS_PLAN.md as references, implement Phase 8: Testing and Quality Gates. Add focused unit tests and a smoke integration test for the transcription pipeline, with documented test commands.
```

## Recommended Work Order

1. Phase 1: Reproducibility and Configuration
2. Phase 2: Timing-Preserving Note Representation
3. Phase 3: Better Note Segmentation
4. Phase 6: Evaluation Improvements
5. Phase 5: Sequence-Aware Tablature Optimization
6. Phase 7: Architecture Refactor
7. Phase 8: Testing and Quality Gates
8. Phase 4: Improved Pitch Detection Confidence

Phase 4 is listed last in the recommended order because pitch detection changes can alter many downstream results. It is safer after timing, segmentation, and evaluation are strong enough to measure the impact.

## Prompt Template for Future Tasks

Use this template when generating implementation prompts:

```text
You are working in the repository "Monophonic Tablature Transcription using YIN and LSTM".

Read these documents first:
- docs/PROJECT_REQUIREMENTS.md
- docs/IMPROVEMENTS_PLAN.md

Implement [PHASE NAME OR SPECIFIC TASK].

Constraints:
- Preserve current behavior unless explicitly changed by the task.
- Keep output artifacts backward compatible where possible.
- Avoid unrelated refactors.
- Add or update tests for changed behavior.
- Update README or docs if user-facing behavior changes.

Expected deliverables:
- Code changes.
- Tests or smoke checks.
- Short summary of changed files.
- Notes about any behavior changes or remaining limitations.
```

## Tracking Checklist

- [x] Phase 1: Reproducibility and Configuration
- [x] Phase 2: Timing-Preserving Note Representation
- [x] Phase 3: Better Note Segmentation
- [x] Phase 4: Improved Pitch Detection Confidence
- [x] Phase 5: Sequence-Aware Tablature Optimization
- [x] Phase 6: Evaluation Improvements
- [x] Phase 7: Architecture Refactor
- [x] Phase 8: Testing and Quality Gates
