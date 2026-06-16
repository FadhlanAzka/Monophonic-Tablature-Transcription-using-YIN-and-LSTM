# Project Requirements Document

## Project Title

Monophonic Tablature Transcription using YIN and LSTM

## Purpose

This project provides an application for converting monophonic guitar audio into guitar tablature. It combines signal processing for pitch extraction with an LSTM-based mapper that predicts playable guitar string and fret positions from detected MIDI notes.

The system is intended to help users upload or select a WAV recording of a monophonic guitar melody and receive practical transcription outputs: CSV note data, ASCII tablature, MIDI, rendered audio, and visual diagnostics.

## Problem Statement

Manual guitar tablature transcription is time-consuming, especially when the user needs both the pitch sequence and a playable string-fret representation. Standard pitch detection can identify notes, but tablature requires deciding where each pitch should be played on the guitar. The same pitch may appear on multiple string-fret combinations, so the project needs both pitch estimation and tablature mapping.

## Goals

- Detect monophonic guitar pitch from WAV audio.
- Convert detected pitch into stable musical notes and MIDI values.
- Map MIDI note sequences into guitar string and fret positions.
- Generate human-readable tablature.
- Export machine-readable and playback-friendly artifacts.
- Provide both local and web-based interfaces.
- Support evaluation of the LSTM tablature mapper.

## Non-Goals

- Polyphonic chord transcription.
- Full expressive guitar technique detection such as bends, slides, hammer-ons, pull-offs, vibrato labeling, or palm muting.
- Real-time streaming transcription.
- Automatic model training from the application UI.
- General-purpose transcription for non-guitar instruments.

## Target Users

- Students or researchers working on automatic music transcription.
- Guitar learners who want a starting tablature from simple monophonic recordings.
- Developers evaluating YIN-based pitch tracking and LSTM-based tablature mapping.
- Thesis reviewers or supervisors inspecting the pipeline and output artifacts.

## Current Functional Requirements

### Audio Input

- The system accepts WAV audio files.
- Audio is loaded as mono by default using `librosa`.
- The current web interface uploads WAV files through a browser form.
- The local scripts use Tkinter file dialogs.

### Pitch Detection Pipeline

The core pitch pipeline must:

- Apply HPSS and use the harmonic component for pitch tracking.
- Estimate fundamental frequency using YIN.
- Restrict pitch detection to the configured guitar range.
- Remove low-energy frames using RMS-based filtering.
- Apply post-processing to reduce octave errors.
- Sample pitch in fixed time blocks.
- Stabilize notes using cent-based pitch continuity logic.
- Normalize note names to sharp-only ASCII notation.

### Tablature Mapping

The mapping stage must:

- Convert stabilized notes into MIDI values.
- Remove invalid or silent frames before LSTM mapping.
- Collapse consecutive repeated MIDI values into sustained notes.
- Load a TorchScript or compatible PyTorch model.
- Load a token index CSV containing `token_idx`, `string`, `fret`, and optionally `midi`.
- Predict token indices from MIDI note sequences.
- Map token predictions back to string and fret.
- Apply a pitch mask when token MIDI values are available, so impossible string-fret predictions are excluded.

### Output Artifacts

For each successful run, the system should produce:

- CSV output containing at least `hz`, `note`, `midi`, `string`, `fret`, and `token_idx`.
- ASCII tablature text.
- MIDI output generated from the CSV note sequence.
- Optional WAV render of the MIDI output using FluidSynth and a SoundFont.
- Visualization image containing pitch and timing diagnostics.
- Manifest JSON describing generated artifacts for web display.

### Web Application

The web application must:

- Serve an upload page.
- Accept WAV upload requests.
- Create a unique result folder per run.
- Run inference using the configured token index and model.
- Redirect to a result page.
- Provide links or previews for generated artifacts.
- Expose a health endpoint.

### Desktop Application

The desktop application should:

- Allow users to choose input WAV, token index, model, and output location.
- Run the same inference logic as the core pipeline.
- Save generated artifacts locally.

### Evaluation

The evaluation tools should:

- Load model artifacts and token datasets.
- Support validation evaluation using a file-wise split.
- Support test-set evaluation from a separate folder.
- Report accuracy, precision, recall, and F1 metrics.
- Save summary JSON and metric plots.
- Optionally save TP, FP, and FN counts for manual analysis.

## Current Non-Functional Requirements

### Usability

- Web interface should be simple enough for non-technical users to upload a file and download results.
- Output files should use predictable names derived from the input WAV filename.

### Portability

- The project currently targets a Windows development environment.
- Future versions should reduce hard-coded local paths.

### Maintainability

- Core pipeline behavior should be reusable across CLI, desktop GUI, and web UI.
- Configuration should be centralized.
- Signal processing, model inference, rendering, and UI code should remain separable.

### Reliability

- Invalid numeric values in audio should be sanitized.
- Empty or silent audio should fail gracefully and still produce predictable output where possible.
- Missing model, token index, SoundFont, or FluidSynth should produce clear warnings or errors.

### Performance

- The pipeline is batch-oriented, not real-time.
- The application should handle short to medium monophonic WAV files without excessive memory usage.
- CUDA may be used when available, with CPU fallback.

## Existing System Overview

### Main Files

- `app.py`: shared pipeline logic and Tkinter-based workflow.
- `app_gui.py`: desktop GUI implementation.
- `app_web.py`: Flask web application.
- `audio_loader.py`: audio loading helper.
- `settings.py`: global pitch and preprocessing constants.
- `tab_midi.py`: CSV-to-MIDI conversion and MIDI writing.
- `analysis/`: pitch analysis, YIN, post-processing, onset, beat, and stabilization modules.
- `preprocessing/`: HPSS, bandpass, and HTDemucs helpers.
- `evaluation/`: model evaluation and music theory utilities.
- `tokens/`: token index CSV files.
- `models/`: TorchScript model artifacts.
- `templates/`: Flask HTML templates.
- `static/results/`: generated web outputs.

### Default Web Configuration

- Token index: `tokens/token index v2.csv`
- Model: `models/optuna.jit`
- Results directory: `static/results`
- Web server: Flask on host `0.0.0.0`, port `5000`

## Current Constraints

- Only monophonic transcription is in scope.
- Current MIDI generation uses a fixed note rate by default rather than detected note durations.
- Current CSV output does not preserve full timing information for each note.
- FluidSynth path is hard-coded for one Windows installation.
- There is no dependency lock file or formal install guide.
- There is no training script in the current repository snapshot.
- Some code comments contain encoding artifacts and should be normalized to UTF-8.

## Success Criteria

The project is successful when:

- A user can run the web app and upload a WAV file.
- The system generates CSV, tablature, MIDI, visualization, and available audio render artifacts.
- The output notes are musically plausible for monophonic guitar melodies.
- The predicted tablature positions are playable on a standard tuned six-string guitar.
- Evaluation scripts can measure model quality on validation and test datasets.
- The project can be installed and reproduced on another machine with documented steps.

## Risks

- YIN pitch detection may produce octave errors or unstable estimates on noisy input.
- Fixed block sampling can blur note boundaries.
- Repeated-note collapse may remove timing information needed for accurate MIDI.
- Token accuracy alone may overstate musical quality.
- Hard-coded external tool paths reduce portability.
- LSTM predictions may be locally valid but physically awkward across a sequence.

## Assumptions

- Input audio contains a single dominant guitar melody line.
- Standard guitar tuning is assumed.
- Token index rows represent valid string-fret positions.
- The trained model accepts MIDI integer sequences and outputs token logits.
- The user has required local dependencies installed.
