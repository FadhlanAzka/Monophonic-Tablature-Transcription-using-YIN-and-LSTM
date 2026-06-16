# ================== GLOBAL SETTINGS ==================
import os
import librosa

HOP_LENGTH   = 512
FRAME_LENGTH = 2048
FMIN         = librosa.note_to_hz('E2')   # ~82.41 Hz
FMAX         = librosa.note_to_hz('D6')   # ~1318.51 Hz
#FMAX         = librosa.note_to_hz('E6')   # ~1318.51 Hz

# Bandpass default (mengikuti range gitar)
BANDPASS_LOW  = FMIN
BANDPASS_HIGH = FMAX

# HTDemucs
USE_TWO_STEMS = True       # True: vocals/no_vocals; False: 4 stems
DEMUCS_MODEL  = "htdemucs" # default CLI model name

# Matplotlib figure layout
FIG_SIZE = (14, 8)
UPDATE_INTERVAL_MS = 50  # animator ~20 FPS

# Pitch detection
# "yin" preserves the historical pipeline. "pyin" enables librosa.pyin with
# voiced probability diagnostics.
PITCH_DETECTION_MODE = os.getenv("YINTAB_PITCH_MODE", "yin").strip().lower()
try:
    PITCH_CONFIDENCE_THRESHOLD = float(os.getenv("YINTAB_PITCH_CONFIDENCE_THRESHOLD", "0.0"))
except ValueError:
    PITCH_CONFIDENCE_THRESHOLD = 0.0

# Tablature sequence optimization
TAB_OPTIMIZER_ENABLED = os.getenv("YINTAB_TAB_OPTIMIZER", "0").strip().lower() in {"1", "true", "yes", "on"}

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default

TAB_COST_FRET_DISTANCE = _env_float("YINTAB_TAB_COST_FRET_DISTANCE", 0.15)
TAB_COST_STRING_CHANGE = _env_float("YINTAB_TAB_COST_STRING_CHANGE", 0.25)
TAB_COST_LARGE_SHIFT = _env_float("YINTAB_TAB_COST_LARGE_SHIFT", 0.5)
TAB_LARGE_SHIFT_THRESHOLD = _env_int("YINTAB_TAB_LARGE_SHIFT_THRESHOLD", 5)
TAB_COST_OPEN_STRING = _env_float("YINTAB_TAB_COST_OPEN_STRING", 0.0)
TAB_COST_POSITION = _env_float("YINTAB_TAB_COST_POSITION", 0.0)
TAB_PREFERRED_FRET = _env_float("YINTAB_TAB_PREFERRED_FRET", 5.0)
