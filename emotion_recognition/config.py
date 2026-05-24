# ─────────────────────────────────────────────
#  config.py  |  Emotion Recognition – RAVDESS
# ─────────────────────────────────────────────

import os

# ── Paths ────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data", "RAVDESS")   # put RAVDESS audio here
MODEL_DIR       = os.path.join(BASE_DIR, "models")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")

# ── RAVDESS Emotion Map ───────────────────────
# RAVDESS filename emotion code → label
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
NUM_CLASSES = len(EMOTION_MAP)           # 8
EMOTIONS    = list(EMOTION_MAP.values()) # ordered list

# ── Audio / Feature Settings ──────────────────
SAMPLE_RATE     = 22050
DURATION        = 3          # seconds to crop / pad each clip
N_MFCC          = 40         # number of MFCC coefficients
N_MELS          = 128
HOP_LENGTH      = 512
N_FFT           = 2048
MAX_PAD_LEN     = 130        # time-steps after fixed-length padding

# ── Augmentation ──────────────────────────────
AUGMENT         = True
NOISE_FACTOR    = 0.005
PITCH_STEPS     = 2          # semitones for pitch shift

# ── Model ─────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 60
LEARNING_RATE   = 1e-3
DROPOUT_RATE    = 0.4
L2_REG          = 1e-4
PATIENCE        = 12         # early-stopping patience

# ── CNN branch ────────────────────────────────
CNN_FILTERS     = [64, 128, 256]
KERNEL_SIZE     = 3

# ── LSTM branch ───────────────────────────────
LSTM_UNITS      = [128, 64]

# ── Misc ──────────────────────────────────────
RANDOM_SEED     = 42
TEST_SPLIT      = 0.20
VAL_SPLIT       = 0.10
