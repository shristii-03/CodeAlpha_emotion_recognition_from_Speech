# ─────────────────────────────────────────────
#  utils/feature_extraction.py
#  MFCC + augmentation helpers
# ─────────────────────────────────────────────

import numpy as np
import librosa
import soundfile as sf
from config import (
    SAMPLE_RATE, DURATION, N_MFCC, HOP_LENGTH, N_FFT,
    MAX_PAD_LEN, NOISE_FACTOR, PITCH_STEPS
)


# ── Load & normalise audio ────────────────────

def load_audio(path: str) -> np.ndarray:
    """Load audio, fix length, mono."""
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True,
                         duration=DURATION)
    # Pad if shorter than DURATION
    target_len = SAMPLE_RATE * DURATION
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


# ── Augmentation ─────────────────────────────

def add_noise(y: np.ndarray) -> np.ndarray:
    noise = np.random.randn(len(y)) * NOISE_FACTOR
    return y + noise

def pitch_shift(y: np.ndarray) -> np.ndarray:
    steps = np.random.choice([-PITCH_STEPS, PITCH_STEPS])
    return librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=steps)

def time_stretch(y: np.ndarray) -> np.ndarray:
    rate = np.random.uniform(0.85, 1.15)
    return librosa.effects.time_stretch(y, rate=rate)

def augment(y: np.ndarray) -> np.ndarray:
    """Randomly apply one augmentation."""
    choice = np.random.randint(0, 3)
    if choice == 0:
        return add_noise(y)
    elif choice == 1:
        return pitch_shift(y)
    else:
        return time_stretch(y)


# ── Feature extraction ────────────────────────

def extract_mfcc(y: np.ndarray, pad: bool = True) -> np.ndarray:
    """
    Returns MFCC of shape (MAX_PAD_LEN, N_MFCC) — ready for LSTM/CNN.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE,
                                 n_mfcc=N_MFCC,
                                 hop_length=HOP_LENGTH,
                                 n_fft=N_FFT)          # (N_MFCC, T)
    mfcc = mfcc.T                                       # (T, N_MFCC)

    if pad:
        if mfcc.shape[0] < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))
        else:
            mfcc = mfcc[:MAX_PAD_LEN]

    # Normalise per-file
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
    return mfcc   # (MAX_PAD_LEN, N_MFCC)


def extract_features(path: str, augment_data: bool = False) -> np.ndarray:
    """Full pipeline: load → (augment) → MFCC."""
    y = load_audio(path)
    if augment_data:
        y = augment(y)
    return extract_mfcc(y)
