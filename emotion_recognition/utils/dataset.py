# ─────────────────────────────────────────────
#  utils/dataset.py
#  RAVDESS loader → numpy arrays
# ─────────────────────────────────────────────

import os
import sys
import numpy as np
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, EMOTION_MAP, AUGMENT
from utils.feature_extraction import extract_features


def parse_ravdess_label(filename: str) -> str | None:
    """
    RAVDESS filename format:
      03-01-06-01-02-01-12.wav
      └─ modality-vocal_channel-emotion-intensity-statement-repetition-actor
    emotion code is the 3rd field (index 2).
    """
    parts = os.path.splitext(filename)[0].split("-")
    if len(parts) < 3:
        return None
    emotion_code = parts[2]
    return EMOTION_MAP.get(emotion_code, None)


def load_ravdess(data_dir: str = DATA_DIR,
                 augment: bool = AUGMENT):
    """
    Walk RAVDESS directory (organized as Actor_01/, Actor_02/, …).
    Returns X: (N, MAX_PAD_LEN, N_MFCC), y: (N,) int labels.
    """
    X, y, label_names = [], [], []
    emotion_to_idx = {e: i for i, e in enumerate(EMOTION_MAP.values())}

    audio_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".wav"):
                audio_files.append(os.path.join(root, f))

    if not audio_files:
        raise FileNotFoundError(
            f"No .wav files found in '{data_dir}'.\n"
            "Download RAVDESS from https://zenodo.org/record/1188976 "
            "and place Actor_* folders inside data/RAVDESS/."
        )

    print(f"[Dataset] Found {len(audio_files)} audio files.")

    for path in tqdm(audio_files, desc="Extracting features"):
        fname   = os.path.basename(path)
        emotion = parse_ravdess_label(fname)
        if emotion is None:
            continue

        # Original sample
        feat = extract_features(path, augment_data=False)
        X.append(feat)
        y.append(emotion_to_idx[emotion])

        # Augmented copy
        if augment:
            feat_aug = extract_features(path, augment_data=True)
            X.append(feat_aug)
            y.append(emotion_to_idx[emotion])

    X = np.array(X, dtype=np.float32)   # (N, T, F)
    y = np.array(y, dtype=np.int32)     # (N,)
    print(f"[Dataset] X shape: {X.shape} | y shape: {y.shape}")
    return X, y
