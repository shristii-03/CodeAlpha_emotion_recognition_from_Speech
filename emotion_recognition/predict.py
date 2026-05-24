# ─────────────────────────────────────────────
#  predict.py  |  Inference on a single audio file
#
#  Usage:
#    python predict.py --audio path/to/speech.wav
#    python predict.py --audio path/to/speech.wav --top 3
# ─────────────────────────────────────────────

import argparse
import os
import sys
import numpy as np
import joblib
import tensorflow as tf

from config import MODEL_DIR, OUTPUT_DIR, EMOTIONS
from utils.feature_extraction import extract_features


# ── Helpers ───────────────────────────────────

def load_model_and_encoder():
    ckpt = os.path.join(MODEL_DIR, "best_model.keras")
    enc  = os.path.join(OUTPUT_DIR, "label_encoder.pkl")

    if not os.path.exists(ckpt):
        sys.exit(f"[Error] Model not found at '{ckpt}'. Run train.py first.")
    if not os.path.exists(enc):
        sys.exit(f"[Error] Label encoder not found at '{enc}'. Run train.py first.")

    model = tf.keras.models.load_model(ckpt)
    le    = joblib.load(enc)
    return model, le


def predict(audio_path: str, model, le, top_k: int = 1):
    """
    Returns list of (emotion, probability) tuples, sorted by confidence.
    """
    if not os.path.exists(audio_path):
        sys.exit(f"[Error] File not found: '{audio_path}'")

    feat  = extract_features(audio_path, augment_data=False)
    feat  = np.expand_dims(feat, axis=0)          # (1, T, F)
    probs = model.predict(feat, verbose=0)[0]     # (NUM_CLASSES,)

    idx_sorted = np.argsort(probs)[::-1][:top_k]
    results = [(le.classes_[i], float(probs[i])) for i in idx_sorted]
    return results


# ── CLI ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict emotion from a speech audio file."
    )
    parser.add_argument("--audio", required=True,
                        help="Path to .wav audio file")
    parser.add_argument("--top", type=int, default=3,
                        help="Show top-K predictions (default: 3)")
    args = parser.parse_args()

    print("\n=== Emotion Recognition — Inference ===")
    print(f"Audio : {args.audio}")

    model, le = load_model_and_encoder()
    results   = predict(args.audio, model, le, top_k=args.top)

    print(f"\nTop-{args.top} Predictions:")
    print("─" * 30)
    for rank, (emotion, prob) in enumerate(results, 1):
        bar = "█" * int(prob * 30)
        print(f"  {rank}. {emotion:<12} {prob*100:5.1f}%  {bar}")
    print()

    top_emotion, top_prob = results[0]
    print(f"🎙  Predicted Emotion: {top_emotion.upper()}  ({top_prob*100:.1f}% confidence)")


if __name__ == "__main__":
    main()
