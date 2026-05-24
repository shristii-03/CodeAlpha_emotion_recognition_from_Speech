# ─────────────────────────────────────────────
#  evaluate.py  |  Batch evaluation on a folder
#
#  Usage:
#    python evaluate.py --dir path/to/wav/folder
#    python evaluate.py --dir data/RAVDESS/Actor_01
# ─────────────────────────────────────────────

import argparse
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from config import MODEL_DIR, OUTPUT_DIR, EMOTIONS
from utils.feature_extraction import extract_features
from utils.dataset import parse_ravdess_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Folder with .wav files")
    args = parser.parse_args()

    # Load model
    ckpt = os.path.join(MODEL_DIR, "best_model.keras")
    enc  = os.path.join(OUTPUT_DIR, "label_encoder.pkl")
    model = tf.keras.models.load_model(ckpt)
    le    = joblib.load(enc)

    records = []
    audio_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(args.dir)
        for f in files if f.endswith(".wav")
    ]

    if not audio_files:
        print(f"No .wav files found in '{args.dir}'")
        return

    print(f"Evaluating {len(audio_files)} files…")

    for path in audio_files:
        fname    = os.path.basename(path)
        true_lbl = parse_ravdess_label(fname) or "unknown"

        feat  = extract_features(path, augment_data=False)
        feat  = np.expand_dims(feat, 0)
        probs = model.predict(feat, verbose=0)[0]
        pred_idx  = np.argmax(probs)
        pred_lbl  = le.classes_[pred_idx]
        confidence = probs[pred_idx]

        records.append({
            "file": fname,
            "true_emotion": true_lbl,
            "predicted_emotion": pred_lbl,
            "confidence": round(float(confidence), 4),
            "correct": true_lbl == pred_lbl,
        })

    df = pd.DataFrame(records)
    out_csv = os.path.join(OUTPUT_DIR, "batch_results.csv")
    df.to_csv(out_csv, index=False)

    acc = df["correct"].mean()
    print(f"\nOverall Accuracy : {acc*100:.2f}%")
    print(f"Results saved to : {out_csv}")
    print("\nPer-emotion breakdown:")
    print(df.groupby("true_emotion")["correct"].agg(["sum","count","mean"])
            .rename(columns={"sum":"correct","count":"total","mean":"accuracy"})
            .sort_values("accuracy", ascending=False)
            .to_string())


if __name__ == "__main__":
    main()
