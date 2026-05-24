# ─────────────────────────────────────────────
#  train.py  |  Full training pipeline
#
#  Usage:
#    python train.py
#
#  Outputs saved to outputs/:
#    - best_model.keras       (best checkpoint)
#    - label_encoder.pkl      (class index → name)
#    - history.png            (loss / accuracy curves)
#    - confusion_matrix.png
#    - classification_report.txt
# ─────────────────────────────────────────────

import os
import sys
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    MODEL_DIR, OUTPUT_DIR, EMOTIONS, NUM_CLASSES,
    BATCH_SIZE, EPOCHS, PATIENCE,
    TEST_SPLIT, VAL_SPLIT, RANDOM_SEED
)
from model import build_cnn_lstm
from utils.dataset import load_ravdess

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ── 1. Load data ──────────────────────────────

print("\n=== Loading RAVDESS dataset ===")
X, y = load_ravdess()

# ── 2. Train / val / test split ───────────────

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, stratify=y, random_state=RANDOM_SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=VAL_SPLIT / (1 - TEST_SPLIT),
    stratify=y_trainval, random_state=RANDOM_SEED
)

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# Save label encoder (int → emotion name)
le = LabelEncoder()
le.fit(list(range(NUM_CLASSES)))
le.classes_ = np.array(EMOTIONS)
joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

# ── 3. Build model ────────────────────────────

print("\n=== Building CNN-LSTM model ===")
model = build_cnn_lstm()
model.summary()

# ── 4. Callbacks ──────────────────────────────

ckpt_path = os.path.join(MODEL_DIR, "best_model.keras")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_accuracy",
        save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=PATIENCE,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=5, min_lr=1e-6, verbose=1
    ),
    tf.keras.callbacks.CSVLogger(
        os.path.join(OUTPUT_DIR, "training_log.csv")
    ),
]

# ── 5. Train ──────────────────────────────────

print("\n=== Training ===")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

# ── 6. Evaluate on test set ───────────────────

print("\n=== Evaluation on test set ===")
model.load_weights(ckpt_path)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)

present_labels = sorted(set(y_test))
present_names = [EMOTIONS[i] for i in present_labels]
report = classification_report(y_test, y_pred, labels=present_labels, target_names=present_names)
print(report)
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Test Accuracy: {acc:.4f}\n\n")
    f.write(report)

# ── 7. Plots ──────────────────────────────────

# Loss / Accuracy curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training History — CNN+LSTM Emotion Recognition", fontsize=13)

axes[0].plot(history.history["loss"],     label="Train Loss",  color="#e74c3c")
axes[0].plot(history.history["val_loss"], label="Val Loss",    color="#e74c3c", linestyle="--")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

axes[1].plot(history.history["accuracy"],     label="Train Acc",  color="#2ecc71")
axes[1].plot(history.history["val_accuracy"], label="Val Acc",    color="#2ecc71", linestyle="--")
axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("Epoch")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "history.png"), dpi=150)
plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax)
ax.set_title("Confusion Matrix — Test Set")
ax.set_ylabel("True"); ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

print(f"\n✅ All outputs saved to '{OUTPUT_DIR}/'")
print(f"   Model saved to '{ckpt_path}'")
