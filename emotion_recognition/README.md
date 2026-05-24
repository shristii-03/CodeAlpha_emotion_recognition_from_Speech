# 🎙 EmotiSense — Emotion Recognition from Speech

> **CNN+LSTM Hybrid · RAVDESS Dataset · TensorFlow · Streamlit**  
---

## 📊 Results at a Glance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **80.49%** |
| **Best Val Accuracy** | 83.25% (Epoch 31) |
| **Dataset** | RAVDESS (24 actors, 6 emotions) |
| **Model** | CNN + LSTM Hybrid |
| **Features** | 40 MFCCs × 130 time-steps |

---

## 🧠 Model Architecture

```
Input: MFCC (130 × 40)
         │
    ┌────┴────────┐
    │             │
[CNN Branch]   [LSTM Branch]
Conv1D(64)     LSTM(128, return_seq=True)
Conv1D(128)    LSTM(64)
Conv1D(256)    Dense(64)
GlobalAvgPool
Dense(128)
    │             │
    └────┬────────┘
    Concatenate (192-d)
      Dense(128, relu)
      Dropout(0.4)
      Dense(6, softmax)
```

**Why CNN + LSTM?**
- **CNN** → captures local spectral texture from MFCCs (formant shapes, energy patterns)
- **LSTM** → models how those patterns evolve over time (emotion dynamics)
- Together they outperform either branch alone on speech emotion tasks

---

## 🗂 Project Structure

```
emotion_recognition/
│
├── app.py                     # 🌐 Streamlit web UI
├── config.py                  # ⚙️  All hyperparameters & paths
├── model.py                   # 🧠 CNN+LSTM architecture
├── train.py                   # 🏋️  Training pipeline
├── predict.py                 # 🔍 CLI inference
├── evaluate.py                # 📊 Batch evaluation
├── requirements.txt
│
├── utils/
│   ├── feature_extraction.py  # MFCC extraction + augmentation
│   └── dataset.py             # RAVDESS loader
│
├── data/
│   └── RAVDESS/               # ← Actor_01/ … Actor_24/ goes here
│
├── models/
│   └── best_model.keras       # Saved after training
│
└── outputs/
    ├── label_encoder.pkl
    ├── history.png
    ├── confusion_matrix.png
    └── classification_report.txt
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/emotion-recognition.git
cd emotion-recognition
pip install -r requirements.txt
pip install streamlit          # for the web UI
```

### 2. Download RAVDESS Dataset

Go to 👉 [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)

Download **Audio_Speech_Actors_01-24.zip**, extract it, and place the Actor folders:

```
data/
  RAVDESS/
    Actor_01/
    Actor_02/
    ...
    Actor_24/
```

### 3. Train

```bash
python train.py
```

Trains for up to 60 epochs with early stopping. Best model saved to `models/best_model.keras`.

### 4. Launch Web UI

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` — drag and drop any `.wav` file.

### 5. CLI Prediction

```bash
python predict.py --audio path/to/speech.wav --top 3
```

```
Top-3 Predictions:
──────────────────────────────
  1. fearful       72.3%  ██████████████████████
  2. angry         18.5%  █████
  3. sad            6.1%  ██

🎙  Predicted Emotion: FEARFUL  (72.3% confidence)
```

### 6. Batch Evaluate

```bash
python evaluate.py --dir data/RAVDESS/Actor_01
```

---

## 🎭 Emotion Classes

| Code | Emotion | Emoji |
|------|---------|-------|
| 01 | Neutral | 😐 |
| 02 | Calm | 😌 |
| 03 | Happy | 😄 |
| 04 | Sad | 😢 |
| 05 | Angry | 😠 |
| 06 | Fearful | 😨 |

---

## 📈 Classification Report

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Neutral | 0.97 | 0.92 | 0.94 | 37 |
| Calm | 0.76 | 1.00 | 0.87 | 74 |
| Happy | 0.96 | 0.63 | 0.76 | 73 |
| Sad | 0.65 | 0.96 | 0.77 | 74 |
| Angry | 0.97 | 0.76 | 0.85 | 74 |
| Fearful | 0.79 | 0.62 | 0.69 | 73 |
| **Overall** | **0.84** | **0.80** | **0.80** | **405** |

---

## ⚙️ Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| `N_MFCC` | 40 |
| `MAX_PAD_LEN` | 130 time-steps |
| `SAMPLE_RATE` | 22,050 Hz |
| `DURATION` | 3 seconds |
| `BATCH_SIZE` | 32 |
| `EPOCHS` | 60 (early stop patience=12) |
| `LEARNING_RATE` | 1e-3 (Adam) |
| `DROPOUT_RATE` | 0.4 |
| `AUGMENT` | True (noise + pitch shift + time stretch) |

---

## 🔧 Data Augmentation

Each training sample is doubled with a randomly applied augmentation:

- **Gaussian Noise** — adds small random noise (factor: 0.005)
- **Pitch Shift** — shifts pitch ±2 semitones
- **Time Stretch** — stretches/compresses speed (rate: 0.85–1.15×)

---

## 📦 Dependencies

```
tensorflow>=2.12.0
librosa>=0.10.0
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
soundfile>=0.12.1
tqdm>=4.65.0
joblib>=1.2.0
streamlit
```

---

## 🌐 Web UI Features

The Streamlit app (`app.py`) includes:

- **Drag & drop** `.wav` / `.mp3` / `.ogg` / `.flac` files
- **Audio playback** in browser
- **Waveform visualization**
- **MFCC heatmap** — what the model actually sees
- **Audio stats** — duration, RMS energy, zero-crossing rate, tempo
- **Prediction card** with emotion emoji + confidence %
- **All-emotions probability bars**
- **Top-3 candidates** side by side

---

## 👤 Author

**Shristi Gupta** 