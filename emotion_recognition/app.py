# ─────────────────────────────────────────────
#  app.py  |  Emotion Recognition — Streamlit UI
#
#  Usage:
#    streamlit run app.py
# ─────────────────────────────────────────────

import os
import sys
import numpy as np
import joblib
import tensorflow as tf
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import soundfile as sf
import tempfile

# ── Page config ───────────────────────────────
st.set_page_config(
    page_title="EmotiSense — Speech Emotion AI",
    page_icon="🎙",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Paths ─────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")
ENC_PATH   = os.path.join(BASE_DIR, "outputs", "label_encoder.pkl")

sys.path.insert(0, BASE_DIR)
from config import SAMPLE_RATE, DURATION, N_MFCC, HOP_LENGTH, N_FFT, MAX_PAD_LEN
from utils.feature_extraction import extract_features

# ── Emotion metadata ──────────────────────────
EMOTION_META = {
    "neutral":   {"emoji": "😐", "color": "#94a3b8", "desc": "Calm, expressionless tone"},
    "calm":      {"emoji": "😌", "color": "#67e8f9", "desc": "Relaxed and composed"},
    "happy":     {"emoji": "😄", "color": "#fde047", "desc": "Joyful and upbeat"},
    "sad":       {"emoji": "😢", "color": "#818cf8", "desc": "Sorrowful and low energy"},
    "angry":     {"emoji": "😠", "color": "#f87171", "desc": "Intense and aggressive"},
    "fearful":   {"emoji": "😨", "color": "#fb923c", "desc": "Anxious and tense"},
    "disgust":   {"emoji": "🤢", "color": "#86efac", "desc": "Revolted and averse"},
    "surprised": {"emoji": "😲", "color": "#c084fc", "desc": "Startled and unexpected"},
}

# ── Custom CSS ────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');

* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #080c14;
    color: #e2e8f0;
}

.stApp {
    background: radial-gradient(ellipse at 20% 20%, #0f1f3d 0%, #080c14 50%, #0a0e1a 100%);
    min-height: 100vh;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.5rem, 6vw, 5rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #e2e8f0 0%, #67e8f9 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}

.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #64748b;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

.upload-zone {
    border: 1.5px dashed #1e3a5f;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    background: rgba(15, 31, 61, 0.4);
    transition: border-color 0.3s;
}

.result-card {
    background: linear-gradient(135deg, rgba(15,31,61,0.8) 0%, rgba(10,14,26,0.9) 100%);
    border: 1px solid rgba(103, 232, 249, 0.15);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #67e8f9, #818cf8, #f87171);
}

.emotion-emoji {
    font-size: 5rem;
    line-height: 1;
    display: block;
    margin-bottom: 0.5rem;
}

.emotion-label {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.confidence-badge {
    display: inline-block;
    background: rgba(103, 232, 249, 0.1);
    border: 1px solid rgba(103, 232, 249, 0.3);
    border-radius: 999px;
    padding: 0.3rem 1rem;
    font-size: 0.8rem;
    color: #67e8f9;
    letter-spacing: 0.05em;
    margin-top: 0.5rem;
}

.bar-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.6rem;
}

.bar-label {
    width: 80px;
    font-size: 0.75rem;
    color: #94a3b8;
    text-align: right;
    flex-shrink: 0;
}

.bar-track {
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.05);
    border-radius: 999px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s ease;
}

.bar-pct {
    width: 45px;
    font-size: 0.75rem;
    color: #64748b;
    text-align: left;
    flex-shrink: 0;
}

.stat-chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    text-align: center;
}

.stat-chip .val {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #67e8f9;
}

.stat-chip .lbl {
    font-size: 0.65rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.1rem;
}

.section-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #475569;
    margin-bottom: 0.75rem;
}

div[data-testid="stFileUploader"] {
    background: rgba(15, 31, 61, 0.3);
    border: 1.5px dashed #1e3a5f;
    border-radius: 16px;
    padding: 1rem;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #67e8f9;
}

.stButton > button {
    background: linear-gradient(135deg, #1e40af, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.5rem !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

.stAudio {
    border-radius: 12px;
    overflow: hidden;
}

footer { display: none !important; }
#MainMenu { display: none !important; }
header { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model (cached) ───────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    le    = joblib.load(ENC_PATH)
    return model, le


# ── Predict ───────────────────────────────────
def predict_emotion(audio_path, model, le):
    feat  = extract_features(audio_path, augment_data=False)
    feat  = np.expand_dims(feat, 0)
    probs = model.predict(feat, verbose=0)[0]
    idx_sorted = np.argsort(probs)[::-1]
    results = [(le.classes_[i], float(probs[i])) for i in idx_sorted]
    return results


# ── Waveform plot ────────────────────────────
def plot_waveform(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
    fig, ax = plt.subplots(figsize=(8, 1.8))
    fig.patch.set_facecolor("#080c14")
    ax.set_facecolor("#080c14")
    times = np.linspace(0, len(y)/sr, len(y))
    ax.fill_between(times, y, alpha=0.6, color="#67e8f9")
    ax.plot(times, y, color="#67e8f9", linewidth=0.5, alpha=0.9)
    ax.axhline(0, color="#1e3a5f", linewidth=0.5)
    ax.set_xlim(0, times[-1])
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


# ── MFCC plot ────────────────────────────────
def plot_mfcc(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)
    fig, ax = plt.subplots(figsize=(8, 2.5))
    fig.patch.set_facecolor("#080c14")
    ax.set_facecolor("#080c14")
    img = librosa.display.specshow(mfcc, sr=sr, hop_length=HOP_LENGTH,
                                   x_axis="time", ax=ax, cmap="cool")
    ax.set_ylabel("MFCC", color="#64748b", fontsize=8)
    ax.tick_params(colors="#475569", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")
    fig.tight_layout(pad=0.3)
    return fig


# ── Audio stats ───────────────────────────────
def audio_stats(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration  = len(y) / sr
    rms       = float(np.sqrt(np.mean(y**2)))
    zcr       = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    tempo, _  = librosa.beat.beat_track(y=y, sr=sr)
    return {
        "Duration": f"{duration:.1f}s",
        "RMS Energy": f"{rms:.4f}",
        "Zero-Cross Rate": f"{zcr:.3f}",
        "Tempo": f"{float(np.atleast_1d(tempo)[0]):.0f} BPM",
    }


# ═══════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════

# Hero
st.markdown("""
<div style="padding: 2.5rem 0 1.5rem 0;">
    <p class="hero-sub">▸ Deep Learning · RAVDESS · CNN+LSTM</p>
    <h1 class="hero-title">EmotiSense</h1>
    <p style="color:#475569; font-size:0.85rem; margin-top:0.75rem;">
        Upload a speech audio file and the model will detect the underlying human emotion.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Load model
model, le = load_model()

if model is None:
    st.error("⚠️ Model not found. Run `python train.py` first to train the model.")
    st.stop()

# Layout
col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.markdown('<p class="section-label">▸ Input Audio</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a .wav file",
        type=["wav", "mp3", "ogg", "flac"],
        label_visibility="collapsed"
    )

    analyze_btn = st.button("⚡  Analyze Emotion", disabled=(uploaded is None))

    if uploaded:
        st.markdown('<p class="section-label" style="margin-top:1.5rem;">▸ Playback</p>', unsafe_allow_html=True)
        st.audio(uploaded, format="audio/wav")

        # Save temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.markdown('<p class="section-label" style="margin-top:1.5rem;">▸ Waveform</p>', unsafe_allow_html=True)
        st.pyplot(plot_waveform(tmp_path), use_container_width=True)

        st.markdown('<p class="section-label" style="margin-top:1rem;">▸ MFCC Features</p>', unsafe_allow_html=True)
        st.pyplot(plot_mfcc(tmp_path), use_container_width=True)

        # Stats
        st.markdown('<p class="section-label" style="margin-top:1rem;">▸ Audio Stats</p>', unsafe_allow_html=True)
        stats = audio_stats(tmp_path)
        s1, s2, s3, s4 = st.columns(4)
        for col, (k, v) in zip([s1, s2, s3, s4], stats.items()):
            col.markdown(f"""
            <div class="stat-chip">
                <div class="val">{v}</div>
                <div class="lbl">{k}</div>
            </div>""", unsafe_allow_html=True)


with col_right:
    st.markdown('<p class="section-label">▸ Prediction</p>', unsafe_allow_html=True)

    if uploaded and analyze_btn:
        with st.spinner("Analyzing speech patterns…"):
            results = predict_emotion(tmp_path, model, le)

        top_emotion, top_prob = results[0]
        meta = EMOTION_META.get(top_emotion, {"emoji": "🎙", "color": "#67e8f9", "desc": ""})

        # Result card
        st.markdown(f"""
        <div class="result-card">
            <span class="emotion-emoji">{meta['emoji']}</span>
            <div class="emotion-label" style="color:{meta['color']}">{top_emotion.upper()}</div>
            <div class="confidence-badge">
                {top_prob*100:.1f}% confidence
            </div>
            <p style="color:#64748b; font-size:0.78rem; margin-top:0.75rem; margin-bottom:0;">
                {meta['desc']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # All emotion bars
        st.markdown('<p class="section-label" style="margin-top:1.5rem;">▸ All Emotions</p>', unsafe_allow_html=True)

        bars_html = ""
        for emotion, prob in results:
            m = EMOTION_META.get(emotion, {"emoji": "•", "color": "#67e8f9"})
            pct = prob * 100
            bars_html += f"""
            <div class="bar-row">
                <div class="bar-label">{m['emoji']} {emotion}</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{pct:.1f}%; background:{m['color']};"></div>
                </div>
                <div class="bar-pct">{pct:.1f}%</div>
            </div>"""

        st.markdown(f'<div style="margin-top:0.5rem;">{bars_html}</div>', unsafe_allow_html=True)

        # Top-3 cards
        st.markdown('<p class="section-label" style="margin-top:1.5rem;">▸ Top 3 Candidates</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col, (emotion, prob) in zip([c1, c2, c3], results[:3]):
            m = EMOTION_META.get(emotion, {"emoji": "🎙", "color": "#67e8f9"})
            col.markdown(f"""
            <div class="stat-chip" style="border-color: {m['color']}22;">
                <div style="font-size:1.8rem;">{m['emoji']}</div>
                <div class="val" style="color:{m['color']}; font-size:0.9rem;">{emotion}</div>
                <div class="lbl">{prob*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        # Clean up
        os.unlink(tmp_path)

    elif not uploaded:
        st.markdown("""
        <div style="
            height: 400px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border: 1.5px dashed #1e3a5f;
            border-radius: 20px;
            color: #334155;
            font-size: 0.85rem;
            text-align: center;
            gap: 1rem;
        ">
            <span style="font-size: 3rem;">🎙</span>
            <div>Upload a .wav file<br>and click Analyze</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="margin-top:3rem; padding-top:1.5rem; border-top: 1px solid #0f1f3d;
     text-align:center; color:#334155; font-size:0.72rem; letter-spacing:0.05em;">
    CNN+LSTM · RAVDESS Dataset · MUJ B.Tech CS (AI/ML) · Final Year Project
</div>
""", unsafe_allow_html=True)
