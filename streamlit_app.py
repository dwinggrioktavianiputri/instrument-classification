import streamlit as st
import librosa
import numpy as np
import joblib
import tensorflow as tf
import io
import matplotlib.pyplot as plt

# Load model dan label encoder
instrument_model = tf.keras.models.load_model("gru_music_model.h5")
label_encoder = joblib.load("label_encoder.pkl")  # hasil training

label_classes = ["Baritone", "Mellophone", "Trumpet", "Tuba"]

# ===================== FUNGSI ===================== #
def extract_features_from_audio_bytes(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True, duration=10)

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    features = [
        np.mean(spec_bw), np.var(spec_bw),
        np.mean(zcr), np.var(zcr),
        np.mean(rolloff), np.var(rolloff),
        1 if np.mean(y_harmonic) > np.mean(y_percussive) else 0
    ]
    features += list(np.mean(mfcc, axis=1))
    features += list(np.mean(spectrogram, axis=1))
    features += list(np.mean(chroma, axis=1))
    features += list(np.mean(contrast, axis=1))

    return np.array(features, dtype=np.float32), y, sr

def detect_chords(y, sr, hop_length=None, max_chords=10):
    if hop_length is None:
        n_frames_desired = 150
        hop_length = max(2048, int(len(y) / n_frames_desired))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    chords_detected = []
    unique_chords = set()

    for i in range(chroma.shape[1]):
        chroma_frame = chroma[:, i]
        dominant_idx = np.argmax(chroma_frame)
        if chroma_frame[dominant_idx] > 0.2:
            chord_name = f"{notes[dominant_idx]}"
        else:
            chord_name = None

        if chord_name and chord_name not in unique_chords:
            time_sec = float(librosa.frames_to_time(i, sr=sr, hop_length=hop_length))
            chords_detected.append({"time": round(time_sec, 2), "chord": chord_name})
            unique_chords.add(chord_name)

            if len(unique_chords) >= max_chords:
                break

    return chords_detected

# ================= STREAMLIT APP ================== #
st.set_page_config(page_title="Klasifikasi Instrumen & Chord", layout="centered")

# --- HEADER ---
st.markdown(
    """
    <div style="background-color:#4CAF50;padding:15px;border-radius:10px">
        <h1 style="color:white;text-align:center;">🎺 Klasifikasi Instrumen Musik Tiup + Deteksi Chord 🎼</h1>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload file audio (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Ekstraksi fitur
    features, y, sr = extract_features_from_audio_bytes(uploaded_file.read())
    X = features.reshape(1, 1, -1)

    # Prediksi instrumen
    prediction = instrument_model.predict(X)
    pred_idx = np.argmax(prediction, axis=1)[0]
    instrument_name = label_classes[pred_idx]

    # --- HASIL KLASIFIKASI (CARD STYLE) ---
    st.markdown(
        f"""
        <div style="background-color:#2196F3;padding:20px;border-radius:10px;margin:20px 0;">
            <h2 style="color:white;text-align:center;">Instrumen Terdeteksi</h2>
            <h1 style="color:white;text-align:center;">{instrument_name}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- DETEKSI CHORD ---
    st.subheader("🎼 Chord yang Terdeteksi (List)")
    chords = detect_chords(y, sr)
    if chords:
        for c in chords:
            st.write(f"⏱ {c['time']}s → {c['chord']}")
    else:
        st.write("Tidak ada chord yang terdeteksi.")

    # --- CHART DETEKSI CHORD ---
    st.subheader("📊 Visualisasi Chord (Chart)")

    if chords:
        times = [c["time"] for c in chords]
        labels = [c["chord"] for c in chords]

        fig, ax = plt.subplots(figsize=(10, 5))

        # Garis penghubung antar chord
        ax.plot(times, range(len(times)), marker="o", linestyle="-", linewidth=2, color="royalblue")

        # Scatter dengan warna berbeda
        scatter = ax.scatter(times, range(len(times)), c=range(len(times)),
                            cmap="plasma", s=120, edgecolor="black", zorder=3)

        # Tambahkan label chord di titiknya
        for i, label in enumerate(labels):
            ax.text(times[i], i, f" {label}", va="center", fontsize=10, fontweight="bold")

        # Styling
        ax.set_xlabel("⏱ Waktu (detik)", fontsize=12)
        ax.set_ylabel("Urutan Chord", fontsize=12)
        ax.set_title("🎼 Perubahan Chord dari Waktu ke Waktu", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)

        # Warna bar legend
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Urutan Chord")

        st.pyplot(fig)
    else:
        st.info("Tidak ada chord yang jelas terdeteksi.")


# --- FOOTER ---
st.markdown(
    """
    <hr>
    <div style="text-align:center;color:pink;">
        Dwinggrit Oktaviani Putri
    </div>
    """,
    unsafe_allow_html=True
)
