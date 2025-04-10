import streamlit as st
import numpy as np
import soundfile as sf
from resemblyzer import VoiceEncoder
from scipy.spatial.distance import cosine
import tempfile

st.set_page_config(page_title="BlueVoiceLock", layout="centered")

st.title("🔵 BlueVoiceLock")
st.subheader("VoIP Caller Authentication using Voiceprint AI")

file1 = st.file_uploader("Upload Verified Voiceprint", type=["wav"])
file2 = st.file_uploader("Upload Incoming Caller Voice", type=["wav"])

# 👉 Custom audio loader using soundfile
def load_audio(file_path):
    wav, sr = sf.read(file_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # Convert stereo to mono
    return wav

if file1 and file2:
    with st.spinner("Analyzing voice similarity..."):
        encoder = VoiceEncoder()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp1:
            tmp1.write(file1.read())
            path1 = tmp1.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp2:
            tmp2.write(file2.read())
            path2 = tmp2.name

        # Use custom loader instead of preprocess_wav
        wav1 = load_audio(path1)
        wav2 = load_audio(path2)

        embed1 = encoder.embed_utterance(wav1)
        embed2 = encoder.embed_utterance(wav2)

        similarity = 1 - cosine(embed1, embed2)
        confidence = round(similarity * 100, 2)

        verdict = "✅ Legit Caller" if confidence > 75 else "🚨 Spoofed Call Detected"

        st.success(f"Match Confidence: {confidence}%")
        st.markdown(f"### Result: {verdict}")
