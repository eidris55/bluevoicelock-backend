import streamlit as st
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import os
from scipy.spatial.distance import cosine
import tempfile
import soundfile as sf

st.set_page_config(page_title="BlueVoiceLock", layout="centered")

st.title("🔵 BlueVoiceLock")
st.subheader("VoIP Caller Authentication using Voiceprint AI")

file1 = st.file_uploader("Upload Verified Voiceprint", type=["wav", "mp3"])
file2 = st.file_uploader("Upload Incoming Caller Voice", type=["wav", "mp3"])

if file1 and file2:
    with st.spinner("Analyzing voice similarity..."):
        encoder = VoiceEncoder()

        with tempfile.NamedTemporaryFile(delete=False) as tmp1:
            tmp1.write(file1.read())
            path1 = tmp1.name

        with tempfile.NamedTemporaryFile(delete=False) as tmp2:
            tmp2.write(file2.read())
            path2 = tmp2.name

        wav1 = preprocess_wav(path1)
        wav2 = preprocess_wav(path2)
        def preprocess_wav(file_path):
            wav, sr = sf.read(file_path)
            return wav

        embed1 = encoder.embed_utterance(wav1)
        embed2 = encoder.embed_utterance(wav2)

        similarity = 1 - cosine(embed1, embed2)
        confidence = round(similarity * 100, 2)

        verdict = "✅ Legit Caller" if confidence > 75 else "🚨 Spoofed Call Detected"

        st.success(f"Match Confidence: {confidence}%")
        st.markdown(f"### Result: {verdict}")
