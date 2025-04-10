# bluevoicelock-backend
# BlueVoiceLock

**VoIP Caller Authentication using Voiceprint AI**

This project uses machine learning to detect spoofed VoIP calls in real time by comparing voiceprints.

## How It Works

- Upload a verified voice sample (WAV format)
- Upload the incoming caller's voice
- The system compares them using voiceprint embeddings
- Output: Match confidence score and spoofing alert

## Tech Stack

- Python
- Streamlit (Web interface)
- Resemblyzer (Voice embedding)
- Soundfile & SciPy (Audio processing)

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
