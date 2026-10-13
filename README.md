---
title: Vocal Typewriter Parakeet V3
emoji: 🦜
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# Parakeet V3 Standalone App (Python)

This is a standalone Python application for **Parakeet TDT 0.6b** speech recognition, designed to run locally or on Hugging Face Spaces.

## Features
- **Offline Speech Recognition**: Uses `sherpa-onnx` with Parakeet TDT (Offline Transducer).
- **Real-time Streaming**: Simulates streaming via WebSocket and VAD (Voice Activity Detection).
- **Raw Mode Only**: Direct speech-to-text transcription without LLM processing.
- **Waveform Visualization**: Real-time audio feedback.

## Setup & Run

### Prerequisites
- Python 3.9+
- `ffmpeg` (for audio conversion)

### Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download models:
   ```bash
   chmod +x setup_models.sh
   ./setup_models.sh
   ```

### Running Locally
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```
Open [http://localhost:7860](http://localhost:7860) in your browser.

## Troubleshooting & Metadata Fixes

If you encounter errors like `window_size does not exist in metadata` or dimension mismatches, here is how we fixed them:

### 1. Missing Metadata in ONNX Models
The original Parakeet ONNX export was missing critical metadata required by `sherpa-onnx`. We created a script `fix_metadata.py` to inject these values:

- **vocab_size**: Injected from `vocab.txt` (8197 tokens).
- **context_size**: Set to 2.
- **subsampling_factor**: Set to 8.
- **pred_rnn_layers**: Set to 2 (derived from model inspection).
- **pred_hidden**: Set to 640.

### 2. Feature Dimension Mismatch (80 vs 128)
Error: `Got: 80 Expected: 128`
- **Cause**: Default feature extractor uses 80 mel bins, but Parakeet TDT expects 128.
- **Fix**: Initialized `OfflineRecognizer` with explicit `feature_dim=128`.

```python
recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    ...
    feature_dim=128,
    model_type="nemo_transducer",
    ...
)
```

### 3. VAD Configuration
Error: `AttributeError: 'SileroVadModelConfig' object has no attribute 'model_file'`
- **Fix**: Changed attribute name to `model` in VAD config.

### 4. WebSocket Communication
- **Issue**: Frontend sends binary audio, but server expected text config or vice versa.
- **Fix**: Unified WebSocket loop to handle both binary (audio) and text (ignored/legacy) messages, and sending JSON responses with transcript and waveform data.
