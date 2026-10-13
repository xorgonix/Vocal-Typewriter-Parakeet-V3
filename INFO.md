# API Documentation - Vocal Typewriter (Parakeet V3)

This standalone application provides a high-performance speech-to-text API using the Parakeet V3 model, designed for CPU execution.

## Base URL
When deployed to Hugging Face Spaces, the Base URL is  `Loomisgitarrist/Vocal-Typewriter-Parakeet-V3`
Locally, it runs on `http://localhost:7860`.

## Endpoints

### 1. WebSocket Streaming API
**Endpoint:** `/ws`  
**Method:** `GET` (WebSocket Upgrade)  
**Description:** Real-time bi-directional audio streaming and transcription.

#### Client -> Server Messages

**Audio Data:**
- Binary messages containing raw PCM audio samples.
- Format: 16-bit PCM, 16kHz, Mono.
- Chunk size: variable (server processes in ~2s segments).

**Configuration:**
- JSON message to switch modes.
```json
{
  "mode": "raw" | "llm"
}
```
- `raw`: Direct transcription.
- `llm`: Simulated post-processing (currently adds visual indicator).

#### Server -> Client Messages

**Transcription:**
```json
{
  "type": "transcript",
  "text": "Hello world",
  "mode": "raw"
}
```

**Speed/Latency Stats:**
```json
{
  "type": "speed",
  "wpm": 120,
  "latency_ms": 150
}
```

**Waveform Data (Visualization):**
```json
{
  "type": "wave",
  "value": 0.5 // Normalized amplitude (0.0 - 1.0)
}
```

### 2. REST API
**Endpoint:** `/transcribe`
**Method:** `POST`
**Content-Type:** `multipart/form-data`
**Description:** Upload an audio file for transcription.

**Parameters:**
- `file`: The audio file to transcribe (raw PCM 16kHz f32le or wav expected).

**Response:**
```json
{
  "text": "Transcription text here",
  "duration_ms": 1234
}
```

### 3. Health Check / Static Files
**Endpoint:** `/`  
**Method:** `GET`  
**Description:** Serves the frontend application (HTML/JS/CSS).

## Deployment Notes (Hugging Face Spaces)
- **Dockerfile**: Included in root. Configured for CPU execution.
- **Port**: Exposes port `7860` (standard for HF Spaces).
- **Permissions**: Requires microphone access in browser.

## Model
- **Engine**: Parakeet TDT (RNN-T)
- **Weights**: Located in `/models` directory (handled by Dockerfile).
- **Hardware**: Optimized for CPU inference.
