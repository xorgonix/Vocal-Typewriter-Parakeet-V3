# Parakeet API Documentation

This application provides a robust speech-to-text API using the Parakeet TDT 0.6b model (Offline Recognizer with VAD).

- **Space Page**: `https://huggingface.co/spaces/Loomisgitarrist/Vocal-Typewriter-Parakeet-V3`

## Base URL
- **Direct API**: `https://loomisgitarrist-vocal-typewriter-parakeet-v3.hf.space`

## Endpoints

### 1. Transcribe Audio File (REST)
Upload an audio file to get a full transcription.

- **URL**: `/transcribe`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: The audio file to transcribe (supports `.wav`, `.mp3`, `.ogg`, `.m4a`, etc. via FFmpeg).

#### Example (cURL)
```bash
curl -X POST "https://loomisgitarrist-vocal-typewriter-parakeet-v3.hf.space/transcribe" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/audio.mp3"
```

#### Response
```json
{
  "text": "Hello world this is a test transcription"
}
```

---

### 2. Get Session Logs (REST)
Retrieve statistics about the current session, including latency, speed, and transcription history.

- **URL**: `/logs`
- **Method**: `GET`

#### Example (cURL)
```bash
curl "https://loomisgitarrist-vocal-typewriter-parakeet-v3.hf.space/logs"
```

#### Response
```json
{
  "total_segments": 15,
  "total_latency_ms": 2500,
  "last_latency_ms": 146,
  "last_speed_x": 4.5,
  "avg_latency_ms": 166.6,
  "history": [
    {
      "timestamp": 1700000000.0,
      "latency_ms": 146.0,
      "speed_x": 4.5,
      "duration_s": 2.5,
      "text": "Hello world"
    }
  ]
}
```

---

### 3. Real-time Streaming (WebSocket)
Bi-directional streaming for real-time transcription.

- **URL**: `/ws`
- **Protocol**: WebSocket

#### Usage Flow
1. **Connect** to `wss://loomisgitarrist-vocal-typewriter-parakeet-v3.hf.space/ws`
2. **Send Audio**: Send raw PCM audio data as binary messages.
   - Format: 32-bit float (`float32`), 16kHz sample rate, mono channel.
   - Buffer: Send chunks of audio (e.g., 100ms - 500ms).
3. **Receive Transcripts**: The server sends JSON messages.
   - Transcript: `{"type": "transcript", "text": "Hello world ", "latency": 145, "speed": 4.2}`
   - Waveform Level: `{"type": "wave", "value": 0.5}` (RMS amplitude for visualization)

#### Example (JavaScript)
```javascript
const ws = new WebSocket("wss://loomisgitarrist-vocal-typewriter-parakeet-v3.hf.space/ws");

ws.onopen = () => {
    // Send audio from AudioContext / ScriptProcessor
    // float32Array is the raw audio buffer from the microphone
    ws.send(float32Array.buffer); 
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "transcript") {
        console.log("Transcribed:", data.text);
        console.log("Latency:", data.latency, "ms");
    } else if (data.type === "wave") {
        // Update visualizer
    }
};
```
