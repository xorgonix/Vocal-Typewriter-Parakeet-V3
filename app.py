import os
import sys
import json
import logging
import asyncio
import subprocess
import tempfile
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, JSONResponse
import sherpa_onnx

import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("parakeet-server")

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MODEL_DIR = "models/parakeet-tdt-0.6b-v3-int8"
VAD_MODEL = "models/silero_vad.onnx"
SAMPLE_RATE = 16000

def get_model_paths():
    files = os.listdir(MODEL_DIR)
    logger.info(f"Files in {MODEL_DIR}: {files}")
    
    config = {}
    
    # Tokens
    if "tokens.txt" in files: config["tokens"] = f"{MODEL_DIR}/tokens.txt"
    elif "vocab.txt" in files: config["tokens"] = f"{MODEL_DIR}/vocab.txt"
    else: raise FileNotFoundError("vocab.txt or tokens.txt not found")
    
    # Encoder
    if "encoder-model.int8.onnx" in files: config["encoder"] = f"{MODEL_DIR}/encoder-model.int8.onnx"
    elif "encoder.int8.onnx" in files: config["encoder"] = f"{MODEL_DIR}/encoder.int8.onnx"
    elif "encoder.onnx" in files: config["encoder"] = f"{MODEL_DIR}/encoder.onnx"
    else: raise FileNotFoundError("encoder model not found")
    
    # Decoder
    if "decoder.int8.onnx" in files: config["decoder"] = f"{MODEL_DIR}/decoder.int8.onnx"
    elif "decoder_joint-model.int8.onnx" in files: config["decoder"] = f"{MODEL_DIR}/decoder_joint-model.int8.onnx"
    elif "decoder.onnx" in files: config["decoder"] = f"{MODEL_DIR}/decoder.onnx"
    else: raise FileNotFoundError("decoder model not found")
    
    # Joiner
    if "joiner.int8.onnx" in files: config["joiner"] = f"{MODEL_DIR}/joiner.int8.onnx"
    elif "joiner.onnx" in files: config["joiner"] = f"{MODEL_DIR}/joiner.onnx"
    # Fallback for combined models (old behavior)
    elif config.get("decoder") and "joint" in config["decoder"]: 
        config["joiner"] = config["decoder"]
    else:
        # If we have decoder but no joiner file, and it's not explicitly named "joint", 
        # it might be unsafe to assume it's combined, but let's try or fail.
        # Given we downloaded separate files, we expect joiner.int8.onnx
        if "joiner.int8.onnx" not in files:
             # Try to see if maybe it is combined but named differently?
             # For now, let's just log warning and try using decoder path if really desperate, 
             # but better to fail if we know we need a joiner.
             # However, the previous logic was config["joiner"] = config["decoder"].
             # Let's keep that ONLY if we didn't find a specific joiner file.
             config["joiner"] = config["decoder"]
    
    return config

# Global Recognizer
recognizer = None

def init_recognizer():
    global recognizer
    try:
        paths = get_model_paths()
        
        # Use OfflineRecognizer.from_transducer factory method
        # Explicitly set feature_dim=128 for Parakeet TDT
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=paths["encoder"],
            decoder=paths["decoder"],
            joiner=paths["joiner"],
            tokens=paths["tokens"],
            num_threads=1,
            sample_rate=16000,
            feature_dim=128,
            model_type="nemo_transducer",
            debug=True,
            decoding_method="greedy_search"
        )
        
        logger.info("Offline Recognizer initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to create recognizer: {e}")
        return False

# Initialize on startup
init_recognizer()

# Session Stats Storage
session_stats = {
    "total_segments": 0,
    "total_latency_ms": 0,
    "last_latency_ms": 0,
    "last_speed_x": 0,
    "avg_latency_ms": 0,
    "history": [] # Keep last 50
}

@app.get("/logs")
async def get_logs():
    """
    Get current session statistics including latency and speed.
    """
    return JSONResponse(content=session_stats)

def create_vad():
    if not os.path.exists(VAD_MODEL):
        logger.error(f"VAD model not found at {VAD_MODEL}")
        return None
        
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = VAD_MODEL
    config.sample_rate = SAMPLE_RATE
    config.silero_vad.threshold = 0.25 # Increased to 0.25 to stop "hanging" on noise/breathing
    config.silero_vad.min_silence_duration = 0.2
    config.silero_vad.min_speech_duration = 0.1 # Catch short words
    
    return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Transcribe an audio file (any format supported by FFmpeg).
    Returns JSON: {"text": "transcription..."}
    """
    if not recognizer:
        raise HTTPException(status_code=503, detail="Recognizer not initialized")

    # Save uploaded file to a temporary file
    suffix = os.path.splitext(file.filename)[1]
    if not suffix: suffix = ".tmp"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Use FFmpeg to convert to 16kHz mono float32 (raw PCM)
        # This handles mp3, wav, ogg, m4a, etc.
        cmd = [
            "ffmpeg",
            "-y", # overwrite
            "-i", tmp_path,
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ac", "1",
            "-ar", "16000",
            "-" # Output to stdout
        ]
        
        # Run FFmpeg
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        
        if process.returncode != 0:
            error_msg = err.decode()
            logger.error(f"FFmpeg error: {error_msg}")
            raise HTTPException(status_code=400, detail=f"Could not process audio: {error_msg}")

        # Convert raw bytes to float32 numpy array
        samples = np.frombuffer(out, dtype=np.float32)
        
        if len(samples) == 0:
             return JSONResponse(content={"text": ""})

        # Create stream and decode
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        recognizer.decode_stream(stream)
        
        text = stream.result.text.strip()
        return JSONResponse(content={"text": text})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")
    
    if not recognizer:
        logger.error("Recognizer not initialized")
        await websocket.close(code=1011)
        return
        
    vad = create_vad()
    if not vad:
        logger.error("VAD not initialized")
        await websocket.close(code=1011)
        return
    
    try:
        while True:
            # Receive generic message
            message = await websocket.receive()
            
            if "bytes" in message:
                # Process audio
                data = message["bytes"]
                if not data: continue
                
                samples = np.frombuffer(data, dtype=np.float32)
                
                # Calculate and send audio level for visualization
                rms = np.sqrt(np.mean(samples**2))
                await websocket.send_text(json.dumps({
                    "type": "wave",
                    "value": float(rms) * 5.0  # Boost visual gain
                }))
                
                # Feed VAD
                vad.accept_waveform(samples)
                
                # Check for detected speech segments
                # is_speech = vad.is_speech_detected()
                # if is_speech:
                #      logger.info("Speech detected by VAD!")
                
                while not vad.empty():
                    segment = vad.front
                    # CRITICAL: Copy samples before popping!
                    segment_samples = segment.samples
                    vad.pop()
                    
                    # We have a speech segment. Decode it.
                    if len(segment_samples) > 0:
                            duration = len(segment_samples) / SAMPLE_RATE
                            logger.info(f"Decoding segment: {duration:.2f}s")
                            
                            start_time = time.time()
                            try:
                                stream = recognizer.create_stream()
                                stream.accept_waveform(SAMPLE_RATE, segment_samples)
                                recognizer.decode_stream(stream)
                                
                                text = stream.result.text.strip()
                                end_time = time.time()
                                
                                latency = (end_time - start_time) * 1000 # ms
                                rtf = (end_time - start_time) / duration if duration > 0 else 0
                                speed = 1 / rtf if rtf > 0 else 0
                                
                                # Update global stats
                                session_stats["total_segments"] += 1
                                session_stats["total_latency_ms"] += latency
                                session_stats["last_latency_ms"] = latency
                                session_stats["last_speed_x"] = speed
                                session_stats["avg_latency_ms"] = session_stats["total_latency_ms"] / session_stats["total_segments"]
                                
                                entry = {
                                    "timestamp": time.time(),
                                    "latency_ms": round(latency, 2),
                                    "speed_x": round(speed, 2),
                                    "duration_s": round(duration, 2),
                                    "text": text
                                }
                                session_stats["history"].append(entry)
                                if len(session_stats["history"]) > 50:
                                    session_stats["history"].pop(0)
                                
                                if text:
                                    logger.info(f"Result: {text} (Latency: {latency:.0f}ms, Speed: {speed:.2f}x)")
                                    # Send text to frontend
                                    await websocket.send_text(json.dumps({
                                        "type": "transcript",
                                        "text": text + " ",
                                        "latency": latency,
                                        "rtf": rtf,
                                        "speed": speed
                                    }))
                            except Exception as e:
                                logger.error(f"Decoding failed: {e}")
                                await websocket.send_text(json.dumps({
                                    "type": "error",
                                    "message": str(e)
                                }))

            # Ignore text messages (config) as we only use raw mode
                    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

# Mount static files at root (must be last to avoid masking other routes)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
