use axum::{
    extract::{ws::{Message, WebSocket, WebSocketUpgrade}, State, Multipart, DefaultBodyLimit},
    response::{IntoResponse, Response, Json},
    routing::{get, post},
    Router,
};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::process::{Command, Stdio};
use std::io::Write;
use tower_http::services::ServeDir;
use tower_http::cors::CorsLayer;
use transcribe_rs::{
    engines::parakeet::{ParakeetEngine, ParakeetModelParams, ParakeetInferenceParams, TimestampGranularity},
    TranscriptionEngine,
};
use serde::{Deserialize, Serialize};

struct AppState {
    engine: Mutex<Option<ParakeetEngine>>,
}

#[derive(Deserialize, Debug)]
struct ConfigMessage {
    mode: String, // "raw" or "llm"
}

#[derive(Serialize)]
struct TranscriptionResponse {
    text: String,
    duration_ms: u128,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    
    println!("🦜 Parakeet Standalone Server Starting...");

    // 1. Initialize Engine
    let model_dir = PathBuf::from("models/parakeet-tdt-0.6b-v3-int8");
    let engine = if model_dir.exists() {
        println!("📂 Found model directory: {:?}", model_dir);
        let mut engine = ParakeetEngine::new();
        match engine.load_model_with_params(&model_dir, ParakeetModelParams::int8()) {
            Ok(_) => {
                println!("✅ Parakeet V3 Model loaded successfully!");
                Some(engine)
            },
            Err(e) => {
                eprintln!("❌ Failed to load model: {}", e);
                None
            }
        }
    } else {
        eprintln!("❌ Model directory not found at: {:?}", model_dir);
        eprintln!("⚠️ Server will start but transcription will fail.");
        None
    };

    let app_state = Arc::new(AppState {
        engine: Mutex::new(engine),
    });

    // 2. Setup Router
    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/transcribe", post(transcribe_handler))
        .nest_service("/", ServeDir::new("frontend"))
        .layer(CorsLayer::permissive())
        .layer(DefaultBodyLimit::max(1024 * 1024 * 50)) // 50MB limit
        .with_state(app_state);

    // 3. Bind and Serve
    let addr = SocketAddr::from(([0, 0, 0, 0], 7860)); // Hugging Face Spaces standard port
    println!("🚀 Server listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn transcribe_handler(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    let mut audio_data = Vec::new();

    // Parse multipart
    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() == Some("file") {
            if let Ok(bytes) = field.bytes().await {
                audio_data = bytes.to_vec();
                break; // Only take the first file
            }
        }
    }

    if audio_data.is_empty() {
        return Json(ErrorResponse { error: "No file provided".to_string() }).into_response();
    }

    // Convert raw PCM bytes to f32 if possible, or handle decoding.
    // NOTE: This simple demo assumes raw f32le 16kHz for simplicity unless we add a full decoder.
    // Ideally we should use symphonia or similar to decode wav/mp3.
    // For now, let's assume the user sends raw f32le 16kHz PCM or we do a basic conversion from WAV header.
    // A robust API needs full decoding.
    
    // Convert audio to 16kHz f32le using ffmpeg
    // This allows the API to accept ANY audio format (wav, mp3, ogg, m4a, etc.)
    let mut child = match Command::new("ffmpeg")
        .args(&[
            "-i", "pipe:0",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_f32le",
            "-f", "f32le",
            "-v", "quiet",
            "pipe:1"
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn() 
    {
        Ok(c) => c,
        Err(e) => return Json(ErrorResponse { error: format!("Failed to spawn ffmpeg: {}", e) }).into_response(),
    };

    if let Some(mut stdin) = child.stdin.take() {
        if let Err(e) = stdin.write_all(&audio_data) {
             return Json(ErrorResponse { error: format!("Failed to write to ffmpeg: {}", e) }).into_response();
        }
    }

    let output = match child.wait_with_output() {
        Ok(o) => o,
        Err(e) => return Json(ErrorResponse { error: format!("Failed to read ffmpeg output: {}", e) }).into_response(),
    };

    if !output.status.success() {
        return Json(ErrorResponse { error: "FFmpeg conversion failed (invalid audio format?)".to_string() }).into_response();
    }

    let floats: Vec<f32> = output.stdout
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // Transcribe
    let mut engine_guard = state.engine.lock().unwrap();
    if let Some(engine) = engine_guard.as_mut() {
        let start = std::time::Instant::now();
        let params = ParakeetInferenceParams {
            timestamp_granularity: TimestampGranularity::Segment,
            ..Default::default()
        };

        match engine.transcribe_samples(floats, Some(params)) {
            Ok(result) => {
                let duration = start.elapsed();
                Json(TranscriptionResponse {
                    text: result.text.trim().to_string(),
                    duration_ms: duration.as_millis(),
                }).into_response()
            },
            Err(e) => {
                Json(ErrorResponse { error: format!("Transcription failed: {}", e) }).into_response()
            }
        }
    } else {
        Json(ErrorResponse { error: "Model not loaded".to_string() }).into_response()
    }
}


async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> Response {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    println!("🔌 Client connected");
    
    let mut current_mode = "raw".to_string();
    let mut audio_buffer: Vec<f32> = Vec::new();
    let sample_rate = 16000; // Expected sample rate
    
    // Process messages
    while let Some(msg) = socket.recv().await {
        match msg {
            Ok(Message::Text(text)) => {
                // Handle config/control messages
                if let Ok(config) = serde_json::from_str::<ConfigMessage>(&text) {
                    println!("⚙️ Mode set to: {}", config.mode);
                    current_mode = config.mode;
                }
            }
            Ok(Message::Binary(data)) => {
                // Assume binary data is raw f32 PCM bytes (little endian)
                // Convert bytes to f32
                let chunks: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();

                // Calculate RMS (Wave Motion)
                let sum_squares: f32 = chunks.iter().map(|&x| x * x).sum();
                let rms = (sum_squares / chunks.len() as f32).sqrt();
                
                // Send Wave Motion update (normalized somewhat)
                let wave_level = (rms * 5.0).min(1.0); // Boost quiet signals
                let _ = socket.send(Message::Text(serde_json::json!({
                    "type": "wave",
                    "value": wave_level
                }).to_string())).await;

                audio_buffer.extend(chunks);

                // If buffer is big enough (~1 second), transcribe
                // In a real app, we'd use a smarter VAD or streaming window
                if audio_buffer.len() >= sample_rate * 2 { // Every 2 seconds
                     let transcription_result = {
                         let mut engine_guard = state.engine.lock().unwrap();
                         if let Some(engine) = engine_guard.as_mut() {
                             let start = std::time::Instant::now();
                             
                             let params = ParakeetInferenceParams {
                                timestamp_granularity: TimestampGranularity::Segment,
                                ..Default::default()
                            };
    
                            match engine.transcribe_samples(audio_buffer.clone(), Some(params)) {
                                Ok(result) => {
                                    let duration = start.elapsed();
                                    Some((result, duration))
                                },
                                Err(e) => {
                                    eprintln!("Transcription error: {}", e);
                                    None
                                }
                            }
                         } else {
                             None
                         }
                     };

                     if let Some((result, duration)) = transcription_result {
                        let audio_duration_sec = audio_buffer.len() as f32 / sample_rate as f32;
                        let wpm = if audio_duration_sec > 0.0 {
                            (result.text.split_whitespace().count() as f32 / (audio_duration_sec / 60.0)) as u32
                        } else {
                            0
                        };

                        let final_text = if current_mode == "llm" {
                            // Simulated LLM mode (just capitalization/formatting for now as requested)
                            format!("✨ {}", result.text.trim()) 
                        } else {
                            result.text.trim().to_string()
                        };

                        let _ = socket.send(Message::Text(serde_json::json!({
                            "type": "transcript",
                            "text": final_text,
                            "mode": current_mode
                        }).to_string())).await;
                        
                        let _ = socket.send(Message::Text(serde_json::json!({
                            "type": "speed",
                            "wpm": wpm,
                            "latency_ms": duration.as_millis()
                        }).to_string())).await;
                     }
                     
                     // Keep some context or clear? For this simple demo, clear.
                     // Better: overlap?
                     audio_buffer.clear(); 
                }
            }
            Ok(Message::Close(_)) => {
                println!("🔌 Client disconnected");
                break;
            }
            Err(e) => {
                eprintln!("❌ Socket error: {}", e);
                break;
            }
            _ => {}
        }
    }
}
