import sherpa_onnx
import numpy as np
import wave
import os
import sys

def test_recognizer():
    print("Initializing recognizer...")
    
    model_dir = "models/parakeet-tdt-0.6b-v3-int8"
    encoder = f"{model_dir}/encoder-model.int8.onnx"
    decoder = "dummy_decoder.onnx"
    joiner = f"{model_dir}/decoder_joint-model.int8.onnx"
    tokens = f"{model_dir}/tokens.txt"
    
    if not os.path.exists(encoder):
        print(f"Error: {encoder} not found")
        return

    try:
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                tokens=tokens,
                num_threads=1,
                sample_rate=16000,
                feature_dim=128,
                model_type="nemo_transducer",
                debug=True,
                decoding_method="greedy_search"
            )
        print("Recognizer initialized.")
    except Exception as e:
        print(f"Failed to init recognizer: {e}")
        return

    # Test with synthetic audio (sine wave) - might not produce text but should run
    print("Testing with synthetic audio (sine wave 440Hz)...")
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)
    recognizer.decode_stream(stream)
    print(f"Result (Sine): '{stream.result.text}'")
    
    # Test with VAD logic
    print("Testing VAD logic...")
    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = "models/silero_vad.onnx"
    # Set threshold slightly lower to ensure detection of our sine wave
    vad_config.silero_vad.threshold = 0.3 
    vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)
    
    # Feed audio to VAD
    vad.accept_waveform(audio)
    
    # Add silence to flush
    silence = np.zeros(int(sample_rate * 1.0), dtype=np.float32)
    vad.accept_waveform(silence)
    
    if vad.is_speech_detected():
        print("VAD detected speech!")
        while not vad.empty():
            segment = vad.front
            vad.pop()
            print(f"Segment found: {len(segment.samples)} samples")
            
            s = recognizer.create_stream()
            s.accept_waveform(sample_rate, segment.samples)
            recognizer.decode_stream(s)
            print(f"Segment Text: '{s.result.text}'")
    else:
        print("VAD did NOT detect speech.")

if __name__ == "__main__":
    test_recognizer()
