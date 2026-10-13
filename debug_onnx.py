import onnxruntime as ort
import numpy as np
import os

MODEL_DIR = "models/parakeet-tdt-0.6b-v3-int8"

def test_encoder():
    print("--- Testing Encoder ---")
    model_path = f"{MODEL_DIR}/encoder-model.int8.onnx"
    sess = ort.InferenceSession(model_path)
    
    # Inputs: audio_signal (Float), length (Int64)
    # Shapes: audio_signal [1, 128, 100], length [1]
    
    audio_signal = np.random.randn(1, 128, 100).astype(np.float32)
    length = np.array([100], dtype=np.int64)
    
    try:
        outputs = sess.run(None, {
            "audio_signal": audio_signal,
            "length": length
        })
        print("Encoder run success!")
        return outputs[0] # encoder_outputs
    except Exception as e:
        print(f"Encoder run failed: {e}")
        return None

def test_decoder(encoder_outputs):
    print("\n--- Testing Decoder ---")
    model_path = f"{MODEL_DIR}/decoder_joint-model.int8.onnx"
    sess = ort.InferenceSession(model_path)
    
    # Inputs: 
    # encoder_outputs (Float) [1, 100, 1024] (assuming output dim 1024)
    # targets (Int32) [1, 2] (Context size 2)
    # target_length (Int32) [1]
    # input_states_1 (Float) [2, 1, 640]
    # input_states_2 (Float) [2, 1, 640]
    
    if encoder_outputs is None:
        # Create dummy
        encoder_outputs = np.random.randn(1, 100, 1024).astype(np.float32)
    
    # Check encoder output shape
    print(f"Encoder output shape: {encoder_outputs.shape}")
    
    targets = np.zeros((1, 2), dtype=np.int32)
    target_length = np.array([2], dtype=np.int32)
    
    input_states_1 = np.zeros((2, 1, 640), dtype=np.float32)
    input_states_2 = np.zeros((2, 1, 640), dtype=np.float32)
    
    inputs = {
        "encoder_outputs": encoder_outputs,
        "targets": targets,
        "target_length": target_length,
        "input_states_1": input_states_1,
        "input_states_2": input_states_2
    }
    
    try:
        outputs = sess.run(None, inputs)
        print("Decoder run success!")
    except Exception as e:
        print(f"Decoder run failed: {e}")
        
        # Try debugging types
        print("Retrying with Int64 targets...")
        inputs["targets"] = inputs["targets"].astype(np.int64)
        inputs["target_length"] = inputs["target_length"].astype(np.int64)
        try:
            sess.run(None, inputs)
            print("Decoder run success with Int64!")
        except Exception as e2:
            print(f"Decoder run failed with Int64: {e2}")

if __name__ == "__main__":
    enc_out = test_encoder()
    test_decoder(enc_out)
