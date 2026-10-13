import sherpa_onnx
import sys

print("Testing OfflineRecognizer initialization...")

try:
    config = sherpa_onnx.OfflineRecognizerConfig()
    # Fill with minimal valid config if needed, though default should exist
    # But usually default constructor works for Config
    
    print(f"Config type: {type(config)}")
    
    # Attempt 1: Positional
    try:
        print("Attempt 1: Positional argument")
        recognizer = sherpa_onnx.OfflineRecognizer(config)
        print("Success positional")
    except Exception as e:
        print(f"Failed positional: {e}")

    # Attempt 2: Keyword
    try:
        print("Attempt 2: Keyword argument")
        recognizer = sherpa_onnx.OfflineRecognizer(config=config)
        print("Success keyword")
    except Exception as e:
        print(f"Failed keyword: {e}")

    # Attempt 3: Check offline_recognizer module
    if hasattr(sherpa_onnx, 'offline_recognizer'):
        print(f"sherpa_onnx.offline_recognizer is: {sherpa_onnx.offline_recognizer}")
        # maybe it has a create function?

except Exception as e:
    print(f"Fatal error: {e}")
