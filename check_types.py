
import sherpa_onnx
try:
    # Pass invalid type to trigger error listing valid types
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder="dummy",
        decoder="dummy",
        joiner="dummy",
        tokens="dummy",
        model_type="invalid_type_to_get_list"
    )
except Exception as e:
    print(e)
