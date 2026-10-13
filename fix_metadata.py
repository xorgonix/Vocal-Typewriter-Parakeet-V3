import os
import onnx

MODEL_DIR = "models/parakeet-tdt-0.6b-v3-int8"

def add_metadata(model_path, vocab_size):
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    try:
        model = onnx.load(model_path)
        meta_dict = {prop.key: prop.value for prop in model.metadata_props}
        modified = False

        # 1. vocab_size
        target_vocab_size = vocab_size
        
        if "vocab_size" not in meta_dict or meta_dict['vocab_size'] != str(target_vocab_size):
            print(f"Setting vocab_size to {target_vocab_size} in {model_path} (was {meta_dict.get('vocab_size', 'missing')})")
            # Remove existing if present to avoid duplicates (though .add() appends)
            # Better to find and update, or add if missing.
            found = False
            for prop in model.metadata_props:
                if prop.key == "vocab_size":
                    prop.value = str(target_vocab_size)
                    found = True
                    break
            if not found:
                meta = model.metadata_props.add()
                meta.key = "vocab_size"
                meta.value = str(target_vocab_size)
            modified = True

        # 2. context_size (usually 2 for Parakeet)
        if "context_size" not in meta_dict:
            print(f"Adding context_size=2 to {model_path}")
            meta = model.metadata_props.add()
            meta.key = "context_size"
            meta.value = "2"
            modified = True
        else:
            print(f"context_size already exists in {model_path} with value {meta_dict['context_size']}")

        # 3. subsampling_factor (from config.json)
        if "subsampling_factor" not in meta_dict:
            print(f"Adding subsampling_factor=8 to {model_path}")
            meta = model.metadata_props.add()
            meta.key = "subsampling_factor"
            meta.value = "8"
            modified = True
        else:
            # Update if it exists but is wrong (I set it to 4 previously)
            if meta_dict['subsampling_factor'] != "8":
                print(f"Updating subsampling_factor from {meta_dict['subsampling_factor']} to 8 in {model_path}")
                for prop in model.metadata_props:
                    if prop.key == "subsampling_factor":
                        prop.value = "8"
                        modified = True
                        break
            else:
                print(f"subsampling_factor already exists in {model_path} with value {meta_dict['subsampling_factor']}")

        # 4. pred_rnn_layers (from onnx inspection: input_states [2, ?, 640] -> 2 layers)
        if "pred_rnn_layers" not in meta_dict:
            print(f"Adding pred_rnn_layers=2 to {model_path}")
            meta = model.metadata_props.add()
            meta.key = "pred_rnn_layers"
            meta.value = "2"
            modified = True
        else:
            print(f"pred_rnn_layers already exists in {model_path} with value {meta_dict['pred_rnn_layers']}")

        # 5. pred_hidden (from onnx inspection: input_states [2, ?, 640] -> 640 hidden)
        if "pred_hidden" not in meta_dict:
            print(f"Adding pred_hidden=640 to {model_path}")
            meta = model.metadata_props.add()
            meta.key = "pred_hidden"
            meta.value = "640"
            modified = True
        else:
            print(f"pred_hidden already exists in {model_path} with value {meta_dict['pred_hidden']}")

        if modified:
            onnx.save(model, model_path)
            print(f"Saved updated model to {model_path}")
        else:
            print(f"No changes needed for {model_path}")

    except Exception as e:
        print(f"Error processing {model_path}: {e}")

def main():
    if not os.path.exists(MODEL_DIR):
        print(f"Model dir {MODEL_DIR} not found")
        return

    # FORCE vocab_size to 8192 (sherpa adds 1 for blank, so 8192+1=8193 matches sliced output)
    vocab_size = 8192
    print(f"Forcing vocab_size: {vocab_size} (expecting sherpa to add 1 -> 8193)")

    files = [
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        "joiner.int8.onnx"
    ]
    
    for f in files:
        path = os.path.join(MODEL_DIR, f)
        add_metadata(path, vocab_size)
        
    # Hack to remove "tdt" from metadata to avoid confusing sherpa-onnx
    # if the model is actually exported as standard transducer
    encoder_path = os.path.join(MODEL_DIR, "encoder.int8.onnx")
    import onnx
    model = onnx.load(encoder_path)
    changed = False
    for prop in model.metadata_props:
        if prop.key == "url" and "tdt" in prop.value.lower():
            print(f"Changing url from {prop.value} to generic")
            prop.value = "https://huggingface.co/nvidia/parakeet-rnnt-0.6b"
            changed = True
        if prop.key == "comment" and "tdt" in prop.value.lower():
             # actually comment said "Only the transducer branch is exported" which is fine
             pass
    
    if changed:
        onnx.save(model, encoder_path)
        print("Updated encoder metadata to remove TDT reference in URL")

if __name__ == "__main__":
    main()
