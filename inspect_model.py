import onnx
import os
import sys

def inspect(model_path):
    print(f"--- Inspecting {model_path} ---")
    try:
        model = onnx.load(model_path)
        print("Metadata:")
        for prop in model.metadata_props:
            print(f"  {prop.key}: {prop.value}")
            
        print("Inputs:")
        for input in model.graph.input:
            print(f"  {input.name}: {input.type.tensor_type.shape}")
            print(f"    Type: {input.type.tensor_type.elem_type} (1=FLOAT, 6=INT32, 7=INT64)")

        print("Outputs:")
        for output in model.graph.output:
            print(f"  {output.name}: {output.type.tensor_type.shape}")
            print(f"    Type: {output.type.tensor_type.elem_type} (1=FLOAT, 6=INT32, 7=INT64)")
            
    except Exception as e:
        print(f"Error: {e}")

def main():
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        # Default fallback
        files = []
    
    for f in files:
        if os.path.exists(f):
            inspect(f)
        else:
            print(f"File not found: {f}")

if __name__ == "__main__":
    main()
