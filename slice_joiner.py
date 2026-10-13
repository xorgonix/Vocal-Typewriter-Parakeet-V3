import onnx
from onnx import helper, TensorProto
import os
import sys

MODEL_DIR = "models/parakeet-tdt-0.6b-v3-int8"
JOINER_PATH = os.path.join(MODEL_DIR, "joiner.int8.onnx")

def slice_model():
    print(f"Starting slice_joiner.py...")
    print(f"Looking for model at: {os.path.abspath(JOINER_PATH)}")
    
    if not os.path.exists(JOINER_PATH):
        print(f"CRITICAL ERROR: {JOINER_PATH} not found!")
        # List directory to debug
        parent_dir = os.path.dirname(JOINER_PATH)
        if os.path.exists(parent_dir):
            print(f"Contents of {parent_dir}:")
            print(os.listdir(parent_dir))
        else:
            print(f"Parent directory {parent_dir} does not exist!")
        sys.exit(1)

    print(f"Loading {JOINER_PATH}...")
    try:
        model = onnx.load(JOINER_PATH)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model: {e}")
        sys.exit(1)
        
    graph = model.graph

    # Target output dimension
    TARGET_DIM = 8193

    # 1. Identify the output tensor
    if len(graph.output) != 1:
        print(f"CRITICAL ERROR: Expected 1 output, found {len(graph.output)}")
        sys.exit(1)
    
    output_tensor_info = graph.output[0]
    original_output_name = output_tensor_info.name
    print(f"Original output name: {original_output_name}")
    
    # Check current shape
    dim_val = output_tensor_info.type.tensor_type.shape.dim[-1].dim_value
    print(f"Current output dimension: {dim_val}")
    
    if dim_val == TARGET_DIM:
        print(f"SUCCESS: Model already has target dimension {TARGET_DIM}. No surgery needed.")
        return

    # 2. Rename the *actual* output of the last layer so we can use it as input to Slice
    producing_node = None
    for node in graph.node:
        if original_output_name in node.output:
            producing_node = node
            break
            
    if not producing_node:
        print(f"CRITICAL ERROR: Could not find node producing {original_output_name}")
        # Debug: print last few nodes
        print("Last 5 nodes in graph:")
        for n in graph.node[-5:]:
            print(f" - Node: {n.name}, Op: {n.op_type}, Outputs: {n.output}")
        sys.exit(1)

    # Create a new intermediate name
    intermediate_name = original_output_name + "_unsliced"
    
    # Update the producing node to output to this intermediate name
    for i, out_name in enumerate(producing_node.output):
        if out_name == original_output_name:
            producing_node.output[i] = intermediate_name
            break
            
    print(f"Renamed output of last node to: {intermediate_name}")

    # 3. Create Slice Node
    starts = onnx.helper.make_tensor("slice_starts", TensorProto.INT64, [1], [0])
    ends = onnx.helper.make_tensor("slice_ends", TensorProto.INT64, [1], [TARGET_DIM])
    axes = onnx.helper.make_tensor("slice_axes", TensorProto.INT64, [1], [-1]) # Slice last axis
    steps = onnx.helper.make_tensor("slice_steps", TensorProto.INT64, [1], [1])
    
    graph.initializer.extend([starts, ends, axes, steps])
    
    slice_node = onnx.helper.make_node(
        "Slice",
        inputs=[intermediate_name, "slice_starts", "slice_ends", "slice_axes", "slice_steps"],
        outputs=[original_output_name], # Output to the original name so graph output matches
        name="OutputSlicer"
    )
    
    graph.node.append(slice_node)
    
    # 4. Update graph output shape information
    output_tensor_info.type.tensor_type.shape.dim[-1].dim_value = TARGET_DIM
    
    print(f"Inserted Slice node. New output dimension: {TARGET_DIM}")
    
    # 5. Save
    print(f"Saving to {JOINER_PATH}...")
    try:
        onnx.save(model, JOINER_PATH)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to save model: {e}")
        sys.exit(1)
        
    print("Verification...")
    # Verify reload
    try:
        v_model = onnx.load(JOINER_PATH)
        v_dim = v_model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
        if v_dim != TARGET_DIM:
            print(f"CRITICAL ERROR: Verification failed! Dimension is {v_dim}, expected {TARGET_DIM}")
            sys.exit(1)
        print(f"VERIFIED: Model output dimension is now {v_dim}")
    except Exception as e:
        print(f"CRITICAL ERROR: Verification crashed: {e}")
        sys.exit(1)

    print("SLICE SUCCESSFUL")

if __name__ == "__main__":
    slice_model()
