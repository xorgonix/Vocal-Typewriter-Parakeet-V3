
import onnx
from onnx import helper, TensorProto

def create_dummy_decoder(output_path):
    # Create a dummy decoder that takes 'targets' and 'target_length'
    # and returns 'dummy_output'
    
    # Inputs
    targets = helper.make_tensor_value_info('targets', TensorProto.INT32, [1, 10])
    target_length = helper.make_tensor_value_info('target_length', TensorProto.INT32, [1])
    
    # Outputs
    dummy_out = helper.make_tensor_value_info('dummy_output', TensorProto.FLOAT, [1, 10, 64])
    
    # Node (Identity or Constant)
    # We just need a valid graph.
    # Let's return a constant tensor matching output shape?
    # Or just Identity of something? 
    # Since inputs and outputs have different types/shapes, we can't use Identity easily.
    # Let's create a Constant node.
    
    constant_node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['dummy_output'],
        value=helper.make_tensor(
            name='const_tensor',
            data_type=TensorProto.FLOAT,
            dims=[1, 10, 64],
            vals=[0.0] * 640
        )
    )
    
    # Graph
    graph = helper.make_graph(
        [constant_node],
        'dummy_decoder',
        [targets, target_length],
        [dummy_out]
    )
    
    # Model
    model = helper.make_model(graph, producer_name='dummy_decoder', ir_version=8)
    
    # Save
    onnx.save(model, output_path)
    print(f"Saved dummy decoder to {output_path}")

if __name__ == "__main__":
    create_dummy_decoder("dummy_decoder.onnx")
