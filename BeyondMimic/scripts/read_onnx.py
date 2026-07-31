"""Script to read and analyze ONNX model files.

This script reads an ONNX model and prints detailed information about:
- Model inputs and outputs
- Model metadata (joint names, observation names, etc.)
- Graph structure
- Node information

Usage:
    # Activate your conda environment first (e.g., unitree_rl)
    python read_onnx.py --onnx_file logs/rsl_rl/temp/exported/policy.onnx
    
    # Or use the conda environment directly:
    /home/sustech/.conda/envs/unitree_rl/bin/python read_onnx.py --onnx_file logs/rsl_rl/temp/exported/policy.onnx
"""

import argparse
import sys

try:
    import onnx
    import onnxruntime as ort
    import numpy as np
    from typing import Any
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Please install required packages:")
    print("  pip install onnx onnxruntime numpy")
    sys.exit(1)


def parse_metadata_value(value: str) -> Any:
    """Parse metadata value from string format."""
    # Try to parse as list (comma-separated)
    if "," in value:
        try:
            # Try to parse as float list
            return [float(x) for x in value.split(",")]
        except ValueError:
            # Return as string list
            return value.split(",")
    # Try to parse as float
    try:
        return float(value)
    except ValueError:
        # Return as string
        return value


def print_section(title: str, char: str = "="):
    """Print a formatted section title."""
    print("\n" + char * 80)
    print(f"  {title}")
    print(char * 80)


def print_model_info(model_path: str):
    """Read and print comprehensive information about an ONNX model."""
    # Load the model
    model = onnx.load(model_path)
    
    # Verify model
    try:
        onnx.checker.check_model(model)
        print("✓ Model is valid")
    except onnx.checker.ValidationError as e:
        print(f"✗ Model validation error: {e}")
        return
    
    # Print basic info
    print_section("MODEL BASIC INFORMATION")
    print(f"Model file: {model_path}")
    print(f"IR Version: {model.ir_version}")
    print(f"Producer Name: {model.producer_name}")
    print(f"Producer Version: {model.producer_version}")
    print(f"Opset Version: {model.opset_import[0].version}")
    
    # Print model doc string
    if model.doc_string:
        print(f"Description: {model.doc_string}")
    
    # Print metadata
    print_section("MODEL METADATA")
    metadata = {}
    for prop in model.metadata_props:
        key = prop.key
        value = prop.value
        parsed_value = parse_metadata_value(value)
        metadata[key] = parsed_value
        print(f"  {key}:")
        if isinstance(parsed_value, list):
            if len(parsed_value) <= 10:
                print(f"    {parsed_value}")
            else:
                print(f"    [list with {len(parsed_value)} items]")
                print(f"    First 5: {parsed_value[:5]}")
                print(f"    Last 5: {parsed_value[-5:]}")
        else:
            print(f"    {parsed_value}")
    
    # Print graph inputs
    print_section("MODEL INPUTS")
    for i, input_tensor in enumerate(model.graph.input):
        print(f"\n  Input {i}: {input_tensor.name}")
        if input_tensor.type.tensor_type.shape.dim:
            shape = []
            for dim in input_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append("?")
            print(f"    Shape: {tuple(shape)}")
        else:
            print("    Shape: Unknown")
        print(f"    Type: {input_tensor.type.tensor_type.elem_type}")
        if input_tensor.doc_string:
            print(f"    Doc: {input_tensor.doc_string}")
    
    # Print graph outputs
    print_section("MODEL OUTPUTS")
    for i, output_tensor in enumerate(model.graph.output):
        print(f"\n  Output {i}: {output_tensor.name}")
        if output_tensor.type.tensor_type.shape.dim:
            shape = []
            for dim in output_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append("?")
            print(f"    Shape: {tuple(shape)}")
        else:
            print("    Shape: Unknown")
        print(f"    Type: {output_tensor.type.tensor_type.elem_type}")
        if output_tensor.doc_string:
            print(f"    Doc: {output_tensor.doc_string}")
    
    # Print graph structure
    print_section("GRAPH STRUCTURE")
    print(f"Total nodes: {len(model.graph.node)}")
    print(f"Total initializers (weights): {len(model.graph.initializer)}")
    
    # Count node types
    node_types = {}
    for node in model.graph.node:
        node_type = node.op_type
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print("\n  Node type distribution:")
    for node_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
        print(f"    {node_type}: {count}")
    
    # Print some example nodes
    print("\n  Sample nodes (first 10):")
    for i, node in enumerate(model.graph.node[:10]):
        print(f"    {i}: {node.op_type} - inputs: {node.input}, outputs: {node.output}")
    
    if len(model.graph.node) > 10:
        print(f"    ... and {len(model.graph.node) - 10} more nodes")
    
    # Print initializers info
    print_section("INITIALIZERS (WEIGHTS)")
    total_params = 0
    for init in model.graph.initializer:
        shape = tuple(init.dims)
        num_params = np.prod(shape) if shape else 1
        total_params += num_params
        print(f"  {init.name}: shape={shape}, params={num_params}")
    
    print(f"\n  Total parameters: {total_params:,}")
    
    # Try to create ONNX Runtime session and get more info
    print_section("ONNX RUNTIME SESSION INFO")
    try:
        session = ort.InferenceSession(model_path)
        
        print("\n  Inputs:")
        for input_meta in session.get_inputs():
            print(f"    {input_meta.name}: shape={input_meta.shape}, type={input_meta.type}")
        
        print("\n  Outputs:")
        for output_meta in session.get_outputs():
            print(f"    {output_meta.name}: shape={output_meta.shape}, type={output_meta.type}")
        
        # Print providers
        print(f"\n  Available providers: {ort.get_available_providers()}")
        print(f"  Providers used: {session.get_providers()}")
        
    except Exception as e:
        print(f"  Error creating ONNX Runtime session: {e}")
    
    # Print detailed metadata interpretation
    if metadata:
        print_section("METADATA INTERPRETATION")
        
        if "observation_names" in metadata:
            obs_names = metadata["observation_names"]
            obs_history = metadata.get("observation_history_lengths", [])
            print("\n  Observation Space:")
            for i, (name, history) in enumerate(zip(obs_names, obs_history)):
                print(f"    {i}: {name} (history_length={history})")
        
        if "joint_names" in metadata:
            joint_names = metadata["joint_names"]
            print(f"\n  Joint Names ({len(joint_names)} joints):")
            for i, name in enumerate(joint_names):
                print(f"    {i}: {name}")
        
        if "action_scale" in metadata:
            action_scale = metadata["action_scale"]
            print(f"\n  Action Scale ({len(action_scale)} values):")
            if len(action_scale) <= 20:
                print(f"    {action_scale}")
            else:
                print(f"    First 10: {action_scale[:10]}")
                print(f"    Last 10: {action_scale[-10:]}")
        
        if "default_joint_pos" in metadata:
            default_pos = metadata["default_joint_pos"]
            print(f"\n  Default Joint Positions ({len(default_pos)} values):")
            if len(default_pos) <= 20:
                print(f"    {default_pos}")
            else:
                print(f"    First 10: {default_pos[:10]}")
                print(f"    Last 10: {default_pos[-10:]}")
        
        if "joint_stiffness" in metadata:
            stiffness = metadata["joint_stiffness"]
            print(f"\n  Joint Stiffness: {stiffness}")
        
        if "joint_damping" in metadata:
            damping = metadata["joint_damping"]
            print(f"\n  Joint Damping: {damping}")
        
        if "command_names" in metadata:
            cmd_names = metadata["command_names"]
            print(f"\n  Command Names: {cmd_names}")
        
        if "anchor_body_name" in metadata:
            print(f"\n  Anchor Body Name: {metadata['anchor_body_name']}")
        
        if "body_names" in metadata:
            print(f"\n  Body Names: {metadata['body_names']}")


def main():
    parser = argparse.ArgumentParser(description="Read and analyze ONNX model files")
    parser.add_argument(
        "--onnx_file",
        type=str,
        required=True,
        help="Path to the ONNX model file",
    )
    args = parser.parse_args()
    
    print_model_info(args.onnx_file)


if __name__ == "__main__":
    main()

