#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "coremltools>=7.0",
#     "onnx>=1.14.0",
#     "torch>=2.0.0",
#     "onnx2torch>=1.5.0",
#     "numpy>=1.24.0",
# ]
# ///
"""Convert SimSwap 512 ONNX to Core ML format."""

import coremltools as ct
import onnx
import torch
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def remove_if_nodes(onnx_model):
    """Try to remove If nodes by running graph optimization."""
    import onnx.shape_inference
    from onnx import optimizer

    # Apply optimizations to remove If nodes where possible
    passes = [
        'eliminate_identity',
        'eliminate_nop_dropout',
        'eliminate_nop_transpose',
        'eliminate_unused_initializer',
        'fuse_consecutive_transposes',
        'fuse_bn_into_conv',
    ]

    try:
        optimized = optimizer.optimize(onnx_model, passes)
        return optimized
    except Exception as e:
        print(f"  Optimization failed: {e}")
        return onnx_model

def convert_via_onnxruntime(onnx_path, output_path):
    """Try to convert by first running through ONNX Runtime to trace."""
    import onnxruntime as ort

    print("\nTrying ONNX Runtime approach...")

    # Create session with fixed input shapes
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(str(onnx_path), sess_options)

    # Get input/output info
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print(f"  Inputs: {[(i.name, i.shape) for i in inputs]}")
    print(f"  Outputs: {[(o.name, o.shape) for o in outputs]}")

    # Create dummy inputs
    target = np.random.randn(1, 3, 512, 512).astype(np.float32)
    source = np.random.randn(1, 512).astype(np.float32)

    # Run inference to verify
    result = session.run(None, {"target": target, "source": source})
    print(f"  Output shape: {result[0].shape}")

    return False  # Can't easily convert from here

def convert_simswap512():
    """Convert SimSwap 512x512 model to Core ML."""
    onnx_path = Path("models/simswap_512_unofficial.onnx")
    output_path = Path("converted_coreml/simswap_512.mlpackage")

    if not onnx_path.exists():
        print(f"Error: {onnx_path} not found")
        return False

    print(f"Loading ONNX model from {onnx_path}...")
    onnx_model = onnx.load(str(onnx_path))

    # Print input/output info
    print("\nModel inputs:")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else "?" for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")

    print("\nModel outputs:")
    for out in onnx_model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else "?" for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")

    # List problematic ops
    print("\nChecking for problematic ops...")
    problematic = []
    for node in onnx_model.graph.node:
        if node.op_type in ['If', 'Loop', 'Scan', 'SequenceConstruct']:
            problematic.append(node.op_type)
    if problematic:
        print(f"  Found problematic ops: {set(problematic)}")
    else:
        print("  No problematic ops found")

    # Try onnx2torch conversion
    print("\nTrying onnx2torch conversion...")
    try:
        from onnx2torch import convert as onnx2torch_convert
        torch_model = onnx2torch_convert(onnx_model)
        torch_model.eval()

        # Trace the model
        example_target = torch.randn(1, 3, 512, 512)
        example_source = torch.randn(1, 512)

        print("  Tracing model...")
        traced_model = torch.jit.trace(torch_model, (example_target, example_source))

        print("  Converting to CoreML...")
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(name="target", shape=(1, 3, 512, 512)),
                ct.TensorType(name="source", shape=(1, 512)),
            ],
            minimum_deployment_target=ct.target.macOS13,
            convert_to="mlprogram",
        )

        print(f"\nSaving to {output_path}...")
        output_path.parent.mkdir(exist_ok=True)
        mlmodel.save(str(output_path))
        print("Done!")
        return True

    except Exception as e:
        print(f"  onnx2torch failed: {e}")

    # The SimSwap model has If ops that prevent conversion
    # These are used for dynamic output size handling
    print("\n" + "="*60)
    print("SimSwap512 cannot be converted to CoreML directly.")
    print("The model contains 'If' operators for dynamic output sizing")
    print("that are not supported by the conversion pipeline.")
    print("\nOptions:")
    print("1. Use ONNX Runtime with CoreML EP (current approach)")
    print("2. Find/train a SimSwap model without dynamic ops")
    print("3. Re-export from PyTorch source with static shapes")
    print("="*60)

    return False

if __name__ == "__main__":
    convert_simswap512()
