#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "coremltools>=7.0",
#     "numpy>=1.24.0",
# ]
# ///
"""Direct ONNX to CoreML conversion for SCRFD without PyTorch intermediate."""

import coremltools as ct
import numpy as np
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR = Path(__file__).parent.parent / "converted_coreml"

def convert_scrfd():
    onnx_path = MODELS_DIR / "scrfd_10g.onnx"
    output_path = OUTPUT_DIR / "scrfd_10g_direct.mlpackage"

    print(f"Converting SCRFD from {onnx_path}")

    # Convert directly from ONNX using unified convert API
    mlmodel = ct.convert(
        str(onnx_path),
        source="onnx",
        minimum_deployment_target=ct.target.macOS13,
        convert_to="mlprogram",
    )

    # Print model info
    spec = mlmodel.get_spec()
    print("\nModel outputs:")
    for output in spec.description.output:
        print(f"  {output.name}")

    print(f"\nSaving to {output_path}")
    mlmodel.save(str(output_path))

    # Test inference
    print("\nTesting inference...")
    test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

    # Get input name
    input_name = spec.description.input[0].name
    print(f"Input name: {input_name}")

    results = mlmodel.predict({input_name: test_input})

    print("\nOutput statistics:")
    for name, arr in results.items():
        arr = np.array(arr)
        print(f"  {name}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}")

if __name__ == "__main__":
    convert_scrfd()
