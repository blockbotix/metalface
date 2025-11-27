#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "coremltools>=7.0",
#     "onnxruntime>=1.15.0",
#     "numpy>=1.24.0",
#     "pillow>=10.0.0",
# ]
# ///
"""Compare SCRFD outputs between ONNX and CoreML."""

import coremltools as ct
import onnxruntime as ort
import numpy as np
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
COREML_DIR = Path(__file__).parent.parent / "converted_coreml"

def preprocess_input(size=640):
    """Create a test input image similar to what Go sends."""
    # Create an image that matches Go preprocessing: (x - 127.5) / 128
    # This creates a dark image similar to what Go is sending
    np.random.seed(42)

    # Simulate a real image preprocessing:
    # - Load image, resize, pad to 640x640
    # - Values between 0-255, then normalized
    # Let's create values similar to Go: min=-0.9961, max=-0.7930
    # which corresponds to pixel values ~0-26

    # But let's also test with a face-like pattern
    # First test: all dark (like padded area)
    img = np.zeros((1, 3, size, size), dtype=np.float32)
    img = (img - 127.5) / 128.0  # This gives all -0.996

    print(f"Test input: min={img.min():.4f}, max={img.max():.4f}")
    return img

def preprocess_input_random(size=640):
    """Create random normalized input."""
    np.random.seed(42)
    img = np.random.randn(1, 3, size, size).astype(np.float32) * 0.5
    print(f"Random input: min={img.min():.4f}, max={img.max():.4f}")
    return img

def test_onnx():
    """Run ONNX model and print outputs."""
    print("=" * 60)
    print("ONNX Runtime")
    print("=" * 60)

    onnx_path = MODELS_DIR / "scrfd_10g.onnx"
    session = ort.InferenceSession(str(onnx_path))

    # Print input/output info
    print("\nInputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape}")

    print("\nOutputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape}")

    # Run inference
    input_data = preprocess_input()
    input_name = session.get_inputs()[0].name

    results = session.run(None, {input_name: input_data})

    print("\nOutput values:")
    for i, out in enumerate(session.get_outputs()):
        arr = results[i]
        print(f"  {out.name}: shape={arr.shape}, min={arr.min():.6f}, max={arr.max():.6f}, nonzero={np.count_nonzero(arr)}")

    return results, [out.name for out in session.get_outputs()]

def test_coreml():
    """Run CoreML model and print outputs."""
    print("\n" + "=" * 60)
    print("CoreML - Dark input (like Go)")
    print("=" * 60)

    coreml_path = COREML_DIR / "scrfd_10g.mlpackage"
    model = ct.models.MLModel(str(coreml_path))

    spec = model.get_spec()
    input_name = spec.description.input[0].name

    # Test with dark input (like Go)
    input_data = preprocess_input()
    results = model.predict({input_name: input_data})

    print("\nOutput values (dark input):")
    for name in sorted(results.keys()):
        arr = np.array(results[name])
        print(f"  {name}: shape={arr.shape}, min={arr.min():.6f}, max={arr.max():.6f}, nonzero={np.count_nonzero(arr)}")

    # Also test with random input
    print("\n" + "=" * 60)
    print("CoreML - Random input")
    print("=" * 60)

    input_data_random = preprocess_input_random()
    results_random = model.predict({input_name: input_data_random})

    print("\nOutput values (random input):")
    for name in sorted(results_random.keys()):
        arr = np.array(results_random[name])
        print(f"  {name}: shape={arr.shape}, min={arr.min():.6f}, max={arr.max():.6f}, nonzero={np.count_nonzero(arr)}")

    return results_random

def main():
    onnx_results, onnx_names = test_onnx()
    coreml_results = test_coreml()

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    # Map ONNX output names to expected sizes for scores
    # ONNX outputs: 448, 471, 494 (scores), 451, 474, 497 (bbox), 454, 477, 500 (kps)
    # for strides 8, 16, 32

    print("\nScore outputs (should have values 0-1):")
    for i, name in enumerate(["448", "471", "494"]):
        onnx_arr = onnx_results[i]
        print(f"  ONNX {name}: min={onnx_arr.min():.6f}, max={onnx_arr.max():.6f}")

    # Find corresponding CoreML outputs by size
    # Stride 8: 12800 scores (80*80*2)
    # Stride 16: 3200 scores (40*40*2)
    # Stride 32: 800 scores (20*20*2)
    score_sizes = [12800, 3200, 800]
    for size in score_sizes:
        for name, arr in coreml_results.items():
            arr = np.array(arr)
            if arr.size == size:
                print(f"  CoreML {name} (size={size}): min={arr.min():.6f}, max={arr.max():.6f}")

if __name__ == "__main__":
    main()
