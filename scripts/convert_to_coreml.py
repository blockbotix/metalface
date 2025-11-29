#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "coremltools>=7.0",
#     "onnx>=1.14.0",
#     "onnx2torch>=1.5.0",
#     "torch>=2.0.0",
#     "numpy>=1.24.0",
# ]
# ///
"""
Convert ML models in metalface/models to Core ML format.
Supports ONNX and PyTorch models.
ONNX models are first converted to PyTorch, then to Core ML.
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import coremltools as ct
import onnx
import torch
import numpy as np
from onnx2torch import convert as onnx2torch_convert

# Paths
MODELS_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR = Path(__file__).parent.parent / "converted_coreml"

def convert_onnx_to_coreml(onnx_path: Path) -> tuple[bool, str]:
    """Convert ONNX model to Core ML via PyTorch."""
    output_path = OUTPUT_DIR / f"{onnx_path.stem}.mlpackage"

    try:
        print(f"  Loading ONNX model...")
        onnx_model = onnx.load(str(onnx_path))

        # Get input shapes from model
        input_info = {}
        for inp in onnx_model.graph.input:
            shape = []
            dtype = inp.type.tensor_type.elem_type
            for i, dim in enumerate(inp.type.tensor_type.shape.dim):
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    # Default dynamic dims: batch=1, channels=3, H/W=640 (for detection)
                    if i == 0:
                        shape.append(1)  # batch
                    elif i == 1:
                        shape.append(3)  # channels
                    else:
                        shape.append(640)  # height/width for face detection
            input_info[inp.name] = {"shape": shape, "dtype": dtype}

        print(f"  Input shapes: {input_info}")

        print(f"  Converting ONNX to PyTorch...")
        torch_model = onnx2torch_convert(onnx_model)
        torch_model.eval()

        # Create example inputs for tracing
        example_inputs = []
        for name, info in input_info.items():
            shape = info["shape"]
            # Ensure valid shape (no zeros)
            shape = [max(1, s) for s in shape]
            example_inputs.append(torch.randn(*shape))

        print(f"  Tracing PyTorch model...")
        if len(example_inputs) == 1:
            traced_model = torch.jit.trace(torch_model, example_inputs[0])
        else:
            traced_model = torch.jit.trace(torch_model, tuple(example_inputs))

        print(f"  Converting to Core ML...")
        # Build input descriptions
        ct_inputs = []
        for i, (name, info) in enumerate(input_info.items()):
            shape = [max(1, s) for s in info["shape"]]
            ct_inputs.append(ct.TensorType(name=name, shape=shape))

        mlmodel = ct.convert(
            traced_model,
            inputs=ct_inputs,
            minimum_deployment_target=ct.target.macOS13,
            convert_to="mlprogram",
        )

        print(f"  Saving to {output_path}...")
        mlmodel.save(str(output_path))

        return True, f"Successfully converted to {output_path.name}"

    except Exception as e:
        import traceback
        return False, f"Failed: {str(e)[:300]}"


def convert_gfpgan_onnx_to_coreml(onnx_path: Path) -> tuple[bool, str]:
    """
    Convert GFPGAN ONNX model to Core ML.
    GFPGAN expects 512x512 input normalized to [-1,1], outputs [-1,1].
    """
    output_path = OUTPUT_DIR / f"{onnx_path.stem}.mlpackage"

    try:
        print(f"  Loading GFPGAN ONNX model...")
        onnx_model = onnx.load(str(onnx_path))

        print(f"  Converting ONNX to PyTorch...")
        torch_model = onnx2torch_convert(onnx_model)
        torch_model.eval()

        # GFPGAN expects 512x512 input
        example_input = torch.randn(1, 3, 512, 512)

        print(f"  Tracing PyTorch model...")
        traced_model = torch.jit.trace(torch_model, example_input)

        print(f"  Converting to Core ML...")
        ct_inputs = [ct.TensorType(name="input", shape=(1, 3, 512, 512))]

        mlmodel = ct.convert(
            traced_model,
            inputs=ct_inputs,
            minimum_deployment_target=ct.target.macOS13,
            convert_to="mlprogram",
        )

        print(f"  Saving to {output_path}...")
        mlmodel.save(str(output_path))

        return True, f"Successfully converted to {output_path.name}"

    except Exception as e:
        import traceback
        return False, f"Failed: {str(e)[:300]}"


def convert_pytorch_gfpgan(pth_path: Path) -> tuple[bool, str]:
    """
    Convert GFPGAN PyTorch model to Core ML.
    GFPGAN requires the model architecture to be defined.
    """
    # If ONNX version exists, use that instead
    onnx_path = pth_path.parent / f"{pth_path.stem}.onnx"
    if onnx_path.exists():
        return convert_gfpgan_onnx_to_coreml(onnx_path)

    # Also check for _1.4 version
    onnx_path_v14 = pth_path.parent / "gfpgan_1.4.onnx"
    if onnx_path_v14.exists():
        return convert_gfpgan_onnx_to_coreml(onnx_path_v14)

    return False, "GFPGAN .pth requires model architecture code. Convert to ONNX first using scripts/convert_gfpgan.py"


def main():
    print("=" * 60)
    print("Core ML Model Converter for Metalface")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Find all models
    model_files = []
    for ext in ['*.onnx', '*.pt', '*.pth', '*.pb', '*.h5']:
        model_files.extend(MODELS_DIR.glob(ext))

    print(f"\nFound {len(model_files)} model(s):\n")

    results = []

    for model_path in sorted(model_files):
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"\n[{model_path.name}] ({size_mb:.1f} MB)")
        print("-" * 50)

        if model_path.suffix == '.onnx':
            if 'gfpgan' in model_path.name.lower():
                success, message = convert_gfpgan_onnx_to_coreml(model_path)
            else:
                success, message = convert_onnx_to_coreml(model_path)
        elif model_path.suffix in ['.pt', '.pth']:
            if 'gfpgan' in model_path.name.lower():
                success, message = convert_pytorch_gfpgan(model_path)
            else:
                success, message = False, "Unknown PyTorch model architecture"
        else:
            success, message = False, f"Unsupported format: {model_path.suffix}"

        status = "SUCCESS" if success else "SKIPPED"
        print(f"  Result: {status} - {message}")
        results.append((model_path.name, status, message))

    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r[1] == "SUCCESS"]
    failed = [r for r in results if r[1] != "SUCCESS"]

    print(f"\nSuccessfully converted: {len(successful)}/{len(results)}")

    if successful:
        print("\nConverted models:")
        for name, _, msg in successful:
            print(f"  - {name} -> {name.rsplit('.', 1)[0]}.mlpackage")

    if failed:
        print("\nSkipped/Failed models:")
        for name, _, msg in failed:
            print(f"  - {name}: {msg}")

    # List output files
    print(f"\nOutput directory contents:")
    if OUTPUT_DIR.exists():
        for f in OUTPUT_DIR.iterdir():
            size = f.stat().st_size / (1024 * 1024) if f.is_file() else sum(
                p.stat().st_size for p in f.rglob('*') if p.is_file()
            ) / (1024 * 1024)
            print(f"  - {f.name} ({size:.1f} MB)")

    return 0 if successful else 1


if __name__ == "__main__":
    sys.exit(main())
