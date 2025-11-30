#!/usr/bin/env python3
"""Convert GPEN-BFR-256 ONNX to Core ML format using onnx-coreml."""

import coremltools as ct
import onnx
import torch
import numpy as np
from onnx2torch import convert as onnx2torch_convert
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def simplify_and_convert():
    """Try simplifying ONNX model first, then convert via onnx2torch"""
    onnx_path = Path("models/gpen_bfr_256.onnx")
    output_path = Path("converted_coreml/gpen_bfr_256.mlpackage")
    
    print(f"Loading and simplifying ONNX model...")
    
    # Try onnxsim to simplify the model first
    try:
        import onnxsim
        onnx_model = onnx.load(str(onnx_path))
        simplified_model, check = onnxsim.simplify(
            onnx_model,
            check_n=0,  # Don't check
            skip_fuse_bn=False,
            skip_constant_folding=False,
        )
        if check:
            print("Simplified model validated successfully")
            onnx_model = simplified_model
        else:
            print("Model simplification validation failed, using original")
    except ImportError:
        print("onnxsim not available, loading original model")
        onnx_model = onnx.load(str(onnx_path))
    except Exception as e:
        print(f"Simplification failed: {e}, using original model")
        onnx_model = onnx.load(str(onnx_path))
    
    print(f"Converting ONNX to PyTorch...")
    try:
        torch_model = onnx2torch_convert(onnx_model)
        torch_model.eval()
    except Exception as e:
        print(f"onnx2torch conversion failed: {e}")
        return False
    
    # GPEN-256 expects 256x256 input
    example_input = torch.randn(1, 3, 256, 256)
    
    print(f"Tracing PyTorch model...")
    traced_model = torch.jit.trace(torch_model, example_input)
    
    print(f"Converting to Core ML...")
    ct_inputs = [ct.TensorType(name="input", shape=(1, 3, 256, 256))]
    
    mlmodel = ct.convert(
        traced_model,
        inputs=ct_inputs,
        minimum_deployment_target=ct.target.macOS13,
        convert_to="mlprogram",
    )
    
    print(f"Saving to {output_path}...")
    mlmodel.save(str(output_path))
    print("Done!")
    return True

if __name__ == "__main__":
    simplify_and_convert()
