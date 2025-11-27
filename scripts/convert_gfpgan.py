#!/usr/bin/env python3
"""
Convert GFPGAN PyTorch model to ONNX format.
Requires: pip install gfpgan torch onnx
"""

import torch
import torch.nn as nn
import argparse
import os

def convert_gfpgan_to_onnx(pth_path: str, onnx_path: str):
    """Convert GFPGAN .pth to .onnx"""

    try:
        from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
    except ImportError:
        print("Installing gfpgan...")
        os.system("pip install gfpgan")
        from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

    # GFPGAN v1.4 architecture
    model = GFPGANv1Clean(
        out_size=512,
        num_style_feat=512,
        channel_multiplier=2,
        decoder_load_path=None,
        fix_decoder=False,
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        narrow=1,
        sft_half=True
    )

    # Load weights
    print(f"Loading weights from {pth_path}...")
    checkpoint = torch.load(pth_path, map_location='cpu')

    if 'params_ema' in checkpoint:
        model.load_state_dict(checkpoint['params_ema'], strict=True)
    elif 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    model.eval()

    # Create dummy input (512x512 RGB image, normalized to [-1, 1])
    dummy_input = torch.randn(1, 3, 512, 512)

    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch'},
            'output': {0: 'batch'}
        },
        opset_version=14,
        do_constant_folding=True
    )

    print(f"Successfully exported to {onnx_path}")

    # Verify
    import onnx
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("ONNX model verified successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert GFPGAN to ONNX')
    parser.add_argument('--input', '-i', default='models/gfpgan.pth', help='Input .pth file')
    parser.add_argument('--output', '-o', default='models/gfpgan.onnx', help='Output .onnx file')
    args = parser.parse_args()

    convert_gfpgan_to_onnx(args.input, args.output)
