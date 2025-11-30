#!/usr/bin/env python3
"""
Convert face enhancer models from PyTorch to CoreML.

This script converts GPEN and CodeFormer PyTorch models to native CoreML format
for better performance on Apple Silicon.

Requirements:
    pip install torch coremltools

Usage:
    python convert_enhancer_to_coreml.py --model gpen256
    python convert_enhancer_to_coreml.py --model gpen512
    python convert_enhancer_to_coreml.py --model codeformer

Sources:
    - GPEN: https://github.com/yangxy/GPEN
    - CodeFormer: https://github.com/sczhou/CodeFormer
    - CoreML Tools: https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html
"""

import argparse
import os
import sys
import torch
import coremltools as ct
from pathlib import Path


def convert_gpen(model_size: int = 256, output_dir: str = "converted_coreml"):
    """
    Convert GPEN model to CoreML.

    GPEN (GAN Prior Embedded Network) for face restoration.
    Input: 256x256 or 512x512 aligned face
    Output: Same size restored face
    """
    print(f"Converting GPEN-{model_size} to CoreML...")

    # Clone GPEN repo if not exists
    gpen_path = Path("GPEN")
    if not gpen_path.exists():
        print("Cloning GPEN repository...")
        os.system("git clone https://github.com/yangxy/GPEN.git")

    sys.path.insert(0, str(gpen_path))

    try:
        # Import GPEN model
        from face_model.gpen_model import FullGenerator

        # Create model instance
        model = FullGenerator(
            size=model_size,
            style_dim=512,
            n_mlp=8,
            channel_multiplier=2,
            narrow=1
        )

        # Load weights
        weights_path = f"models/gpen_bfr_{model_size}.pth"
        if not os.path.exists(weights_path):
            print(f"Downloading GPEN-{model_size} weights...")
            # Download from model zoo
            url = f"https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-{model_size}.pth"
            os.system(f"curl -L -o {weights_path} {url}")

        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        # Create example input (NCHW, normalized to [-1, 1])
        example_input = torch.randn(1, 3, model_size, model_size)

        # Trace the model
        print("Tracing model...")
        traced_model = torch.jit.trace(model, example_input)

        # Convert to CoreML
        print("Converting to CoreML...")
        mlmodel = ct.convert(
            traced_model,
            convert_to="mlprogram",
            inputs=[ct.TensorType(name="input", shape=(1, 3, model_size, model_size))],
            outputs=[ct.TensorType(name="output")],
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=ct.precision.FLOAT16,
        )

        # Save
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/gpen_{model_size}.mlpackage"
        mlmodel.save(output_path)
        print(f"Saved to {output_path}")

        return output_path

    except Exception as e:
        print(f"Error converting GPEN: {e}")
        print("\nGPEN has complex architecture with custom CUDA ops.")
        print("Try using the ONNX model with ONNX Runtime CoreML backend instead.")
        return None


def convert_codeformer(output_dir: str = "converted_coreml"):
    """
    Convert CodeFormer model to CoreML.

    CodeFormer uses a codebook lookup transformer for face restoration.
    Input: 512x512 aligned face
    Output: 512x512 restored face
    """
    print("Converting CodeFormer to CoreML...")

    # Clone CodeFormer repo if not exists
    cf_path = Path("CodeFormer")
    if not cf_path.exists():
        print("Cloning CodeFormer repository...")
        os.system("git clone https://github.com/sczhou/CodeFormer.git")

    sys.path.insert(0, str(cf_path))

    try:
        # Import CodeFormer model
        from basicsr.archs.codeformer_arch import CodeFormer

        # Create model instance
        model = CodeFormer(
            dim_embd=512,
            n_head=8,
            n_layers=9,
            codebook_size=1024,
            latent_size=256,
            connect_list=['32', '64', '128', '256'],
            fix_modules=['quantize', 'generator']
        )

        # Load weights
        weights_path = "models/codeformer.pth"
        if not os.path.exists(weights_path):
            print("Downloading CodeFormer weights...")
            url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
            os.system(f"curl -L -o {weights_path} {url}")

        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        # CodeFormer has two inputs: x (image) and w (fidelity weight)
        # For CoreML, we'll bake in a default fidelity weight
        class CodeFormerWrapper(torch.nn.Module):
            def __init__(self, model, w=0.5):
                super().__init__()
                self.model = model
                self.w = w

            def forward(self, x):
                return self.model(x, w=self.w, adain=True)[0]

        wrapped_model = CodeFormerWrapper(model, w=0.5)

        # Create example input (NCHW, normalized to [-1, 1])
        example_input = torch.randn(1, 3, 512, 512)

        # Trace the model
        print("Tracing model...")
        traced_model = torch.jit.trace(wrapped_model, example_input)

        # Convert to CoreML
        print("Converting to CoreML...")
        mlmodel = ct.convert(
            traced_model,
            convert_to="mlprogram",
            inputs=[ct.TensorType(name="x", shape=(1, 3, 512, 512))],
            outputs=[ct.TensorType(name="output")],
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=ct.precision.FLOAT16,
        )

        # Save
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/codeformer.mlpackage"
        mlmodel.save(output_path)
        print(f"Saved to {output_path}")

        return output_path

    except Exception as e:
        print(f"Error converting CodeFormer: {e}")
        print("\nCodeFormer has transformer architecture that may have unsupported ops.")
        print("Common issues:")
        print("  - Transformer attention layers may need special handling")
        print("  - Vector quantization codebook lookup may not be supported")
        print("\nTry using the ONNX model with ONNX Runtime CoreML backend instead.")
        return None


# Define RRDBNet architecture locally to avoid basicsr dependency issues
class ResidualDenseBlock(torch.nn.Module):
    """Residual Dense Block for RRDB"""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(torch.nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(torch.nn.Module):
    """RRDBNet architecture for Real-ESRGAN"""
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        self.conv_first = torch.nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = torch.nn.ModuleList([RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # Upsampling
        self.conv_up1 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = torch.nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = feat
        for block in self.body:
            body_feat = block(body_feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat
        # Upsample
        feat = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def convert_realesrgan(output_dir: str = "converted_coreml"):
    """
    Convert Real-ESRGAN model to CoreML.

    Real-ESRGAN is a general image upscaler.
    Input: any size image
    Output: 4x upscaled image
    """
    print("Converting Real-ESRGAN to CoreML...")

    try:
        # Real-ESRGAN x4v3 architecture
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )

        weights_path = "models/realesr-general-x4v3.pth"
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            # Handle 'params_ema' key if present
            if 'params_ema' in state_dict:
                state_dict = state_dict['params_ema']
            elif 'params' in state_dict:
                state_dict = state_dict['params']
            model.load_state_dict(state_dict, strict=False)
            print("Loaded weights from", weights_path)
        else:
            print(f"Warning: {weights_path} not found, using random weights")

        model.eval()

        # Use fixed input size for CoreML (128x128 -> 512x512)
        example_input = torch.randn(1, 3, 128, 128)

        print("Tracing model...")
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)

        print("Converting to CoreML...")
        mlmodel = ct.convert(
            traced_model,
            convert_to="mlprogram",
            inputs=[ct.TensorType(name="input", shape=(1, 3, 128, 128))],
            outputs=[ct.TensorType(name="output")],
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=ct.precision.FLOAT16,
        )

        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/realesrgan_x4.mlpackage"
        mlmodel.save(output_path)
        print(f"Saved to {output_path}")

        return output_path

    except Exception as e:
        print(f"Error converting Real-ESRGAN: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert face enhancer models to CoreML")
    parser.add_argument("--model", type=str, required=True,
                       choices=["gpen256", "gpen512", "codeformer", "realesrgan"],
                       help="Model to convert")
    parser.add_argument("--output", type=str, default="converted_coreml",
                       help="Output directory for converted models")

    args = parser.parse_args()

    print(f"Converting {args.model} to CoreML...")
    print("=" * 50)

    if args.model == "gpen256":
        convert_gpen(256, args.output)
    elif args.model == "gpen512":
        convert_gpen(512, args.output)
    elif args.model == "codeformer":
        convert_codeformer(args.output)
    elif args.model == "realesrgan":
        convert_realesrgan(args.output)


if __name__ == "__main__":
    main()
