#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

mkdir -p "$MODELS_DIR"

echo "Downloading models to $MODELS_DIR..."

# Face detector - SCRFD 10G
# Source: InsightFace
SCRFD_URL="https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_10g_bnkps.onnx"
if [ ! -f "$MODELS_DIR/scrfd_10g.onnx" ]; then
    echo "Downloading SCRFD face detector..."
    curl -L -o "$MODELS_DIR/scrfd_10g.onnx" "$SCRFD_URL"
else
    echo "SCRFD already exists, skipping..."
fi

# ArcFace encoder - w600k_r50
# Source: InsightFace
ARCFACE_URL="https://github.com/deepinsight/insightface/releases/download/v0.7/w600k_r50.onnx"
if [ ! -f "$MODELS_DIR/arcface.onnx" ]; then
    echo "Downloading ArcFace encoder..."
    curl -L -o "$MODELS_DIR/arcface.onnx" "$ARCFACE_URL"
else
    echo "ArcFace already exists, skipping..."
fi

# Inswapper - face swap model
# Note: This model requires agreement to terms
# You may need to download manually from: https://huggingface.co/deepinsight/inswapper
INSWAPPER_URL="https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx"
if [ ! -f "$MODELS_DIR/inswapper.onnx" ]; then
    echo "Downloading Inswapper model..."
    echo "Note: If this fails, download manually from HuggingFace"
    curl -L -o "$MODELS_DIR/inswapper.onnx" "$INSWAPPER_URL" || echo "Manual download required"
else
    echo "Inswapper already exists, skipping..."
fi

# GFPGAN - face enhancement
# Source: TencentARC
GFPGAN_URL="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
if [ ! -f "$MODELS_DIR/gfpgan.pth" ]; then
    echo "Downloading GFPGAN model..."
    curl -L -o "$MODELS_DIR/gfpgan.pth" "$GFPGAN_URL"
else
    echo "GFPGAN already exists, skipping..."
fi

echo ""
echo "Model download complete!"
echo ""
echo "Models directory contents:"
ls -lh "$MODELS_DIR"
