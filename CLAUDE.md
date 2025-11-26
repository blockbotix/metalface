# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Development build
go build -o metalface ./cmd/metalface

# Release build
go build -ldflags="-s -w" -o metalface ./cmd/metalface

# Run tests
go test ./...

# Run with race detector
go test -race ./...

# Benchmark
go test -bench=. ./...

# Download required models (before first run)
./scripts/download_models.sh
```

## Why Go + Metal?

Existing Python face-swap solutions (Deep-Live-Cam, FaceFusion) use ONNX Runtime with CoreML backend, but the inswapper model doesn't accelerate properly on CoreML—it falls back to CPU, making real-time impossible. By using Go with native Metal bindings via go-metal, we bypass this limitation entirely for Apple Silicon Macs.

## Current State

The project is scaffolded but implementation has not started:
- `cmd/metalface/main.go` has CLI parsing only
- `internal/` packages are empty directories
- Dependencies need to be added to `go.mod`
- Models need to be downloaded via `scripts/download_models.sh`

## Architecture

Internal packages (all need implementation):
- **camera/** - Webcam capture using gocv
- **detector/** - SCRFD face detection and landmark extraction
- **swapper/** - ArcFace encoder + inswapper generator + blending
- **enhancer/** - GFPGAN face restoration (optional)
- **vcam/** - Virtual camera output for OBS/Zoom
- **pipeline/** - Orchestrates all components
- **ui/** - Preview window display

## Key Dependencies

```go
require (
    github.com/tsawler/go-metal  // Metal GPU acceleration
    gocv.io/x/gocv               // OpenCV bindings for camera/detection
)
```

Note: CGo required for Metal/macOS specific code.

## Development Phases

### Phase 1: Camera + Preview
- [ ] Webcam capture with gocv
- [ ] Preview window display
- [ ] FPS counter

### Phase 2: Face Detection
- [ ] Load SCRFD model
- [ ] Detect faces in frame
- [ ] Draw bounding boxes

### Phase 3: Face Embedding
- [ ] Load ArcFace model
- [ ] Extract embeddings from detected faces
- [ ] Cache source face embedding

### Phase 4: Face Swap
- [ ] Load inswapper model
- [ ] Implement swap generator
- [ ] Face alignment pre/post processing
- [ ] Soft mask blending

### Phase 5: Enhancement
- [ ] Load GFPGAN model
- [ ] Apply face restoration
- [ ] Toggle option

### Phase 6: Virtual Camera
- [ ] Research macOS virtual camera options
- [ ] Implement output to virtual device
- [ ] Test with OBS

## Model Conversion

ONNX models need conversion to a format go-metal can use:

1. **Option A**: Use go-metal's ONNX import if supported
2. **Option B**: Convert ONNX to custom format using Python, load weights in Go
3. **Option C**: Use gocv's DNN module for ONNX inference (may be slower)

## Performance Targets

- Face Detection: < 10ms per frame
- Face Swap: < 20ms per frame
- Total Pipeline: < 33ms (30fps target)
- Memory: < 2GB GPU memory

## Known Challenges

1. **Model Conversion**: ONNX to go-metal format may require custom tooling
2. **Virtual Camera**: macOS virtual camera requires kernel extension or OBS approach
3. **Face Alignment**: Need precise affine transforms for good swap quality
4. **Blending**: Soft mask generation is critical for realistic results
