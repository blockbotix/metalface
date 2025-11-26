# MetalFace - Development Guidelines

## Project Overview

MetalFace is a real-time face swapping application written in Go, designed specifically for Apple Silicon Macs. It leverages Metal GPU acceleration via the go-metal library to achieve real-time performance that Python/ONNX-based solutions cannot match on macOS.

## Why Go + Metal?

The existing Python face-swap solutions (Deep-Live-Cam, FaceFusion) use ONNX Runtime with CoreML backend, but the inswapper model doesn't accelerate properly on CoreML - it falls back to CPU, making real-time impossible. By using Go with native Metal bindings, we bypass this limitation entirely.

## Core Requirements

### 1. Live Preview
- Real-time webcam capture at 30fps minimum
- Low-latency preview window showing swapped face
- Frame timing display for performance monitoring

### 2. Source Face Loading
- Load source face from image file (PNG, JPG)
- Extract and cache face embedding on load
- Support multiple source faces for selection

### 3. Face Swap Pipeline
- Face detection using SCRFD or similar (via gocv/OpenCV DNN)
- Face alignment and landmark detection
- Face embedding extraction (ArcFace-style encoder)
- Face swap generation (inswapper-style generator)
- Face blending with soft mask

### 4. Face Enhancement (Optional)
- GFPGAN-style face restoration
- Toggle on/off for performance vs quality tradeoff

### 5. Virtual Webcam Output
- Create virtual camera device visible to other apps
- Compatible with OBS, Zoom, Google Meet, etc.
- Use OBS Virtual Camera or similar on macOS

## Architecture

```
metalface/
├── cmd/
│   └── metalface/          # Main application entry point
│       └── main.go
├── internal/
│   ├── camera/             # Webcam capture (gocv)
│   │   └── capture.go
│   ├── detector/           # Face detection
│   │   ├── scrfd.go        # SCRFD face detector
│   │   └── landmarks.go    # Facial landmark detection
│   ├── swapper/            # Face swap neural network
│   │   ├── encoder.go      # ArcFace embedding extractor
│   │   ├── generator.go    # Swap generator network
│   │   └── blend.go        # Face blending/masking
│   ├── enhancer/           # Face enhancement
│   │   └── gfpgan.go       # GFPGAN restoration
│   ├── vcam/               # Virtual camera output
│   │   └── output.go       # Virtual webcam device
│   ├── pipeline/           # Processing pipeline
│   │   └── pipeline.go     # Orchestrates all components
│   └── ui/                 # User interface
│       └── window.go       # Preview window
├── models/                 # Neural network weights
│   ├── scrfd_10g.onnx      # Face detector
│   ├── arcface.onnx        # Face encoder
│   ├── inswapper.onnx      # Face swap generator
│   └── gfpgan.onnx         # Face enhancer
├── scripts/
│   └── download_models.sh  # Model download script
├── go.mod
├── go.sum
├── CLAUDE.md               # This file
└── README.md
```

## Key Dependencies

```go
require (
    github.com/tsawler/go-metal  // Metal GPU acceleration
    gocv.io/x/gocv               // OpenCV bindings for camera/detection
    // Note: May need CGo for some Metal/macOS specific code
)
```

## Performance Targets

- **Face Detection**: < 10ms per frame
- **Face Swap**: < 20ms per frame
- **Total Pipeline**: < 33ms (30fps target)
- **Memory**: < 2GB GPU memory

## Model Conversion

The existing ONNX models need to be converted to a format go-metal can use:

1. **Option A**: Use go-metal's ONNX import if supported
2. **Option B**: Convert ONNX to custom format using Python, load weights in Go
3. **Option C**: Use gocv's DNN module for ONNX inference (may be slower)

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

## CLI Interface

```bash
# Basic usage
metalface --source face.jpg

# With options
metalface --source face.jpg --camera 0 --enhance --vcam

# Flags
--source, -s     Source face image (required)
--camera, -c     Camera device index (default: 0)
--enhance, -e    Enable face enhancement
--vcam, -v       Output to virtual camera
--preview, -p    Show preview window (default: true)
--fps            Target FPS (default: 30)
```

## Known Challenges

1. **Model Conversion**: ONNX to go-metal format may require custom tooling
2. **Virtual Camera**: macOS virtual camera requires kernel extension or OBS approach
3. **Face Alignment**: Need precise affine transforms for good swap quality
4. **Blending**: Soft mask generation is critical for realistic results

## Resources

- go-metal: https://github.com/tsawler/go-metal
- gocv: https://gocv.io/
- InsightFace (reference): https://github.com/deepinsight/insightface
- SimSwap (reference): https://github.com/neuralchen/SimSwap

## Testing

```bash
# Run tests
go test ./...

# Run with race detector
go test -race ./...

# Benchmark
go test -bench=. ./...
```

## Build

```bash
# Development build
go build -o metalface ./cmd/metalface

# Release build with optimizations
go build -ldflags="-s -w" -o metalface ./cmd/metalface
```
