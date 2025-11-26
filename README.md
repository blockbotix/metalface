# MetalFace

Real-time face swapping for Apple Silicon Macs using native Metal GPU acceleration.

## Features

- **Live Preview**: Real-time webcam face swapping at 30fps
- **Source Face Loading**: Load any face image as swap source
- **Face Swap**: GPU-accelerated face swap using Metal Performance Shaders
- **Face Enhancement**: Optional GFPGAN-style face restoration
- **Virtual Webcam**: Output to virtual camera for OBS, Zoom, etc.

## Why MetalFace?

Existing face swap solutions (Deep-Live-Cam, FaceFusion) rely on ONNX Runtime with CoreML, but the face swap models don't accelerate properly on CoreML - they fall back to CPU. MetalFace uses Go with native Metal bindings to achieve true GPU acceleration on Apple Silicon.

## Requirements

- macOS 12.0+ (Monterey or later)
- Apple Silicon Mac (M1/M2/M3/M4)
- Go 1.21+
- OpenCV 4.x (via Homebrew)

## Installation

```bash
# Install dependencies
brew install opencv go

# Clone and build
git clone https://github.com/dudu/metalface.git
cd metalface
./scripts/download_models.sh
go build -o metalface ./cmd/metalface
```

## Usage

```bash
# Basic usage - swap your face with source image
./metalface --source face.jpg

# With face enhancement
./metalface --source face.jpg --enhance

# Output to virtual webcam (for OBS)
./metalface --source face.jpg --vcam

# All options
./metalface --source face.jpg --camera 0 --enhance --vcam --fps 30
```

## Options

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--source` | `-s` | Source face image (required) | - |
| `--camera` | `-c` | Camera device index | 0 |
| `--enhance` | `-e` | Enable face enhancement | false |
| `--vcam` | `-v` | Output to virtual camera | false |
| `--preview` | `-p` | Show preview window | true |
| `--fps` | - | Target frames per second | 30 |

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐
│   Webcam    │───▶│ Face Detect  │───▶│  Face Swap  │───▶│  Output  │
│   (gocv)    │    │   (Metal)    │    │   (Metal)   │    │  Window  │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────┘
                                              │
                                              ▼
                                       ┌─────────────┐
                                       │   Virtual   │
                                       │   Webcam    │
                                       └─────────────┘
```

## Performance

| Component | Target | Actual |
|-----------|--------|--------|
| Face Detection | <10ms | TBD |
| Face Swap | <20ms | TBD |
| Enhancement | <15ms | TBD |
| **Total** | **<33ms (30fps)** | TBD |

## Models

MetalFace uses the following neural network models:

- **SCRFD**: Face detection (from InsightFace)
- **ArcFace**: Face embedding extraction
- **Inswapper**: Face swap generator
- **GFPGAN**: Face enhancement (optional)

Run `./scripts/download_models.sh` to download all required models.

## Development

```bash
# Run tests
go test ./...

# Run with race detector
go test -race ./...

# Build for release
go build -ldflags="-s -w" -o metalface ./cmd/metalface
```

## Project Structure

```
metalface/
├── cmd/metalface/      # Main application
├── internal/
│   ├── camera/         # Webcam capture
│   ├── detector/       # Face detection
│   ├── swapper/        # Face swap network
│   ├── enhancer/       # Face enhancement
│   ├── vcam/           # Virtual camera
│   ├── pipeline/       # Processing pipeline
│   └── ui/             # Preview window
├── models/             # Neural network weights
└── scripts/            # Utility scripts
```

## License

MIT

## Acknowledgments

- [go-metal](https://github.com/tsawler/go-metal) - Metal GPU acceleration for Go
- [gocv](https://gocv.io/) - OpenCV bindings for Go
- [InsightFace](https://github.com/deepinsight/insightface) - Face analysis toolkit
- [SimSwap](https://github.com/neuralchen/SimSwap) - Face swap reference
