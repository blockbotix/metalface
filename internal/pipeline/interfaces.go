package pipeline

import (
	"gocv.io/x/gocv"

	"github.com/dudu/metalface/internal/detector"
	"github.com/dudu/metalface/internal/swapper"
)

// Backend represents the inference backend to use
type Backend string

const (
	BackendONNX   Backend = "onnx"
	BackendCoreML Backend = "coreml"
)

// ModelType represents the face swap model to use
type ModelType string

const (
	ModelInswapper  ModelType = "inswapper"
	ModelSimSwap512 ModelType = "simswap512"
)

// FaceDetector interface for face detection
type FaceDetector interface {
	Detect(img gocv.Mat) ([]detector.Face, error)
	Close() error
}

// LandmarkDetector interface for 106-point landmark detection
type LandmarkDetector interface {
	Detect(img gocv.Mat, face *detector.Face) error
	Close() error
}

// FaceEncoder interface for face embedding extraction
type FaceEncoder interface {
	Extract(alignedFace gocv.Mat) (*swapper.Embedding, error)
	Close() error
}

// FaceSwapper interface for face swapping
type FaceSwapper interface {
	Swap(targetFace gocv.Mat, sourceEmbedding *swapper.Embedding) (gocv.Mat, error)
	Close() error
}
