package swapper

import (
	"fmt"
	"image"
	"math"

	"gocv.io/x/gocv"

	"github.com/dudu/metalface/internal/coreml"
)

// ArcFaceEncoderCoreML extracts face embeddings using ArcFace with CoreML
type ArcFaceEncoderCoreML struct {
	session *coreml.Session
}

// NewArcFaceEncoderCoreML creates a new ArcFace encoder using CoreML
func NewArcFaceEncoderCoreML(modelPath string) (*ArcFaceEncoderCoreML, error) {
	// w600k_r50.onnx has input.1 which becomes input_1 in CoreML
	// Output is var_1110 (512-dim embedding)
	inputNames := []string{"input_1"}
	outputNames := []string{"var_1110"}

	session, err := coreml.NewSession(modelPath, inputNames, outputNames)
	if err != nil {
		return nil, fmt.Errorf("failed to create ArcFace CoreML session: %w", err)
	}

	return &ArcFaceEncoderCoreML{
		session: session,
	}, nil
}

// Extract computes the 512-dim embedding from an aligned 112x112 face
func (e *ArcFaceEncoderCoreML) Extract(alignedFace gocv.Mat) (*Embedding, error) {
	if alignedFace.Rows() != 112 || alignedFace.Cols() != 112 {
		return nil, fmt.Errorf("expected 112x112 input, got %dx%d", alignedFace.Cols(), alignedFace.Rows())
	}

	// Preprocess
	inputData := e.preprocess(alignedFace)

	// Run inference
	output, err := e.session.Run(inputData, []int64{1, 3, 112, 112}, 512)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// Normalize embedding
	embedding := e.normalizeEmbedding(output)

	return embedding, nil
}

// preprocess converts aligned face to model input format
// Matches insightface ArcFace preprocessing:
// blob = cv2.dnn.blobFromImage(img, 1.0/input_std, input_size, (input_mean,)*3, swapRB=True)
// where input_std=127.5, input_mean=127.5, resulting in (x-127.5)/127.5 = (x/127.5 - 1)
func (e *ArcFaceEncoderCoreML) preprocess(img gocv.Mat) []float32 {
	// ArcFace uses [-1, 1] normalization: (pixel - 127.5) / 127.5
	// BlobFromImage formula: out = (in - mean) * scale
	// We need: out = (in - 127.5) / 127.5 = in/127.5 - 1
	// So: scale = 1/127.5, mean = 127.5
	blob := gocv.BlobFromImage(img, 1.0/127.5, image.Pt(112, 112),
		gocv.NewScalar(127.5, 127.5, 127.5, 0), true, false)
	defer blob.Close()

	blobData := blob.ToBytes()
	return bytesToFloat32Slice(blobData)
}

// normalizeEmbedding L2-normalizes the embedding
func (e *ArcFaceEncoderCoreML) normalizeEmbedding(data []float32) *Embedding {
	var embedding Embedding

	// Compute L2 norm
	var norm float64
	for _, v := range data[:512] {
		norm += float64(v * v)
	}
	norm = math.Sqrt(norm)

	if norm < 1e-10 {
		norm = 1
	}

	// Normalize
	for i := 0; i < 512; i++ {
		embedding[i] = data[i] / float32(norm)
	}

	return &embedding
}

// Close releases encoder resources
func (e *ArcFaceEncoderCoreML) Close() error {
	return e.session.Destroy()
}
