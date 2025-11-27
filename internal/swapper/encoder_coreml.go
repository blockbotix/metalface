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
	inputNames := []string{"input"}
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
func (e *ArcFaceEncoderCoreML) preprocess(img gocv.Mat) []float32 {
	// Convert BGR to RGB
	rgb := gocv.NewMat()
	gocv.CvtColor(img, &rgb, gocv.ColorBGRToRGB)
	defer rgb.Close()

	// Convert to float and normalize to [0, 1]
	floatImg := gocv.NewMat()
	rgb.ConvertTo(&floatImg, gocv.MatTypeCV32FC3)
	defer floatImg.Close()

	// Create blob (HWC to NCHW)
	blob := gocv.BlobFromImage(floatImg, 1.0/255.0, image.Pt(112, 112),
		gocv.NewScalar(0, 0, 0, 0), false, false)
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
