package swapper

import (
	"fmt"
	"image"
	"math"

	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"

	"github.com/dudu/metalface/internal/inference"
)

// Embedding represents a 512-dimensional face embedding
type Embedding [512]float32

// ArcFaceEncoder extracts face embeddings using ArcFace
type ArcFaceEncoder struct {
	session *inference.Session
}

// NewArcFaceEncoder creates a new ArcFace encoder
func NewArcFaceEncoder(modelPath string) (*ArcFaceEncoder, error) {
	// ArcFace has 1 input and 1 output
	inputNames := []string{"input.1"}
	outputNames := []string{"683"} // output node name from model

	session, err := inference.NewSession(modelPath, inputNames, outputNames)
	if err != nil {
		return nil, fmt.Errorf("failed to create ArcFace session: %w", err)
	}

	return &ArcFaceEncoder{
		session: session,
	}, nil
}

// Extract computes the 512-dim embedding from an aligned 112x112 face
func (e *ArcFaceEncoder) Extract(alignedFace gocv.Mat) (*Embedding, error) {
	// Verify input size
	if alignedFace.Rows() != 112 || alignedFace.Cols() != 112 {
		return nil, fmt.Errorf("expected 112x112 input, got %dx%d", alignedFace.Cols(), alignedFace.Rows())
	}

	// Preprocess
	inputData := e.preprocess(alignedFace)

	// Create input tensor
	inputTensor, err := ort.NewTensor(
		ort.NewShape(1, 3, 112, 112),
		inputData,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Create output tensor
	outputTensor, err := inference.CreateEmptyTensor[float32]([]int64{1, 512})
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Run inference
	err = e.session.Run([]ort.Value{inputTensor}, []ort.Value{outputTensor})
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// Extract and normalize embedding
	outputData := outputTensor.GetData()
	embedding := e.normalizeEmbedding(outputData)

	return embedding, nil
}

// preprocess converts aligned face to model input format
func (e *ArcFaceEncoder) preprocess(img gocv.Mat) []float32 {
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

	// Extract data
	blobData := blob.ToBytes()
	return bytesToFloat32Slice(blobData)
}

// normalizeEmbedding L2-normalizes the embedding
func (e *ArcFaceEncoder) normalizeEmbedding(data []float32) *Embedding {
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
func (e *ArcFaceEncoder) Close() error {
	return e.session.Destroy()
}

// CosineSimilarity computes cosine similarity between two embeddings
func CosineSimilarity(a, b *Embedding) float32 {
	var dot float32
	for i := 0; i < 512; i++ {
		dot += a[i] * b[i]
	}
	// Since embeddings are L2-normalized, dot product = cosine similarity
	return dot
}

func bytesToFloat32Slice(data []byte) []float32 {
	result := make([]float32, len(data)/4)
	for i := range result {
		bits := uint32(data[i*4]) | uint32(data[i*4+1])<<8 | uint32(data[i*4+2])<<16 | uint32(data[i*4+3])<<24
		result[i] = math.Float32frombits(bits)
	}
	return result
}
