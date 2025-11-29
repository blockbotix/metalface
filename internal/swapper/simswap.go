package swapper

import (
	"fmt"
	"image"
	"math"

	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"

	"github.com/dudu/metalface/internal/inference"
)

var simswapDebugOnce bool

// l2NormalizeEmbedding normalizes embedding to unit length
func l2NormalizeEmbedding(emb *Embedding) []float32 {
	result := make([]float32, 512)
	var sum float64
	for _, v := range emb {
		sum += float64(v * v)
	}
	norm := float32(math.Sqrt(sum))
	if norm > 0 {
		for i, v := range emb {
			result[i] = v / norm
		}
	} else {
		copy(result, emb[:])
	}
	return result
}

// SimSwap512 performs face swapping using the SimSwap 512x512 model
type SimSwap512 struct {
	session *inference.Session
}

// NewSimSwap512 creates a new SimSwap 512x512 face swapper
func NewSimSwap512(modelPath string) (*SimSwap512, error) {
	// SimSwap has 2 inputs: target face and source embedding
	inputNames := []string{"target", "source"}
	outputNames := []string{"output"}

	session, err := inference.NewSession(modelPath, inputNames, outputNames)
	if err != nil {
		return nil, fmt.Errorf("failed to create SimSwap512 session: %w", err)
	}

	return &SimSwap512{
		session: session,
	}, nil
}

// Swap generates a swapped face from target face and source embedding
// targetFace should be aligned to 512x512
// Returns the swapped face as 512x512 BGR image
func (s *SimSwap512) Swap(targetFace gocv.Mat, sourceEmbedding *Embedding) (gocv.Mat, error) {
	// Verify input size
	if targetFace.Rows() != 512 || targetFace.Cols() != 512 {
		return gocv.NewMat(), fmt.Errorf("expected 512x512 target, got %dx%d", targetFace.Cols(), targetFace.Rows())
	}

	// Preprocess target face
	targetData := s.preprocessTarget(targetFace)

	// L2 normalize the embedding for SimSwap
	normalizedEmb := l2NormalizeEmbedding(sourceEmbedding)

	// Create input tensors
	targetTensor, err := ort.NewTensor(
		ort.NewShape(1, 3, 512, 512),
		targetData,
	)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to create target tensor: %w", err)
	}
	defer targetTensor.Destroy()

	sourceTensor, err := ort.NewTensor(
		ort.NewShape(1, 512),
		normalizedEmb[:],
	)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to create source tensor: %w", err)
	}
	defer sourceTensor.Destroy()

	// Create output tensor
	outputTensor, err := inference.CreateEmptyTensor[float32]([]int64{1, 3, 512, 512})
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Run inference
	err = s.session.Run(
		[]ort.Value{targetTensor, sourceTensor},
		[]ort.Value{outputTensor},
	)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("inference failed: %w", err)
	}

	// Postprocess output
	outputData := outputTensor.GetData()
	result := s.postprocess(outputData)

	return result, nil
}

// preprocessTarget converts target face to model input format
// SimSwap 512 unofficial expects RGB input normalized to [0, 1] range
func (s *SimSwap512) preprocessTarget(img gocv.Mat) []float32 {
	// BlobFromImage with:
	// - scalefactor = 1/255.0 for [0, 1] normalization
	// - swapRB = true to convert BGR (OpenCV) to RGB (model expects RGB)
	// - crop = false
	blob := gocv.BlobFromImage(img, 1.0/255.0, image.Pt(512, 512),
		gocv.NewScalar(0, 0, 0, 0), true, false)
	defer blob.Close()

	// Extract data
	blobData := blob.ToBytes()
	return bytesToFloat32Slice(blobData)
}

// postprocess converts model output to BGR image
func (s *SimSwap512) postprocess(data []float32) gocv.Mat {
	// Debug: check output range (only once)
	if !simswapDebugOnce {
		simswapDebugOnce = true
		var minVal, maxVal float32 = data[0], data[0]
		for _, v := range data {
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}
		fmt.Printf("\n[SimSwap] Output range: min=%.3f, max=%.3f\n", minVal, maxVal)
	}

	result := gocv.NewMatWithSize(512, 512, gocv.MatTypeCV8UC3)

	for y := 0; y < 512; y++ {
		for x := 0; x < 512; x++ {
			// Get RGB values from CHW layout (model outputs RGB)
			rIdx := 0*512*512 + y*512 + x
			gIdx := 1*512*512 + y*512 + x
			bIdx := 2*512*512 + y*512 + x

			// Output is in [0, 1] range, convert to [0, 255]
			r := clampByte(data[rIdx] * 255.0)
			g := clampByte(data[gIdx] * 255.0)
			b := clampByte(data[bIdx] * 255.0)

			// Write as BGR for OpenCV (swap R and B)
			result.SetUCharAt(y, x*3+0, b) // B
			result.SetUCharAt(y, x*3+1, g) // G
			result.SetUCharAt(y, x*3+2, r) // R
		}
	}

	return result
}

// Close releases swapper resources
func (s *SimSwap512) Close() error {
	return s.session.Destroy()
}
