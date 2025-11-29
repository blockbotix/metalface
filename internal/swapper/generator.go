package swapper

import (
	"fmt"
	"image"

	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"

	"github.com/dudu/metalface/internal/inference"
)

// Inswapper performs face swapping using the inswapper model
type Inswapper struct {
	session *inference.Session
}

// NewInswapper creates a new face swapper
func NewInswapper(modelPath string) (*Inswapper, error) {
	// Inswapper has 2 inputs: target face and source embedding
	inputNames := []string{"target", "source"}
	outputNames := []string{"output"}

	session, err := inference.NewSession(modelPath, inputNames, outputNames)
	if err != nil {
		return nil, fmt.Errorf("failed to create Inswapper session: %w", err)
	}

	return &Inswapper{
		session: session,
	}, nil
}

// Swap generates a swapped face from target face and source embedding
// targetFace should be aligned to 128x128
// Returns the swapped face as 128x128 BGR image
func (s *Inswapper) Swap(targetFace gocv.Mat, sourceEmbedding *Embedding) (gocv.Mat, error) {
	// Verify input size
	if targetFace.Rows() != 128 || targetFace.Cols() != 128 {
		return gocv.NewMat(), fmt.Errorf("expected 128x128 target, got %dx%d", targetFace.Cols(), targetFace.Rows())
	}

	// Preprocess target face
	targetData := s.preprocessTarget(targetFace)

	// Create input tensors
	targetTensor, err := ort.NewTensor(
		ort.NewShape(1, 3, 128, 128),
		targetData,
	)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to create target tensor: %w", err)
	}
	defer targetTensor.Destroy()

	sourceTensor, err := ort.NewTensor(
		ort.NewShape(1, 512),
		sourceEmbedding[:],
	)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to create source tensor: %w", err)
	}
	defer sourceTensor.Destroy()

	// Create output tensor
	outputTensor, err := inference.CreateEmptyTensor[float32]([]int64{1, 3, 128, 128})
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
// Matches insightface preprocessing:
// blob = cv2.dnn.blobFromImage(aimg, 1.0/255, input_size, (0,0,0), swapRB=True)
func (s *Inswapper) preprocessTarget(img gocv.Mat) []float32 {
	// BlobFromImage with:
	// - scalefactor = 1/255 for [0, 1] normalization
	// - swapRB = true to convert BGR (OpenCV) to RGB (model expects RGB)
	// - mean = (0, 0, 0)
	// - crop = false
	blob := gocv.BlobFromImage(img, 1.0/255.0, image.Pt(128, 128),
		gocv.NewScalar(0, 0, 0, 0), true, false)
	defer blob.Close()

	// Extract data
	blobData := blob.ToBytes()
	return bytesToFloat32Slice(blobData)
}

// postprocess converts model output to BGR image
func (s *Inswapper) postprocess(data []float32) gocv.Mat {
	// Output is NCHW [1, 3, 128, 128] with values in [0, 1]
	// Convert to HWC BGR image

	result := gocv.NewMatWithSize(128, 128, gocv.MatTypeCV8UC3)

	for y := 0; y < 128; y++ {
		for x := 0; x < 128; x++ {
			// Get RGB values from CHW layout
			rIdx := 0*128*128 + y*128 + x
			gIdx := 1*128*128 + y*128 + x
			bIdx := 2*128*128 + y*128 + x

			r := clampByte(data[rIdx] * 255.0)
			g := clampByte(data[gIdx] * 255.0)
			b := clampByte(data[bIdx] * 255.0)

			// Set BGR pixel
			result.SetUCharAt(y, x*3+0, b)
			result.SetUCharAt(y, x*3+1, g)
			result.SetUCharAt(y, x*3+2, r)
		}
	}

	return result
}

// Close releases swapper resources
func (s *Inswapper) Close() error {
	return s.session.Destroy()
}

func clampByte(v float32) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}
