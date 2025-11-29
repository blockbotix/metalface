package swapper

import (
	"fmt"
	"image"

	"gocv.io/x/gocv"

	"github.com/dudu/metalface/internal/coreml"
)

// InswapperCoreML performs face swapping using the inswapper model with CoreML
type InswapperCoreML struct {
	session *coreml.Session
}

// NewInswapperCoreML creates a new face swapper using CoreML
func NewInswapperCoreML(modelPath string) (*InswapperCoreML, error) {
	inputNames := []string{"target", "source"}
	outputNames := []string{"var_1144"}

	session, err := coreml.NewSession(modelPath, inputNames, outputNames)
	if err != nil {
		return nil, fmt.Errorf("failed to create Inswapper CoreML session: %w", err)
	}

	return &InswapperCoreML{
		session: session,
	}, nil
}

// Swap generates a swapped face from target face and source embedding
func (s *InswapperCoreML) Swap(targetFace gocv.Mat, sourceEmbedding *Embedding) (gocv.Mat, error) {
	if targetFace.Rows() != 128 || targetFace.Cols() != 128 {
		return gocv.NewMat(), fmt.Errorf("expected 128x128 target, got %dx%d", targetFace.Cols(), targetFace.Rows())
	}

	// Preprocess target face
	targetData := s.preprocessTarget(targetFace)

	// Prepare inputs for multi-input inference
	inputs := [][]float32{targetData, sourceEmbedding[:]}
	shapes := [][]int64{{1, 3, 128, 128}, {1, 512}}

	// Run multi-input inference
	output, err := s.session.RunMulti(inputs, shapes, 1*3*128*128)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("inference failed: %w", err)
	}

	// Postprocess output
	result := s.postprocess(output)

	return result, nil
}

// preprocessTarget converts target face to model input format
// Matches insightface preprocessing:
// blob = cv2.dnn.blobFromImage(aimg, 1.0/255, input_size, (0,0,0), swapRB=True)
func (s *InswapperCoreML) preprocessTarget(img gocv.Mat) []float32 {
	// BlobFromImage with:
	// - scalefactor = 1/255 for [0, 1] normalization
	// - swapRB = true to convert BGR (OpenCV) to RGB (model expects RGB)
	// - mean = (0, 0, 0)
	// - crop = false
	blob := gocv.BlobFromImage(img, 1.0/255.0, image.Pt(128, 128),
		gocv.NewScalar(0, 0, 0, 0), true, false)
	defer blob.Close()

	blobData := blob.ToBytes()
	return bytesToFloat32Slice(blobData)
}

// postprocess converts model output to BGR image
func (s *InswapperCoreML) postprocess(data []float32) gocv.Mat {
	result := gocv.NewMatWithSize(128, 128, gocv.MatTypeCV8UC3)

	for y := 0; y < 128; y++ {
		for x := 0; x < 128; x++ {
			rIdx := 0*128*128 + y*128 + x
			gIdx := 1*128*128 + y*128 + x
			bIdx := 2*128*128 + y*128 + x

			r := clampByte(data[rIdx] * 255.0)
			g := clampByte(data[gIdx] * 255.0)
			b := clampByte(data[bIdx] * 255.0)

			result.SetUCharAt(y, x*3+0, b)
			result.SetUCharAt(y, x*3+1, g)
			result.SetUCharAt(y, x*3+2, r)
		}
	}

	return result
}

// Close releases swapper resources
func (s *InswapperCoreML) Close() error {
	return s.session.Destroy()
}
