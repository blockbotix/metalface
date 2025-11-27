package detector

import (
	"fmt"
	"image"

	"gocv.io/x/gocv"

	"github.com/dudu/metalface/internal/coreml"
)

// Landmark106CoreML detects 106 facial landmarks using CoreML
type Landmark106CoreML struct {
	session   *coreml.Session
	inputSize int
	inputMean float32
	inputStd  float32
}

// NewLandmark106CoreML creates a new 106-point landmark detector using CoreML
func NewLandmark106CoreML(modelPath string) (*Landmark106CoreML, error) {
	inputNames := []string{"data"}
	outputNames := []string{"fc1"}

	session, err := coreml.NewSession(modelPath, inputNames, outputNames)
	if err != nil {
		return nil, fmt.Errorf("failed to create landmark CoreML session: %w", err)
	}

	return &Landmark106CoreML{
		session:   session,
		inputSize: 192,
		inputMean: 127.5,
		inputStd:  128.0,
	}, nil
}

// Detect extracts 106 landmarks for a detected face
func (l *Landmark106CoreML) Detect(img gocv.Mat, face *Face) error {
	bbox := face.BoundingBox

	// Calculate crop parameters (1.5x expansion like insightface)
	w := bbox.Width()
	h := bbox.Height()
	centerX := (bbox.X1 + bbox.X2) / 2
	centerY := (bbox.Y1 + bbox.Y2) / 2
	maxDim := w
	if h > w {
		maxDim = h
	}
	scale := float32(l.inputSize) / (maxDim * 1.5)

	// Create affine transform matrix for cropping
	M := l.getTransformMatrix(centerX, centerY, scale)
	defer M.Close()

	// Warp image to get aligned face
	aligned := gocv.NewMat()
	defer aligned.Close()
	gocv.WarpAffine(img, &aligned, M, image.Pt(l.inputSize, l.inputSize))

	// Preprocess
	inputData := l.preprocess(aligned)

	// Run inference (output is 212 values = 106 landmarks * 2 coords)
	output, err := l.session.Run(inputData, []int64{1, 3, int64(l.inputSize), int64(l.inputSize)}, 212)
	if err != nil {
		return fmt.Errorf("landmark inference failed: %w", err)
	}

	// Transform landmarks back to original image coordinates
	landmarks := l.postprocess(output, centerX, centerY, scale)
	face.Landmarks106 = &landmarks

	return nil
}

// preprocess converts image to model input format
func (l *Landmark106CoreML) preprocess(aligned gocv.Mat) []float32 {
	// Convert to RGB
	rgb := gocv.NewMat()
	gocv.CvtColor(aligned, &rgb, gocv.ColorBGRToRGB)
	defer rgb.Close()

	// Convert to float32 and normalize
	floatMat := gocv.NewMat()
	rgb.ConvertTo(&floatMat, gocv.MatTypeCV32FC3)
	defer floatMat.Close()

	// Apply normalization: (x - mean) / std
	gocv.AddWeighted(floatMat, 1.0/float64(l.inputStd), floatMat, 0, -float64(l.inputMean)/float64(l.inputStd), &floatMat)

	// Convert HWC to NCHW blob
	blob := gocv.BlobFromImage(floatMat, 1.0, image.Pt(l.inputSize, l.inputSize),
		gocv.NewScalar(0, 0, 0, 0), false, false)
	defer blob.Close()

	blobData := blob.ToBytes()
	return bytesToFloat32(blobData)
}

// getTransformMatrix creates affine transform for face crop
func (l *Landmark106CoreML) getTransformMatrix(centerX, centerY, scale float32) gocv.Mat {
	M := gocv.NewMatWithSize(2, 3, gocv.MatTypeCV64F)

	M.SetDoubleAt(0, 0, float64(scale))
	M.SetDoubleAt(0, 1, 0)
	M.SetDoubleAt(0, 2, float64(l.inputSize)/2-float64(centerX*scale))
	M.SetDoubleAt(1, 0, 0)
	M.SetDoubleAt(1, 1, float64(scale))
	M.SetDoubleAt(1, 2, float64(l.inputSize)/2-float64(centerY*scale))

	return M
}

// postprocess transforms landmarks from model output to original image coordinates
func (l *Landmark106CoreML) postprocess(output []float32, centerX, centerY, scale float32) Landmarks106 {
	var landmarks Landmarks106

	halfSize := float32(l.inputSize) / 2

	for i := 0; i < 106; i++ {
		// Model output is in range [-1, 1], transform to [0, inputSize]
		x := (output[i*2] + 1) * halfSize
		y := (output[i*2+1] + 1) * halfSize

		// Transform back to original image coordinates
		landmarks[i] = Point{
			X: (x - halfSize) / scale + centerX,
			Y: (y - halfSize) / scale + centerY,
		}
	}

	return landmarks
}

// Close releases detector resources
func (l *Landmark106CoreML) Close() error {
	return l.session.Destroy()
}
