package detector

import (
	"fmt"
	"image"
	"math"

	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"

	"github.com/dudu/metalface/internal/inference"
)

// Landmark106 detects 106 facial landmarks using insightface's 2d106det model
type Landmark106 struct {
	session   *inference.Session
	inputSize int
	inputMean float32
	inputStd  float32
}

// NewLandmark106 creates a new 106-point landmark detector
func NewLandmark106(modelPath string) (*Landmark106, error) {
	inputNames := []string{"data"}
	outputNames := []string{"fc1"}

	session, err := inference.NewSession(modelPath, inputNames, outputNames)
	if err != nil {
		return nil, fmt.Errorf("failed to create landmark session: %w", err)
	}

	return &Landmark106{
		session:   session,
		inputSize: 192,
		inputMean: 127.5,
		inputStd:  128.0,
	}, nil
}

// Detect extracts 106 landmarks for a detected face
func (l *Landmark106) Detect(img gocv.Mat, face *Face) error {
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

	// Warp image to get aligned face
	aligned := gocv.NewMat()
	defer aligned.Close()
	gocv.WarpAffine(img, &aligned, M, image.Pt(l.inputSize, l.inputSize))
	M.Close()

	// Preprocess: convert to RGB, normalize
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

	// Create input tensor
	blobData := blob.ToBytes()
	floatData := bytesToFloat32(blobData)

	inputTensor, err := ort.NewTensor(
		ort.NewShape(1, 3, int64(l.inputSize), int64(l.inputSize)),
		floatData,
	)
	if err != nil {
		return fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Create output tensor (1, 212) = 106 landmarks * 2 coords
	outputTensor, err := inference.CreateEmptyTensor[float32]([]int64{1, 212})
	if err != nil {
		return fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Run inference
	err = l.session.Run([]ort.Value{inputTensor}, []ort.Value{outputTensor})
	if err != nil {
		return fmt.Errorf("landmark inference failed: %w", err)
	}

	// Get output and reshape to landmarks
	output := outputTensor.GetData()

	// Transform landmarks back to original image coordinates
	landmarks := l.postprocess(output, centerX, centerY, scale)
	face.Landmarks106 = &landmarks

	return nil
}

// getTransformMatrix creates affine transform for face crop
func (l *Landmark106) getTransformMatrix(centerX, centerY, scale float32) gocv.Mat {
	// Create rotation matrix (no rotation, just scale and translate)
	M := gocv.NewMatWithSize(2, 3, gocv.MatTypeCV64F)

	// Scale and center
	M.SetDoubleAt(0, 0, float64(scale))
	M.SetDoubleAt(0, 1, 0)
	M.SetDoubleAt(0, 2, float64(l.inputSize)/2 - float64(centerX*scale))
	M.SetDoubleAt(1, 0, 0)
	M.SetDoubleAt(1, 1, float64(scale))
	M.SetDoubleAt(1, 2, float64(l.inputSize)/2 - float64(centerY*scale))

	return M
}

// postprocess transforms landmarks from model output to original image coordinates
func (l *Landmark106) postprocess(output []float32, centerX, centerY, scale float32) Landmarks106 {
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
func (l *Landmark106) Close() error {
	return l.session.Destroy()
}

// GetFaceOutlineIndices returns the landmark indices for face outline (chin to forehead)
// Based on insightface's 106-point layout
func GetFaceOutlineIndices() []int {
	// Face contour: indices 0-32 (chin to ears)
	return []int{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	}
}

// GetLowerLipIndices returns the landmark indices for lower lip/mouth area
// Based on insightface's 106-point layout used in Deep-Live-Cam
func GetLowerLipIndices() []int {
	return []int{
		65, 66, 62, 70, 69, 18, 19, 20, 21, 22, 23, 24, 0, 8, 7, 6, 5, 4, 3, 2, 65,
	}
}

// GetLeftEyeIndices returns indices for left eye landmarks
func GetLeftEyeIndices() []int {
	return []int{87, 88, 89, 90, 91, 92, 93, 94, 95, 96}
}

// GetRightEyeIndices returns indices for right eye landmarks
func GetRightEyeIndices() []int {
	return []int{33, 34, 35, 36, 37, 38, 39, 40, 41, 42}
}

// Helper to get points from Landmarks106 by indices
func (l *Landmarks106) GetPoints(indices []int) []Point {
	points := make([]Point, len(indices))
	for i, idx := range indices {
		if idx < 106 {
			points[i] = l[idx]
		}
	}
	return points
}

// ConvexHull computes convex hull of points (simplified version)
func ConvexHull(points []Point) []Point {
	if len(points) < 3 {
		return points
	}

	// Find leftmost point
	minIdx := 0
	for i := 1; i < len(points); i++ {
		if points[i].X < points[minIdx].X {
			minIdx = i
		}
	}

	hull := []Point{}
	p := minIdx
	for {
		hull = append(hull, points[p])
		q := (p + 1) % len(points)

		for i := 0; i < len(points); i++ {
			if orientation(points[p], points[i], points[q]) == 2 {
				q = i
			}
		}

		p = q
		if p == minIdx {
			break
		}
	}

	return hull
}

// orientation returns:
// 0 -> Collinear, 1 -> Clockwise, 2 -> Counterclockwise
func orientation(p, q, r Point) int {
	val := (q.Y-p.Y)*(r.X-q.X) - (q.X-p.X)*(r.Y-q.Y)

	if math.Abs(float64(val)) < 1e-9 {
		return 0
	}
	if val > 0 {
		return 1
	}
	return 2
}
