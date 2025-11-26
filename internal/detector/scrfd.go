package detector

import (
	"fmt"
	"image"
	"math"

	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"

	"github.com/dudu/metalface/internal/inference"
)

// SCRFD implements the SCRFD face detector
type SCRFD struct {
	session        *inference.Session
	inputSize      int
	confThreshold  float32
	nmsThreshold   float32
	featureStrides []int
	numAnchors     int
}

// NewSCRFD creates a new SCRFD detector
func NewSCRFD(modelPath string, inputSize int, confThreshold, nmsThreshold float32) (*SCRFD, error) {
	// SCRFD has 1 input and 9 outputs (3 levels × 3 outputs each: score, bbox, kps)
	inputNames := []string{"input.1"}
	outputNames := []string{
		"score_8", "score_16", "score_32",
		"bbox_8", "bbox_16", "bbox_32",
		"kps_8", "kps_16", "kps_32",
	}

	session, err := inference.NewSession(modelPath, inputNames, outputNames)
	if err != nil {
		return nil, fmt.Errorf("failed to create SCRFD session: %w", err)
	}

	return &SCRFD{
		session:        session,
		inputSize:      inputSize,
		confThreshold:  confThreshold,
		nmsThreshold:   nmsThreshold,
		featureStrides: []int{8, 16, 32},
		numAnchors:     2, // anchors per position
	}, nil
}

// Detect finds faces in an image
func (s *SCRFD) Detect(img gocv.Mat) ([]Face, error) {
	// Get original dimensions
	origHeight := img.Rows()
	origWidth := img.Cols()

	// Preprocess: resize and normalize
	inputBlob, scale := s.preprocess(img)
	defer inputBlob.Close()

	// Create input tensor
	blobData := inputBlob.ToBytes()
	floatData := bytesToFloat32(blobData)

	inputTensor, err := ort.NewTensor(
		ort.NewShape(1, 3, int64(s.inputSize), int64(s.inputSize)),
		floatData,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Create output tensors
	fmHeight := []int{s.inputSize / 8, s.inputSize / 16, s.inputSize / 32}
	fmWidth := []int{s.inputSize / 8, s.inputSize / 16, s.inputSize / 32}

	outputs := make([]ort.Value, 9)
	outputTensors := make([]*ort.Tensor[float32], 9)

	for i := 0; i < 3; i++ {
		numAnchors := fmHeight[i] * fmWidth[i] * s.numAnchors

		// Score output
		scoreTensor, _ := inference.CreateEmptyTensor[float32]([]int64{int64(numAnchors), 1})
		outputs[i] = scoreTensor
		outputTensors[i] = scoreTensor

		// Bbox output
		bboxTensor, _ := inference.CreateEmptyTensor[float32]([]int64{int64(numAnchors), 4})
		outputs[i+3] = bboxTensor
		outputTensors[i+3] = bboxTensor

		// Keypoints output
		kpsTensor, _ := inference.CreateEmptyTensor[float32]([]int64{int64(numAnchors), 10})
		outputs[i+6] = kpsTensor
		outputTensors[i+6] = kpsTensor
	}
	defer func() {
		for _, t := range outputTensors {
			t.Destroy()
		}
	}()

	// Run inference
	err = s.session.Run([]ort.Value{inputTensor}, outputs)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// Decode outputs
	faces := s.postprocess(outputTensors, scale, origWidth, origHeight)

	// Apply NMS
	faces = nms(faces, s.nmsThreshold)

	return faces, nil
}

// preprocess resizes and normalizes the image
func (s *SCRFD) preprocess(img gocv.Mat) (gocv.Mat, float32) {
	// Calculate scale to fit input size while maintaining aspect ratio
	height := img.Rows()
	width := img.Cols()

	scale := float32(s.inputSize) / float32(max(height, width))

	newWidth := int(float32(width) * scale)
	newHeight := int(float32(height) * scale)

	// Resize
	resized := gocv.NewMat()
	gocv.Resize(img, &resized, image.Pt(newWidth, newHeight), 0, 0, gocv.InterpolationLinear)

	// Create padded image (letterbox)
	padded := gocv.NewMatWithSize(s.inputSize, s.inputSize, gocv.MatTypeCV8UC3)
	padded.SetTo(gocv.NewScalar(0, 0, 0, 0))

	// Copy resized to center of padded
	roi := padded.Region(image.Rect(0, 0, newWidth, newHeight))
	resized.CopyTo(&roi)
	roi.Close()
	resized.Close()

	// Convert BGR to RGB
	rgb := gocv.NewMat()
	gocv.CvtColor(padded, &rgb, gocv.ColorBGRToRGB)
	padded.Close()

	// Convert to float and normalize: (x - 127.5) / 128.0
	blob := gocv.NewMat()
	rgb.ConvertTo(&blob, gocv.MatTypeCV32FC3)
	rgb.Close()

	// Normalize
	gocv.AddWeighted(blob, 1.0/128.0, blob, 0, -127.5/128.0, &blob)

	// Convert HWC to CHW (blob format)
	blobNCHW := gocv.BlobFromImage(blob, 1.0, image.Pt(s.inputSize, s.inputSize),
		gocv.NewScalar(0, 0, 0, 0), false, false)
	blob.Close()

	return blobNCHW, scale
}

// postprocess decodes model outputs to faces
func (s *SCRFD) postprocess(outputs []*ort.Tensor[float32], scale float32, origWidth, origHeight int) []Face {
	var faces []Face

	for level := 0; level < 3; level++ {
		stride := s.featureStrides[level]
		fmHeight := s.inputSize / stride
		fmWidth := s.inputSize / stride

		scoreData := outputs[level].GetData()
		bboxData := outputs[level+3].GetData()
		kpsData := outputs[level+6].GetData()

		anchorIdx := 0
		for y := 0; y < fmHeight; y++ {
			for x := 0; x < fmWidth; x++ {
				for a := 0; a < s.numAnchors; a++ {
					score := sigmoid(scoreData[anchorIdx])

					if score > s.confThreshold {
						// Anchor center
						cx := (float32(x) + 0.5) * float32(stride)
						cy := (float32(y) + 0.5) * float32(stride)

						// Decode bbox (distance to edges)
						bboxIdx := anchorIdx * 4
						x1 := (cx - bboxData[bboxIdx]*float32(stride)) / scale
						y1 := (cy - bboxData[bboxIdx+1]*float32(stride)) / scale
						x2 := (cx + bboxData[bboxIdx+2]*float32(stride)) / scale
						y2 := (cy + bboxData[bboxIdx+3]*float32(stride)) / scale

						// Clamp to image bounds
						x1 = clamp(x1, 0, float32(origWidth))
						y1 = clamp(y1, 0, float32(origHeight))
						x2 = clamp(x2, 0, float32(origWidth))
						y2 = clamp(y2, 0, float32(origHeight))

						// Decode keypoints
						kpsIdx := anchorIdx * 10
						landmarks := Landmarks{
							LeftEye:    Point{(cx + kpsData[kpsIdx]*float32(stride)) / scale, (cy + kpsData[kpsIdx+1]*float32(stride)) / scale},
							RightEye:   Point{(cx + kpsData[kpsIdx+2]*float32(stride)) / scale, (cy + kpsData[kpsIdx+3]*float32(stride)) / scale},
							Nose:       Point{(cx + kpsData[kpsIdx+4]*float32(stride)) / scale, (cy + kpsData[kpsIdx+5]*float32(stride)) / scale},
							LeftMouth:  Point{(cx + kpsData[kpsIdx+6]*float32(stride)) / scale, (cy + kpsData[kpsIdx+7]*float32(stride)) / scale},
							RightMouth: Point{(cx + kpsData[kpsIdx+8]*float32(stride)) / scale, (cy + kpsData[kpsIdx+9]*float32(stride)) / scale},
						}

						faces = append(faces, Face{
							BoundingBox: BoundingBox{X1: x1, Y1: y1, X2: x2, Y2: y2},
							Landmarks:   landmarks,
							Score:       score,
						})
					}
					anchorIdx++
				}
			}
		}
	}

	return faces
}

// Close releases detector resources
func (s *SCRFD) Close() error {
	return s.session.Destroy()
}

func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

func clamp(x, min, max float32) float32 {
	if x < min {
		return min
	}
	if x > max {
		return max
	}
	return x
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func bytesToFloat32(data []byte) []float32 {
	result := make([]float32, len(data)/4)
	for i := range result {
		bits := uint32(data[i*4]) | uint32(data[i*4+1])<<8 | uint32(data[i*4+2])<<16 | uint32(data[i*4+3])<<24
		result[i] = math.Float32frombits(bits)
	}
	return result
}
