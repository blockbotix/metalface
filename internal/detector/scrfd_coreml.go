package detector

import (
	"fmt"
	"image"

	"gocv.io/x/gocv"

	"github.com/dudu/metalface/internal/coreml"
)

// SCRFDCoreML implements the SCRFD face detector using CoreML
type SCRFDCoreML struct {
	session        *coreml.Session
	inputSize      int
	confThreshold  float32
	nmsThreshold   float32
	featureStrides []int
	numAnchors     int
	outputNames    []string
}

// NewSCRFDCoreML creates a new SCRFD detector using CoreML
func NewSCRFDCoreML(modelPath string, inputSize int, confThreshold, nmsThreshold float32) (*SCRFDCoreML, error) {
	inputNames := []string{"input_1"}
	// Output names in the order we want them concatenated: all scores, then all bbox, then all kps
	// Based on actual tensor sizes:
	// - var_717=12800 (stride8 score), var_731=51200 (stride8 bbox), var_745=128000 (stride8 kps)
	// - var_830=3200 (stride16 score), var_844=12800 (stride16 bbox), var_858=32000 (stride16 kps)
	// - var_943=800 (stride32 score), var_957=3200 (stride32 bbox), var_971=8000 (stride32 kps)
	outputNames := []string{
		"var_717", "var_830", "var_943", // scores for stride 8, 16, 32
		"var_731", "var_844", "var_957", // bbox for stride 8, 16, 32
		"var_745", "var_858", "var_971", // kps for stride 8, 16, 32
	}

	session, err := coreml.NewSession(modelPath, inputNames, outputNames)
	if err != nil {
		return nil, fmt.Errorf("failed to create SCRFD CoreML session: %w", err)
	}

	return &SCRFDCoreML{
		session:        session,
		inputSize:      inputSize,
		confThreshold:  confThreshold,
		nmsThreshold:   nmsThreshold,
		featureStrides: []int{8, 16, 32},
		numAnchors:     2,
		outputNames:    outputNames,
	}, nil
}

// Detect finds faces in an image
func (s *SCRFDCoreML) Detect(img gocv.Mat) ([]Face, error) {
	origHeight := img.Rows()
	origWidth := img.Cols()

	// Preprocess
	inputData, scale := s.preprocess(img)

	// Calculate output sizes for each stride
	fmHeight := []int{s.inputSize / 8, s.inputSize / 16, s.inputSize / 32}
	fmWidth := []int{s.inputSize / 8, s.inputSize / 16, s.inputSize / 32}

	totalOutputSize := 0
	for i := 0; i < 3; i++ {
		numAnchors := fmHeight[i] * fmWidth[i] * s.numAnchors
		totalOutputSize += numAnchors * (1 + 4 + 10) // score + bbox + kps
	}

	// Calculate output size for all 9 outputs concatenated
	outputSize := 0
	for i := 0; i < 3; i++ {
		numAnchors := fmHeight[i] * fmWidth[i] * s.numAnchors
		outputSize += numAnchors * 1  // scores
		outputSize += numAnchors * 4  // bbox
		outputSize += numAnchors * 10 // kps
	}

	// Run inference with multi-output to get all outputs in the correct order
	output, err := s.session.RunMultiOutput(inputData, []int64{1, 3, int64(s.inputSize), int64(s.inputSize)}, s.outputNames, outputSize)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// Parse outputs and decode faces
	faces := s.postprocess(output, fmHeight, fmWidth, scale, origWidth, origHeight)

	// Apply NMS
	faces = nms(faces, s.nmsThreshold)

	return faces, nil
}

// preprocess resizes and normalizes the image
func (s *SCRFDCoreML) preprocess(img gocv.Mat) ([]float32, float32) {
	height := img.Rows()
	width := img.Cols()

	scale := float32(s.inputSize) / float32(max(height, width))

	newWidth := int(float32(width) * scale)
	newHeight := int(float32(height) * scale)

	// Resize
	resized := gocv.NewMat()
	gocv.Resize(img, &resized, image.Pt(newWidth, newHeight), 0, 0, gocv.InterpolationLinear)
	defer resized.Close()

	// Create padded image (letterbox)
	padded := gocv.NewMatWithSize(s.inputSize, s.inputSize, gocv.MatTypeCV8UC3)
	padded.SetTo(gocv.NewScalar(0, 0, 0, 0))
	defer padded.Close()

	// Copy resized to top-left
	roi := padded.Region(image.Rect(0, 0, newWidth, newHeight))
	resized.CopyTo(&roi)
	roi.Close()

	// Convert BGR to RGB
	rgb := gocv.NewMat()
	gocv.CvtColor(padded, &rgb, gocv.ColorBGRToRGB)
	defer rgb.Close()

	// Convert to float and normalize: (x - 127.5) / 128.0
	blob := gocv.NewMat()
	rgb.ConvertTo(&blob, gocv.MatTypeCV32FC3)
	defer blob.Close()

	gocv.AddWeighted(blob, 1.0/128.0, blob, 0, -127.5/128.0, &blob)

	// Convert HWC to CHW
	blobNCHW := gocv.BlobFromImage(blob, 1.0, image.Pt(s.inputSize, s.inputSize),
		gocv.NewScalar(0, 0, 0, 0), false, false)
	defer blobNCHW.Close()

	blobData := blobNCHW.ToBytes()
	return bytesToFloat32(blobData), scale
}

// postprocess decodes model outputs to faces
func (s *SCRFDCoreML) postprocess(output []float32, fmHeight, fmWidth []int, scale float32, origWidth, origHeight int) []Face {
	var faces []Face

	// Calculate offsets for each output in the concatenated array
	// CoreML returns outputs in order: scores[3], bbox[3], kps[3]
	offsets := make([]int, 9)
	offset := 0
	for i := 0; i < 3; i++ {
		numAnchors := fmHeight[i] * fmWidth[i] * s.numAnchors
		offsets[i] = offset
		offset += numAnchors * 1 // scores
	}
	for i := 0; i < 3; i++ {
		numAnchors := fmHeight[i] * fmWidth[i] * s.numAnchors
		offsets[i+3] = offset
		offset += numAnchors * 4 // bbox
	}
	for i := 0; i < 3; i++ {
		numAnchors := fmHeight[i] * fmWidth[i] * s.numAnchors
		offsets[i+6] = offset
		offset += numAnchors * 10 // kps
	}

	for level := 0; level < 3; level++ {
		stride := s.featureStrides[level]
		fmH := fmHeight[level]
		fmW := fmWidth[level]

		scoreOffset := offsets[level]
		bboxOffset := offsets[level+3]
		kpsOffset := offsets[level+6]

		anchorIdx := 0
		for y := 0; y < fmH; y++ {
			for x := 0; x < fmW; x++ {
				for a := 0; a < s.numAnchors; a++ {
					score := output[scoreOffset+anchorIdx]

					if score > s.confThreshold {
						cx := (float32(x) + 0.5) * float32(stride)
						cy := (float32(y) + 0.5) * float32(stride)

						bboxIdx := bboxOffset + anchorIdx*4
						x1 := (cx - output[bboxIdx]*float32(stride)) / scale
						y1 := (cy - output[bboxIdx+1]*float32(stride)) / scale
						x2 := (cx + output[bboxIdx+2]*float32(stride)) / scale
						y2 := (cy + output[bboxIdx+3]*float32(stride)) / scale

						x1 = clamp(x1, 0, float32(origWidth))
						y1 = clamp(y1, 0, float32(origHeight))
						x2 = clamp(x2, 0, float32(origWidth))
						y2 = clamp(y2, 0, float32(origHeight))

						kpsIdx := kpsOffset + anchorIdx*10
						landmarks := Landmarks{
							LeftEye:    Point{(cx + output[kpsIdx]*float32(stride)) / scale, (cy + output[kpsIdx+1]*float32(stride)) / scale},
							RightEye:   Point{(cx + output[kpsIdx+2]*float32(stride)) / scale, (cy + output[kpsIdx+3]*float32(stride)) / scale},
							Nose:       Point{(cx + output[kpsIdx+4]*float32(stride)) / scale, (cy + output[kpsIdx+5]*float32(stride)) / scale},
							LeftMouth:  Point{(cx + output[kpsIdx+6]*float32(stride)) / scale, (cy + output[kpsIdx+7]*float32(stride)) / scale},
							RightMouth: Point{(cx + output[kpsIdx+8]*float32(stride)) / scale, (cy + output[kpsIdx+9]*float32(stride)) / scale},
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
func (s *SCRFDCoreML) Close() error {
	return s.session.Destroy()
}
