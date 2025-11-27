package enhancer

import (
	"fmt"
	"image"
	"math"

	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"

	"github.com/dudu/metalface/internal/inference"
)

const (
	gfpganInputSize = 512
)

// GFPGAN performs face enhancement/restoration
type GFPGAN struct {
	session *inference.Session
}

// NewGFPGAN creates a new GFPGAN face enhancer
func NewGFPGAN(modelPath string) (*GFPGAN, error) {
	inputNames := []string{"input"}
	outputNames := []string{"output"}

	session, err := inference.NewSession(modelPath, inputNames, outputNames)
	if err != nil {
		return nil, fmt.Errorf("failed to create GFPGAN session: %w", err)
	}

	return &GFPGAN{
		session: session,
	}, nil
}

// Enhance enhances a face image (expects aligned 128x128 or similar face crop)
// Returns enhanced face at 512x512 resolution
func (g *GFPGAN) Enhance(face gocv.Mat) (gocv.Mat, error) {
	// Resize to 512x512 if needed
	resized := gocv.NewMat()
	if face.Rows() != gfpganInputSize || face.Cols() != gfpganInputSize {
		gocv.Resize(face, &resized, image.Pt(gfpganInputSize, gfpganInputSize), 0, 0, gocv.InterpolationLinear)
	} else {
		face.CopyTo(&resized)
	}
	defer resized.Close()

	// Convert BGR to RGB
	rgb := gocv.NewMat()
	gocv.CvtColor(resized, &rgb, gocv.ColorBGRToRGB)
	defer rgb.Close()

	// Convert to float32 and normalize to [0, 1]
	floatMat := gocv.NewMat()
	rgb.ConvertTo(&floatMat, gocv.MatTypeCV32FC3)
	defer floatMat.Close()

	// Normalize: divide by 255 to get [0, 1]
	floatMat.DivideFloat(255.0)

	// Create NCHW blob (1, 3, 512, 512)
	blob := gocv.BlobFromImage(floatMat, 1.0, image.Pt(gfpganInputSize, gfpganInputSize),
		gocv.NewScalar(0, 0, 0, 0), false, false)
	defer blob.Close()

	// Get float data
	blobData := blob.ToBytes()
	floatData := bytesToFloat32(blobData)

	// Create input tensor
	inputTensor, err := ort.NewTensor(
		ort.NewShape(1, 3, gfpganInputSize, gfpganInputSize),
		floatData,
	)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Create output tensor (1, 3, 512, 512)
	outputTensor, err := inference.CreateEmptyTensor[float32]([]int64{1, 3, gfpganInputSize, gfpganInputSize})
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Run inference
	err = g.session.Run([]ort.Value{inputTensor}, []ort.Value{outputTensor})
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("GFPGAN inference failed: %w", err)
	}

	// Get output data
	output := outputTensor.GetData()

	// Convert NCHW output to HWC image
	result := g.postprocess(output)

	return result, nil
}

// postprocess converts model output to BGR image
func (g *GFPGAN) postprocess(output []float32) gocv.Mat {
	// Output is in NCHW format, values in [-1, 1] or [0, 1]
	// Need to convert to HWC BGR uint8

	size := gfpganInputSize * gfpganInputSize
	pixels := make([]byte, size*3)

	for y := 0; y < gfpganInputSize; y++ {
		for x := 0; x < gfpganInputSize; x++ {
			idx := y*gfpganInputSize + x

			// Get RGB values (NCHW: [0,c,h,w])
			r := output[0*size+idx]
			g := output[1*size+idx]
			b := output[2*size+idx]

			// Clamp to [0, 1] and convert to uint8
			// Output may be in [-1, 1] or [0, 1] depending on model
			// Try [0, 1] first
			r = clamp(r, 0, 1)
			g = clamp(g, 0, 1)
			b = clamp(b, 0, 1)

			pixIdx := (y*gfpganInputSize + x) * 3
			// BGR order for OpenCV
			pixels[pixIdx+0] = uint8(b * 255)
			pixels[pixIdx+1] = uint8(g * 255)
			pixels[pixIdx+2] = uint8(r * 255)
		}
	}

	// Create Mat from bytes
	result, _ := gocv.NewMatFromBytes(gfpganInputSize, gfpganInputSize, gocv.MatTypeCV8UC3, pixels)
	return result
}

// Close releases resources
func (g *GFPGAN) Close() error {
	return g.session.Destroy()
}

func clamp(v, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

// bytesToFloat32 converts byte slice to float32 slice
func bytesToFloat32(b []byte) []float32 {
	floats := make([]float32, len(b)/4)
	for i := 0; i < len(floats); i++ {
		bits := uint32(b[i*4]) | uint32(b[i*4+1])<<8 | uint32(b[i*4+2])<<16 | uint32(b[i*4+3])<<24
		floats[i] = math.Float32frombits(bits)
	}
	return floats
}
