package enhancer

import (
	"fmt"
	"image"
	"math"

	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"

	"github.com/dudu/metalface/internal/inference"
)

// GPENSize represents the GPEN model input size
type GPENSize int

const (
	GPEN256 GPENSize = 256
	GPEN512 GPENSize = 512
)

// GPEN performs fast face enhancement using GPEN-BFR model
type GPEN struct {
	session   *inference.Session
	inputSize int
}

// NewGPEN creates a new GPEN face enhancer
func NewGPEN(modelPath string, size GPENSize) (*GPEN, error) {
	inputNames := []string{"input"}
	outputNames := []string{"output"}

	session, err := inference.NewSession(modelPath, inputNames, outputNames)
	if err != nil {
		return nil, fmt.Errorf("failed to create GPEN session: %w", err)
	}

	return &GPEN{
		session:   session,
		inputSize: int(size),
	}, nil
}

// Enhance enhances a face image
// Returns enhanced face at the model's native resolution
func (g *GPEN) Enhance(face gocv.Mat) (gocv.Mat, error) {
	// Resize to model input size if needed
	resized := gocv.NewMat()
	if face.Rows() != g.inputSize || face.Cols() != g.inputSize {
		gocv.Resize(face, &resized, image.Pt(g.inputSize, g.inputSize), 0, 0, gocv.InterpolationLinear)
	} else {
		face.CopyTo(&resized)
	}
	defer resized.Close()

	// Manual preprocessing to match Python exactly:
	// 1. BGR to RGB: [:,:,::-1]
	// 2. /255.0 to [0,1]
	// 3. (x-0.5)/0.5 to [-1,1]
	// 4. HWC to CHW transpose
	size := g.inputSize
	floatData := make([]float32, 3*size*size)

	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			// Get BGR pixel from OpenCV Mat
			pixel := resized.GetVecbAt(y, x)
			b := float32(pixel[0])
			gg := float32(pixel[1])
			r := float32(pixel[2])

			// Normalize to [-1, 1]: (pixel/255 - 0.5) / 0.5
			r = (r/255.0 - 0.5) / 0.5
			gg = (gg/255.0 - 0.5) / 0.5
			b = (b/255.0 - 0.5) / 0.5

			// Store in NCHW format (RGB order)
			idx := y*size + x
			floatData[0*size*size+idx] = r // Channel 0 = R
			floatData[1*size*size+idx] = gg // Channel 1 = G
			floatData[2*size*size+idx] = b // Channel 2 = B
		}
	}

	// Create input tensor
	inputTensor, err := ort.NewTensor(
		ort.NewShape(1, 3, int64(g.inputSize), int64(g.inputSize)),
		floatData,
	)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Create output tensor
	outputTensor, err := inference.CreateEmptyTensor[float32]([]int64{1, 3, int64(g.inputSize), int64(g.inputSize)})
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Run inference
	err = g.session.Run([]ort.Value{inputTensor}, []ort.Value{outputTensor})
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("GPEN inference failed: %w", err)
	}

	// Get output data
	output := outputTensor.GetData()

	// Convert NCHW output to HWC image
	result := g.postprocess(output)

	return result, nil
}

// postprocess converts model output to BGR image
func (g *GPEN) postprocess(output []float32) gocv.Mat {
	// Output is in NCHW format (RGB order), values in [-1, 1]
	// Denormalization: pixel = (value + 1) * 127.5

	size := g.inputSize * g.inputSize
	pixels := make([]byte, size*3)

	for y := 0; y < g.inputSize; y++ {
		for x := 0; x < g.inputSize; x++ {
			idx := y*g.inputSize + x

			// Get channel values from NCHW output
			// Model outputs RGB order (channel 0=R, 1=G, 2=B)
			c0 := output[0*size+idx]
			c1 := output[1*size+idx]
			c2 := output[2*size+idx]

			// Denormalize from [-1, 1] to [0, 255]
			// Formula: (clip(x, -1, 1) + 1) * 0.5 * 255
			c0 = gpenClamp(c0, -1, 1)
			c1 = gpenClamp(c1, -1, 1)
			c2 = gpenClamp(c2, -1, 1)
			c0 = (c0 + 1) * 0.5 * 255
			c1 = (c1 + 1) * 0.5 * 255
			c2 = (c2 + 1) * 0.5 * 255

			// Clamp to [0, 255]
			c0 = gpenClamp(c0, 0, 255)
			c1 = gpenClamp(c1, 0, 255)
			c2 = gpenClamp(c2, 0, 255)

			pixIdx := (y*g.inputSize + x) * 3
			// Output is RGB from model, need BGR for OpenCV
			// So: B=c2, G=c1, R=c0
			pixels[pixIdx+0] = uint8(c2) // B
			pixels[pixIdx+1] = uint8(c1) // G
			pixels[pixIdx+2] = uint8(c0) // R
		}
	}

	result, _ := gocv.NewMatFromBytes(g.inputSize, g.inputSize, gocv.MatTypeCV8UC3, pixels)
	return result
}

// Close releases resources
func (g *GPEN) Close() error {
	return g.session.Destroy()
}

func gpenClamp(v, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func gpenBytesToFloat32(b []byte) []float32 {
	floats := make([]float32, len(b)/4)
	for i := 0; i < len(floats); i++ {
		bits := uint32(b[i*4]) | uint32(b[i*4+1])<<8 | uint32(b[i*4+2])<<16 | uint32(b[i*4+3])<<24
		floats[i] = math.Float32frombits(bits)
	}
	return floats
}
