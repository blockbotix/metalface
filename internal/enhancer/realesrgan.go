package enhancer

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"

	"github.com/dudu/metalface/internal/inference"
)

// RealESRGAN performs fast image upscaling using Real-ESRGAN x4v3 tiny model
// This is a 4x upscaler (~5MB model) designed for general scenes
// Input: any size image, Output: 4x upscaled image
type RealESRGAN struct {
	session *inference.Session
}

// NewRealESRGAN creates a new Real-ESRGAN enhancer
func NewRealESRGAN(modelPath string) (*RealESRGAN, error) {
	// The model has dynamic input/output names
	inputNames := []string{"input"}
	outputNames := []string{"output"}

	session, err := inference.NewSession(modelPath, inputNames, outputNames)
	if err != nil {
		return nil, fmt.Errorf("failed to create RealESRGAN session: %w", err)
	}

	return &RealESRGAN{
		session: session,
	}, nil
}

// Enhance upscales a face image by 4x
// Input: 128x128 face -> Output: 512x512 enhanced face
func (r *RealESRGAN) Enhance(face gocv.Mat) (gocv.Mat, error) {
	height := face.Rows()
	width := face.Cols()

	// Preprocess: BGR->RGB, normalize to [0,1]
	floatData := make([]float32, 3*height*width)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			pixel := face.GetVecbAt(y, x)
			b := float32(pixel[0]) / 255.0
			g := float32(pixel[1]) / 255.0
			r := float32(pixel[2]) / 255.0

			// Store in NCHW format (RGB order)
			idx := y*width + x
			floatData[0*height*width+idx] = r // Channel 0 = R
			floatData[1*height*width+idx] = g // Channel 1 = G
			floatData[2*height*width+idx] = b // Channel 2 = B
		}
	}

	// Create input tensor
	inputTensor, err := ort.NewTensor(
		ort.NewShape(1, 3, int64(height), int64(width)),
		floatData,
	)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Output is 4x the input size
	outHeight := height * 4
	outWidth := width * 4

	// Create output tensor
	outputTensor, err := inference.CreateEmptyTensor[float32]([]int64{1, 3, int64(outHeight), int64(outWidth)})
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Run inference
	err = r.session.Run([]ort.Value{inputTensor}, []ort.Value{outputTensor})
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("RealESRGAN inference failed: %w", err)
	}

	// Get output data
	output := outputTensor.GetData()

	// Convert NCHW output to HWC BGR image
	result := r.postprocess(output, outHeight, outWidth)

	return result, nil
}

// postprocess converts model output to BGR image
func (r *RealESRGAN) postprocess(output []float32, height, width int) gocv.Mat {
	// Output is in NCHW format (RGB order), values in [0, 1]
	size := height * width
	pixels := make([]byte, size*3)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			idx := y*width + x

			// Get channel values from NCHW output (RGB order)
			rVal := output[0*size+idx]
			gVal := output[1*size+idx]
			bVal := output[2*size+idx]

			// Denormalize from [0, 1] to [0, 255] and clamp
			rVal = clamp(rVal*255.0, 0, 255)
			gVal = clamp(gVal*255.0, 0, 255)
			bVal = clamp(bVal*255.0, 0, 255)

			pixIdx := (y*width + x) * 3
			// Output is RGB from model, need BGR for OpenCV
			pixels[pixIdx+0] = uint8(bVal) // B
			pixels[pixIdx+1] = uint8(gVal) // G
			pixels[pixIdx+2] = uint8(rVal) // R
		}
	}

	result, _ := gocv.NewMatFromBytes(height, width, gocv.MatTypeCV8UC3, pixels)
	return result
}

// Close releases resources
func (r *RealESRGAN) Close() error {
	return r.session.Destroy()
}
