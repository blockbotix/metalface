package enhancer

import (
	"fmt"
	"image"

	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"

	"github.com/dudu/metalface/internal/inference"
)

// CodeFormer performs face restoration using the CodeFormer model
// Input: 512x512 aligned face, Output: 512x512 restored face
// Uses a codebook lookup transformer for high quality restoration
type CodeFormer struct {
	session *inference.Session
}

// NewCodeFormer creates a new CodeFormer enhancer
func NewCodeFormer(modelPath string) (*CodeFormer, error) {
	// CodeFormer has two inputs: 'x' (face image) and 'w' (fidelity weight)
	// The ONNX model from HuggingFace may have 'w' baked in
	// We try with just 'x' first, fall back to both if needed
	inputNames := []string{"x"}
	outputNames := []string{"output"}

	session, err := inference.NewSession(modelPath, inputNames, outputNames)
	if err != nil {
		// Try with both inputs
		inputNames = []string{"x", "w"}
		session, err = inference.NewSession(modelPath, inputNames, outputNames)
		if err != nil {
			return nil, fmt.Errorf("failed to create CodeFormer session: %w", err)
		}
	}

	return &CodeFormer{
		session: session,
	}, nil
}

// Enhance performs face restoration on a 128x128 or 512x512 face
// Input: aligned face image (any size)
// Output: 512x512 restored face
func (c *CodeFormer) Enhance(face gocv.Mat) (gocv.Mat, error) {
	const targetSize = 512

	// Resize to 512x512 if needed
	var resized gocv.Mat
	if face.Rows() != targetSize || face.Cols() != targetSize {
		resized = gocv.NewMat()
		gocv.Resize(face, &resized, image.Pt(targetSize, targetSize), 0, 0, gocv.InterpolationLanczos4)
	} else {
		resized = face.Clone()
	}
	defer resized.Close()

	// Preprocess: BGR->RGB, normalize to [-1, 1] with mean=0.5, std=0.5
	// Formula: (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1
	floatData := make([]float32, 3*targetSize*targetSize)

	for y := 0; y < targetSize; y++ {
		for x := 0; x < targetSize; x++ {
			pixel := resized.GetVecbAt(y, x)
			b := float32(pixel[0])
			g := float32(pixel[1])
			r := float32(pixel[2])

			// Normalize to [-1, 1]
			r = r/127.5 - 1.0
			g = g/127.5 - 1.0
			b = b/127.5 - 1.0

			// Store in NCHW format (RGB order)
			idx := y*targetSize + x
			floatData[0*targetSize*targetSize+idx] = r // Channel 0 = R
			floatData[1*targetSize*targetSize+idx] = g // Channel 1 = G
			floatData[2*targetSize*targetSize+idx] = b // Channel 2 = B
		}
	}

	// Create input tensor
	inputTensor, err := ort.NewTensor(
		ort.NewShape(1, 3, int64(targetSize), int64(targetSize)),
		floatData,
	)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Create output tensor
	outputTensor, err := inference.CreateEmptyTensor[float32]([]int64{1, 3, int64(targetSize), int64(targetSize)})
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Run inference
	err = c.session.Run([]ort.Value{inputTensor}, []ort.Value{outputTensor})
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("CodeFormer inference failed: %w", err)
	}

	// Get output data
	output := outputTensor.GetData()

	// Convert NCHW output to HWC BGR image
	result := c.postprocess(output, targetSize)

	return result, nil
}

// postprocess converts model output to BGR image
func (c *CodeFormer) postprocess(output []float32, size int) gocv.Mat {
	// Output is in NCHW format (RGB order), values in [-1, 1]
	pixelCount := size * size
	pixels := make([]byte, pixelCount*3)

	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			idx := y*size + x

			// Get channel values from NCHW output (RGB order)
			rVal := output[0*pixelCount+idx]
			gVal := output[1*pixelCount+idx]
			bVal := output[2*pixelCount+idx]

			// Denormalize from [-1, 1] to [0, 255]
			// Formula: (val + 1) * 127.5
			rVal = clamp((rVal+1.0)*127.5, 0, 255)
			gVal = clamp((gVal+1.0)*127.5, 0, 255)
			bVal = clamp((bVal+1.0)*127.5, 0, 255)

			pixIdx := (y*size + x) * 3
			// Output is RGB from model, need BGR for OpenCV
			pixels[pixIdx+0] = uint8(bVal) // B
			pixels[pixIdx+1] = uint8(gVal) // G
			pixels[pixIdx+2] = uint8(rVal) // R
		}
	}

	result, _ := gocv.NewMatFromBytes(size, size, gocv.MatTypeCV8UC3, pixels)
	return result
}

// Close releases resources
func (c *CodeFormer) Close() error {
	return c.session.Destroy()
}
