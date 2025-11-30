package enhancer

import (
	"fmt"
	"image"
	"math"

	"gocv.io/x/gocv"

	"github.com/dudu/metalface/internal/coreml"
)

// GFPGANCoreML performs face enhancement using CoreML
type GFPGANCoreML struct {
	session *coreml.Session
}

// NewGFPGANCoreML creates a new GFPGAN face enhancer using CoreML
func NewGFPGANCoreML(modelPath string) (*GFPGANCoreML, error) {
	inputNames := []string{"input"}
	outputNames := []string{"output"}

	session, err := coreml.NewSession(modelPath, inputNames, outputNames)
	if err != nil {
		return nil, fmt.Errorf("failed to create GFPGAN CoreML session: %w", err)
	}

	return &GFPGANCoreML{
		session: session,
	}, nil
}

// Enhance enhances a face image (expects aligned face crop)
// Returns enhanced face at 512x512 resolution
func (g *GFPGANCoreML) Enhance(face gocv.Mat) (gocv.Mat, error) {
	// Resize to 512x512 if needed
	resized := gocv.NewMat()
	if face.Rows() != gfpganInputSize || face.Cols() != gfpganInputSize {
		gocv.Resize(face, &resized, image.Pt(gfpganInputSize, gfpganInputSize), 0, 0, gocv.InterpolationLinear)
	} else {
		face.CopyTo(&resized)
	}
	defer resized.Close()

	// Preprocess
	inputData := g.preprocess(resized)

	// Run inference (output size: 3 * 512 * 512 = 786432)
	outputSize := 3 * gfpganInputSize * gfpganInputSize
	output, err := g.session.Run(inputData, []int64{1, 3, gfpganInputSize, gfpganInputSize}, outputSize)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("GFPGAN inference failed: %w", err)
	}

	// Postprocess
	result := g.postprocess(output)

	return result, nil
}

// preprocess converts face image to model input format
func (g *GFPGANCoreML) preprocess(img gocv.Mat) []float32 {
	// Convert BGR to RGB
	rgb := gocv.NewMat()
	gocv.CvtColor(img, &rgb, gocv.ColorBGRToRGB)
	defer rgb.Close()

	// Convert to float32
	floatImg := gocv.NewMat()
	rgb.ConvertTo(&floatImg, gocv.MatTypeCV32FC3)
	defer floatImg.Close()

	// Create blob (HWC to NCHW)
	// Normalize to [-1, 1]: scale = 1/127.5, mean = (127.5, 127.5, 127.5)
	// This computes: (pixel - 127.5) / 127.5 = pixel/127.5 - 1
	blob := gocv.BlobFromImage(floatImg, 1.0/127.5, image.Pt(gfpganInputSize, gfpganInputSize),
		gocv.NewScalar(127.5, 127.5, 127.5, 0), false, false)
	defer blob.Close()

	blobData := blob.ToBytes()
	return bytesToFloat32CoreML(blobData)
}

// postprocess converts model output to BGR image
func (g *GFPGANCoreML) postprocess(output []float32) gocv.Mat {
	// Output is in NCHW format, values in [-1, 1]
	// Denormalization: pixel = (value + 1) * 127.5

	size := gfpganInputSize * gfpganInputSize
	pixels := make([]byte, size*3)

	for y := 0; y < gfpganInputSize; y++ {
		for x := 0; x < gfpganInputSize; x++ {
			idx := y*gfpganInputSize + x

			// Get RGB values (NCHW: [0,c,h,w])
			r := output[0*size+idx]
			gg := output[1*size+idx]
			b := output[2*size+idx]

			// Denormalize from [-1, 1] to [0, 255]
			r = (r + 1) * 127.5
			gg = (gg + 1) * 127.5
			b = (b + 1) * 127.5

			// Clamp to [0, 255]
			r = clampCoreML(r, 0, 255)
			gg = clampCoreML(gg, 0, 255)
			b = clampCoreML(b, 0, 255)

			pixIdx := (y*gfpganInputSize + x) * 3
			// BGR order for OpenCV
			pixels[pixIdx+0] = uint8(b)
			pixels[pixIdx+1] = uint8(gg)
			pixels[pixIdx+2] = uint8(r)
		}
	}

	result, _ := gocv.NewMatFromBytes(gfpganInputSize, gfpganInputSize, gocv.MatTypeCV8UC3, pixels)
	return result
}

// Close releases resources
func (g *GFPGANCoreML) Close() error {
	return g.session.Destroy()
}

func clampCoreML(v, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

// bytesToFloat32CoreML converts byte slice to float32 slice
func bytesToFloat32CoreML(b []byte) []float32 {
	floats := make([]float32, len(b)/4)
	for i := 0; i < len(floats); i++ {
		bits := uint32(b[i*4]) | uint32(b[i*4+1])<<8 | uint32(b[i*4+2])<<16 | uint32(b[i*4+3])<<24
		floats[i] = math.Float32frombits(bits)
	}
	return floats
}
