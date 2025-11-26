package swapper

import (
	"image"
	"image/color"

	"gocv.io/x/gocv"

	"github.com/dudu/metalface/internal/detector"
)

// Blender handles face blending operations
type Blender struct {
	erosionKernel  gocv.Mat
	dilationKernel gocv.Mat
	blurSize       int
}

// NewBlender creates a new face blender
func NewBlender(blurSize int) *Blender {
	erosionKernel := gocv.GetStructuringElement(gocv.MorphEllipse, image.Pt(3, 3))
	dilationKernel := gocv.GetStructuringElement(gocv.MorphEllipse, image.Pt(3, 3))

	return &Blender{
		erosionKernel:  erosionKernel,
		dilationKernel: dilationKernel,
		blurSize:       blurSize,
	}
}

// BlendFace blends a swapped face onto the original frame
// swappedFace: the 128x128 swapped face
// frame: the original frame
// transform: the alignment transform used (to inverse warp)
// landmarks: original face landmarks (for mask generation)
func (b *Blender) BlendFace(swappedFace, frame gocv.Mat, transform gocv.Mat, landmarks detector.Landmarks) {
	// Create face mask
	mask := b.createMask(swappedFace.Cols(), swappedFace.Rows(), landmarks)
	defer mask.Close()

	// Soften mask edges
	softMask := b.softenMask(mask)
	defer softMask.Close()

	// Inverse warp the swapped face and mask to original frame coordinates
	invTransform := gocv.NewMat()
	gocv.InvertAffineTransform(transform, &invTransform)
	defer invTransform.Close()

	frameSize := image.Pt(frame.Cols(), frame.Rows())

	warpedFace := gocv.NewMat()
	gocv.WarpAffine(swappedFace, &warpedFace, invTransform, frameSize)
	defer warpedFace.Close()

	warpedMask := gocv.NewMat()
	gocv.WarpAffine(softMask, &warpedMask, invTransform, frameSize)
	defer warpedMask.Close()

	// Convert mask to 3-channel for blending
	warpedMask3 := gocv.NewMat()
	gocv.CvtColor(warpedMask, &warpedMask3, gocv.ColorGrayToBGR)
	defer warpedMask3.Close()

	// Convert to float for blending
	frameFloat := gocv.NewMat()
	frame.ConvertTo(&frameFloat, gocv.MatTypeCV32FC3)
	defer frameFloat.Close()

	faceFloat := gocv.NewMat()
	warpedFace.ConvertTo(&faceFloat, gocv.MatTypeCV32FC3)
	defer faceFloat.Close()

	maskFloat := gocv.NewMat()
	warpedMask3.ConvertTo(&maskFloat, gocv.MatTypeCV32FC3)
	gocv.Multiply(maskFloat, gocv.NewMatFromScalar(gocv.NewScalar(1.0/255.0, 1.0/255.0, 1.0/255.0, 0), gocv.MatTypeCV32FC3), &maskFloat)
	defer maskFloat.Close()

	// Alpha blend: result = face * mask + frame * (1 - mask)
	invMask := gocv.NewMat()
	gocv.Subtract(gocv.NewMatFromScalar(gocv.NewScalar(1, 1, 1, 0), gocv.MatTypeCV32FC3), maskFloat, &invMask)
	defer invMask.Close()

	term1 := gocv.NewMat()
	gocv.Multiply(faceFloat, maskFloat, &term1)
	defer term1.Close()

	term2 := gocv.NewMat()
	gocv.Multiply(frameFloat, invMask, &term2)
	defer term2.Close()

	result := gocv.NewMat()
	gocv.Add(term1, term2, &result)
	defer result.Close()

	// Convert back to uint8 and copy to frame
	result.ConvertTo(&frame, gocv.MatTypeCV8UC3)
}

// createMask creates a face mask from landmarks
func (b *Blender) createMask(width, height int, landmarks detector.Landmarks) gocv.Mat {
	mask := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)

	// Create convex hull from facial landmarks
	// Scale landmarks to mask size (assumes landmarks are relative to some reference)
	// For simplicity, create an elliptical mask centered on the face

	// Face center (approximate from landmarks)
	centerX := (landmarks.LeftEye.X + landmarks.RightEye.X + landmarks.Nose.X +
		landmarks.LeftMouth.X + landmarks.RightMouth.X) / 5
	centerY := (landmarks.LeftEye.Y + landmarks.RightEye.Y + landmarks.Nose.Y +
		landmarks.LeftMouth.Y + landmarks.RightMouth.Y) / 5

	// Scale to 128x128
	scale := float32(width) / 112.0 // landmarks are in 112x112 space
	centerX *= scale
	centerY *= scale

	// Face dimensions (approximate)
	faceWidth := float32(width) * 0.7
	faceHeight := float32(height) * 0.8

	// Draw filled ellipse
	gocv.Ellipse(&mask,
		image.Pt(int(centerX), int(centerY)),
		image.Pt(int(faceWidth/2), int(faceHeight/2)),
		0, 0, 360,
		color.RGBA{R: 255, G: 255, B: 255, A: 255},
		-1, // filled
	)

	return mask
}

// softenMask applies erosion, dilation, and blur to create soft edges
func (b *Blender) softenMask(mask gocv.Mat) gocv.Mat {
	// Erode to shrink mask slightly
	eroded := gocv.NewMat()
	gocv.Erode(mask, &eroded, b.erosionKernel)

	// Gaussian blur for soft edges
	blurred := gocv.NewMat()
	blurSize := b.blurSize
	if blurSize%2 == 0 {
		blurSize++
	}
	gocv.GaussianBlur(eroded, &blurred, image.Pt(blurSize, blurSize), 0, 0, gocv.BorderDefault)
	eroded.Close()

	return blurred
}

// Close releases blender resources
func (b *Blender) Close() {
	b.erosionKernel.Close()
	b.dilationKernel.Close()
}
