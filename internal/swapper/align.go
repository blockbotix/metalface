package swapper

import (
	"image"
	"math"

	"gocv.io/x/gocv"

	"github.com/dudu/metalface/internal/detector"
)

// ArcFace reference landmarks for 112x112 aligned face
var arcfaceDst = []detector.Point{
	{X: 38.2946, Y: 51.6963},  // left eye
	{X: 73.5318, Y: 51.5014},  // right eye
	{X: 56.0252, Y: 71.7366},  // nose
	{X: 41.5493, Y: 92.3655},  // left mouth
	{X: 70.7299, Y: 92.2041},  // right mouth
}

// FaceAligner handles face alignment transformations
type FaceAligner struct {
	arcfaceSize    int
	inswapperSize  int
	arcfaceDstMat  gocv.Mat
	inswapperDstMat gocv.Mat
}

// NewFaceAligner creates a new face aligner
func NewFaceAligner() *FaceAligner {
	// Create destination matrices for ArcFace (112x112)
	arcfaceDstMat := gocv.NewMatWithSize(5, 2, gocv.MatTypeCV32F)
	for i, pt := range arcfaceDst {
		arcfaceDstMat.SetFloatAt(i, 0, pt.X)
		arcfaceDstMat.SetFloatAt(i, 1, pt.Y)
	}

	// Create destination matrices for Inswapper (128x128)
	// Scale reference points from 112 to 128
	scale := float32(128) / float32(112)
	inswapperDstMat := gocv.NewMatWithSize(5, 2, gocv.MatTypeCV32F)
	for i, pt := range arcfaceDst {
		inswapperDstMat.SetFloatAt(i, 0, pt.X*scale)
		inswapperDstMat.SetFloatAt(i, 1, pt.Y*scale)
	}

	return &FaceAligner{
		arcfaceSize:    112,
		inswapperSize:  128,
		arcfaceDstMat:  arcfaceDstMat,
		inswapperDstMat: inswapperDstMat,
	}
}

// AlignResult contains alignment results
type AlignResult struct {
	AlignedFace gocv.Mat // The aligned face image
	Transform   gocv.Mat // 2x3 affine transform matrix
}

// AlignForArcFace aligns a face to 112x112 for ArcFace embedding
func (a *FaceAligner) AlignForArcFace(img gocv.Mat, landmarks detector.Landmarks) (*AlignResult, error) {
	return a.alignFace(img, landmarks, a.arcfaceDstMat, a.arcfaceSize)
}

// AlignForInswapper aligns a face to 128x128 for Inswapper
func (a *FaceAligner) AlignForInswapper(img gocv.Mat, landmarks detector.Landmarks) (*AlignResult, error) {
	return a.alignFace(img, landmarks, a.inswapperDstMat, a.inswapperSize)
}

// alignFace performs the alignment
func (a *FaceAligner) alignFace(img gocv.Mat, landmarks detector.Landmarks, dstPts gocv.Mat, size int) (*AlignResult, error) {
	// Create source points matrix from detected landmarks
	srcPts := gocv.NewMatWithSize(5, 2, gocv.MatTypeCV32F)
	defer srcPts.Close()

	srcPts.SetFloatAt(0, 0, landmarks.LeftEye.X)
	srcPts.SetFloatAt(0, 1, landmarks.LeftEye.Y)
	srcPts.SetFloatAt(1, 0, landmarks.RightEye.X)
	srcPts.SetFloatAt(1, 1, landmarks.RightEye.Y)
	srcPts.SetFloatAt(2, 0, landmarks.Nose.X)
	srcPts.SetFloatAt(2, 1, landmarks.Nose.Y)
	srcPts.SetFloatAt(3, 0, landmarks.LeftMouth.X)
	srcPts.SetFloatAt(3, 1, landmarks.LeftMouth.Y)
	srcPts.SetFloatAt(4, 0, landmarks.RightMouth.X)
	srcPts.SetFloatAt(4, 1, landmarks.RightMouth.Y)

	// Estimate similarity transform
	transform := estimateSimilarityTransform(srcPts, dstPts)

	// Warp the image
	aligned := gocv.NewMat()
	gocv.WarpAffine(img, &aligned, transform, image.Pt(size, size))

	return &AlignResult{
		AlignedFace: aligned,
		Transform:   transform,
	}, nil
}

// InverseWarp applies the inverse transform to put a face back
func (a *FaceAligner) InverseWarp(face gocv.Mat, transform gocv.Mat, targetSize image.Point) gocv.Mat {
	// Invert the transform
	invTransform := gocv.NewMat()
	gocv.InvertAffineTransform(transform, &invTransform)
	defer invTransform.Close()

	// Warp back
	result := gocv.NewMat()
	gocv.WarpAffine(face, &result, invTransform, targetSize)

	return result
}

// Close releases aligner resources
func (a *FaceAligner) Close() {
	a.arcfaceDstMat.Close()
	a.inswapperDstMat.Close()
}

// estimateSimilarityTransform computes a 2D similarity transform (rotation, scale, translation)
// from source points to destination points
func estimateSimilarityTransform(src, dst gocv.Mat) gocv.Mat {
	// We need to solve for: dst = M * src
	// where M is a 2x3 matrix [s*cos(θ), -s*sin(θ), tx; s*sin(θ), s*cos(θ), ty]
	//
	// Using least squares to find optimal s, θ, tx, ty

	n := src.Rows()

	// Compute centroids
	var srcCx, srcCy, dstCx, dstCy float32
	for i := 0; i < n; i++ {
		srcCx += src.GetFloatAt(i, 0)
		srcCy += src.GetFloatAt(i, 1)
		dstCx += dst.GetFloatAt(i, 0)
		dstCy += dst.GetFloatAt(i, 1)
	}
	srcCx /= float32(n)
	srcCy /= float32(n)
	dstCx /= float32(n)
	dstCy /= float32(n)

	// Center the points
	var srcNorm, dstNorm float64
	srcCentered := make([]float32, n*2)
	dstCentered := make([]float32, n*2)

	for i := 0; i < n; i++ {
		srcCentered[i*2] = src.GetFloatAt(i, 0) - srcCx
		srcCentered[i*2+1] = src.GetFloatAt(i, 1) - srcCy
		dstCentered[i*2] = dst.GetFloatAt(i, 0) - dstCx
		dstCentered[i*2+1] = dst.GetFloatAt(i, 1) - dstCy

		srcNorm += float64(srcCentered[i*2]*srcCentered[i*2] + srcCentered[i*2+1]*srcCentered[i*2+1])
		dstNorm += float64(dstCentered[i*2]*dstCentered[i*2] + dstCentered[i*2+1]*dstCentered[i*2+1])
	}

	srcNorm = math.Sqrt(srcNorm)
	dstNorm = math.Sqrt(dstNorm)

	// Compute cross-covariance
	var a11, a12, a21, a22 float64
	for i := 0; i < n; i++ {
		sx := float64(srcCentered[i*2])
		sy := float64(srcCentered[i*2+1])
		dx := float64(dstCentered[i*2])
		dy := float64(dstCentered[i*2+1])

		a11 += sx * dx
		a12 += sx * dy
		a21 += sy * dx
		a22 += sy * dy
	}

	// SVD-like solution for 2D similarity
	// cos(θ) ∝ a11 + a22, sin(θ) ∝ a21 - a12
	norm := math.Sqrt((a11+a22)*(a11+a22) + (a21-a12)*(a21-a12))
	if norm < 1e-10 {
		norm = 1
	}

	cosTheta := (a11 + a22) / norm
	sinTheta := (a21 - a12) / norm

	// Compute scale
	scale := dstNorm / srcNorm

	// Build transformation matrix
	transform := gocv.NewMatWithSize(2, 3, gocv.MatTypeCV64F)
	transform.SetDoubleAt(0, 0, scale*cosTheta)
	transform.SetDoubleAt(0, 1, -scale*sinTheta)
	transform.SetDoubleAt(1, 0, scale*sinTheta)
	transform.SetDoubleAt(1, 1, scale*cosTheta)

	// Translation: dstC - scale * R * srcC
	tx := float64(dstCx) - scale*(cosTheta*float64(srcCx)-sinTheta*float64(srcCy))
	ty := float64(dstCy) - scale*(sinTheta*float64(srcCx)+cosTheta*float64(srcCy))
	transform.SetDoubleAt(0, 2, tx)
	transform.SetDoubleAt(1, 2, ty)

	return transform
}
