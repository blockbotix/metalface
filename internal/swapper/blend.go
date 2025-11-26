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
// frame: pointer to the original frame (modified in place)
// transform: the alignment transform used (to inverse warp)
// landmarks: original face landmarks (for mask generation)
func (b *Blender) BlendFace(swappedFace gocv.Mat, frame *gocv.Mat, transform gocv.Mat, landmarks detector.Landmarks) {
	// For now, use simple overlay instead of complex blending
	// to verify the pipeline works

	// Inverse warp the swapped face to original frame coordinates
	invTransform := gocv.NewMat()
	gocv.InvertAffineTransform(transform, &invTransform)
	defer invTransform.Close()

	frameSize := image.Pt(frame.Cols(), frame.Rows())

	warpedFace := gocv.NewMat()
	gocv.WarpAffine(swappedFace, &warpedFace, invTransform, frameSize)
	defer warpedFace.Close()

	// Create simple elliptical mask for the warped face area
	mask := gocv.NewMatWithSize(frame.Rows(), frame.Cols(), gocv.MatTypeCV8U)
	defer mask.Close()

	// Get face bounding box from landmarks
	centerX := (landmarks.LeftEye.X + landmarks.RightEye.X + landmarks.Nose.X +
		landmarks.LeftMouth.X + landmarks.RightMouth.X) / 5
	centerY := (landmarks.LeftEye.Y + landmarks.RightEye.Y + landmarks.Nose.Y +
		landmarks.LeftMouth.Y + landmarks.RightMouth.Y) / 5

	// Face size based on eye distance
	eyeDist := landmarks.RightEye.X - landmarks.LeftEye.X
	faceWidth := eyeDist * 2.5
	faceHeight := eyeDist * 3.0

	// Draw filled ellipse for mask
	gocv.Ellipse(&mask,
		image.Pt(int(centerX), int(centerY)),
		image.Pt(int(faceWidth/2), int(faceHeight/2)),
		0, 0, 360,
		color.RGBA{R: 255, G: 255, B: 255, A: 255},
		-1,
	)

	// Blur mask for soft edges
	blurredMask := gocv.NewMat()
	gocv.GaussianBlur(mask, &blurredMask, image.Pt(21, 21), 0, 0, gocv.BorderDefault)
	defer blurredMask.Close()

	// Simple alpha blend using mask
	// Copy warped face to frame where mask > 0
	warpedFace.CopyToWithMask(frame, blurredMask)
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

// BlendFaceEnhanced performs enhanced blending using 106-point landmarks
// Implements techniques from Deep-Live-Cam: color transfer, mouth mask, convex hull mask, sharpening
func (b *Blender) BlendFaceEnhanced(swappedFace gocv.Mat, frame *gocv.Mat, transform gocv.Mat,
	face *detector.Face, enableMouthMask, enableColorTransfer bool, sharpness float32) {

	// Inverse warp the swapped face to original frame coordinates
	invTransform := gocv.NewMat()
	gocv.InvertAffineTransform(transform, &invTransform)
	defer invTransform.Close()

	frameSize := image.Pt(frame.Cols(), frame.Rows())

	warpedFace := gocv.NewMat()
	gocv.WarpAffine(swappedFace, &warpedFace, invTransform, frameSize)
	defer warpedFace.Close()

	// Create face mask using 106 landmarks if available, otherwise use 5-point
	var mask gocv.Mat
	if face.Landmarks106 != nil {
		mask = b.createConvexHullMask(frame.Rows(), frame.Cols(), face.Landmarks106)
	} else {
		mask = b.createEllipseMask(frame.Rows(), frame.Cols(), face.Landmarks)
	}
	defer mask.Close()

	// Apply color transfer if enabled
	if enableColorTransfer {
		b.applyColorTransfer(&warpedFace, frame, mask)
	}

	// Blur mask for soft edges
	blurredMask := gocv.NewMat()
	blurSize := b.blurSize
	if blurSize%2 == 0 {
		blurSize++
	}
	gocv.GaussianBlur(mask, &blurredMask, image.Pt(blurSize, blurSize), 0, 0, gocv.BorderDefault)
	defer blurredMask.Close()

	// Store original frame for mouth preservation
	var originalFrame gocv.Mat
	var mouthMask gocv.Mat
	var mouthBox image.Rectangle
	if enableMouthMask && face.Landmarks106 != nil {
		originalFrame = frame.Clone()
		defer originalFrame.Close()
		mouthMask, mouthBox = b.createMouthMask(frame.Rows(), frame.Cols(), face.Landmarks106)
		defer mouthMask.Close()
	}

	// Blend the swapped face onto the frame
	warpedFace.CopyToWithMask(frame, blurredMask)

	// Restore mouth area from original frame if enabled
	if enableMouthMask && face.Landmarks106 != nil && !mouthMask.Empty() {
		b.restoreMouthArea(frame, &originalFrame, mouthMask, mouthBox)
	}

	// Apply sharpening if enabled
	if sharpness > 0 {
		b.applySharpening(frame, face, sharpness)
	}
}

// createConvexHullMask creates a mask from the convex hull of 106 landmarks
func (b *Blender) createConvexHullMask(height, width int, landmarks *detector.Landmarks106) gocv.Mat {
	mask := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)

	// Get face outline indices (0-32 for chin to ears)
	outlineIndices := detector.GetFaceOutlineIndices()
	points := landmarks.GetPoints(outlineIndices)

	// Convert to gocv.PointVector for convex hull
	pointsVec := gocv.NewPointVectorFromPoints(b.pointsToImagePoints(points))
	defer pointsVec.Close()

	// Compute convex hull
	hull := gocv.NewMat()
	defer hull.Close()
	gocv.ConvexHull(pointsVec, &hull, true, false)

	// Draw filled convex hull
	if !hull.Empty() {
		hullPoints := b.matToPoints(hull)
		if len(hullPoints) >= 3 {
			ptsVec := gocv.NewPointsVectorFromPoints([][]image.Point{hullPoints})
			defer ptsVec.Close()
			gocv.FillPoly(&mask, ptsVec, color.RGBA{R: 255, G: 255, B: 255, A: 255})
		}
	}

	return mask
}

// createEllipseMask creates an elliptical mask from 5-point landmarks (fallback)
func (b *Blender) createEllipseMask(height, width int, landmarks detector.Landmarks) gocv.Mat {
	mask := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)

	// Face center
	centerX := (landmarks.LeftEye.X + landmarks.RightEye.X + landmarks.Nose.X +
		landmarks.LeftMouth.X + landmarks.RightMouth.X) / 5
	centerY := (landmarks.LeftEye.Y + landmarks.RightEye.Y + landmarks.Nose.Y +
		landmarks.LeftMouth.Y + landmarks.RightMouth.Y) / 5

	// Face size based on eye distance
	eyeDist := landmarks.RightEye.X - landmarks.LeftEye.X
	faceWidth := eyeDist * 2.5
	faceHeight := eyeDist * 3.0

	gocv.Ellipse(&mask,
		image.Pt(int(centerX), int(centerY)),
		image.Pt(int(faceWidth/2), int(faceHeight/2)),
		0, 0, 360,
		color.RGBA{R: 255, G: 255, B: 255, A: 255},
		-1,
	)

	return mask
}

// createMouthMask creates a mask for the mouth area using 106 landmarks
func (b *Blender) createMouthMask(height, width int, landmarks *detector.Landmarks106) (gocv.Mat, image.Rectangle) {
	mask := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)
	box := image.Rectangle{}

	// Get lower lip indices
	lipIndices := detector.GetLowerLipIndices()
	points := landmarks.GetPoints(lipIndices)

	if len(points) < 3 {
		return mask, box
	}

	// Calculate center and expand the polygon
	var centerX, centerY float32
	for _, p := range points {
		centerX += p.X
		centerY += p.Y
	}
	centerX /= float32(len(points))
	centerY /= float32(len(points))

	// Expand polygon by 20%
	expansion := float32(1.2)
	expandedPoints := make([]image.Point, len(points))
	minX, minY := width, height
	maxX, maxY := 0, 0
	for i, p := range points {
		newX := int((p.X-centerX)*expansion + centerX)
		newY := int((p.Y-centerY)*expansion + centerY)
		expandedPoints[i] = image.Pt(newX, newY)
		if newX < minX {
			minX = newX
		}
		if newX > maxX {
			maxX = newX
		}
		if newY < minY {
			minY = newY
		}
		if newY > maxY {
			maxY = newY
		}
	}

	// Add padding
	padding := 10
	minX = max(0, minX-padding)
	minY = max(0, minY-padding)
	maxX = min(width, maxX+padding)
	maxY = min(height, maxY+padding)
	box = image.Rect(minX, minY, maxX, maxY)

	// Draw filled polygon
	ptsVec := gocv.NewPointsVectorFromPoints([][]image.Point{expandedPoints})
	defer ptsVec.Close()
	gocv.FillPoly(&mask, ptsVec, color.RGBA{R: 255, G: 255, B: 255, A: 255})

	// Blur for soft edges
	gocv.GaussianBlur(mask, &mask, image.Pt(15, 15), 5, 5, gocv.BorderDefault)

	return mask, box
}

// restoreMouthArea blends the original mouth area back onto the swapped frame
func (b *Blender) restoreMouthArea(frame, original *gocv.Mat, mouthMask gocv.Mat, box image.Rectangle) {
	if box.Empty() {
		return
	}

	// Extract ROIs
	frameROI := frame.Region(box)
	defer frameROI.Close()
	originalROI := original.Region(box)
	defer originalROI.Close()
	maskROI := mouthMask.Region(box)
	defer maskROI.Close()

	// Blend original mouth back
	originalROI.CopyToWithMask(&frameROI, maskROI)
}

// applyColorTransfer transfers color from target frame to swapped face using LAB color space
func (b *Blender) applyColorTransfer(source, target *gocv.Mat, mask gocv.Mat) {
	// Convert to LAB color space
	sourceLab := gocv.NewMat()
	defer sourceLab.Close()
	targetLab := gocv.NewMat()
	defer targetLab.Close()

	gocv.CvtColor(*source, &sourceLab, gocv.ColorBGRToLab)
	gocv.CvtColor(*target, &targetLab, gocv.ColorBGRToLab)

	// Calculate mean and std for source and target
	sourceMeanMat := gocv.NewMat()
	defer sourceMeanMat.Close()
	sourceStdMat := gocv.NewMat()
	defer sourceStdMat.Close()
	targetMeanMat := gocv.NewMat()
	defer targetMeanMat.Close()
	targetStdMat := gocv.NewMat()
	defer targetStdMat.Close()

	gocv.MeanStdDev(sourceLab, &sourceMeanMat, &sourceStdMat)
	gocv.MeanStdDev(targetLab, &targetMeanMat, &targetStdMat)

	// Convert source to float for calculations
	sourceFloat := gocv.NewMat()
	defer sourceFloat.Close()
	sourceLab.ConvertTo(&sourceFloat, gocv.MatTypeCV32FC3)

	// Process each channel
	channels := gocv.Split(sourceFloat)
	resultChannels := make([]gocv.Mat, 3)
	for i := 0; i < 3; i++ {
		resultChannels[i] = gocv.NewMat()
		defer channels[i].Close()
		defer resultChannels[i].Close()

		// Get mean and std for this channel
		srcMean := sourceMeanMat.GetDoubleAt(i, 0)
		srcStd := sourceStdMat.GetDoubleAt(i, 0)
		tgtMean := targetMeanMat.GetDoubleAt(i, 0)
		tgtStd := targetStdMat.GetDoubleAt(i, 0)

		if srcStd < 1e-6 {
			srcStd = 1e-6
		}

		scale := tgtStd / srcStd
		offset := tgtMean - srcMean*scale

		gocv.AddWeighted(channels[i], scale, channels[i], 0, offset, &resultChannels[i])
	}

	resultFloat := gocv.NewMat()
	defer resultFloat.Close()
	gocv.Merge(resultChannels, &resultFloat)

	// Convert back to uint8
	resultLab := gocv.NewMat()
	defer resultLab.Close()
	resultFloat.ConvertTo(&resultLab, gocv.MatTypeCV8UC3)

	// Convert back to BGR
	resultBGR := gocv.NewMat()
	defer resultBGR.Close()
	gocv.CvtColor(resultLab, &resultBGR, gocv.ColorLabToBGR)

	// Copy result to source
	resultBGR.CopyTo(source)
}

// applySharpening applies unsharp mask to the face region
func (b *Blender) applySharpening(frame *gocv.Mat, face *detector.Face, sharpness float32) {
	// Get bounding box from face
	bbox := face.BoundingBox
	x1 := max(0, int(bbox.X1))
	y1 := max(0, int(bbox.Y1))
	x2 := min(frame.Cols(), int(bbox.X2))
	y2 := min(frame.Rows(), int(bbox.Y2))

	if x2 <= x1 || y2 <= y1 {
		return
	}

	// Extract face region
	roi := frame.Region(image.Rect(x1, y1, x2, y2))
	defer roi.Close()

	// Create blurred version
	blurred := gocv.NewMat()
	defer blurred.Close()
	gocv.GaussianBlur(roi, &blurred, image.Pt(0, 0), 2, 2, gocv.BorderDefault)

	// Apply unsharp mask: sharpened = original + sharpness * (original - blurred)
	sharpened := gocv.NewMat()
	defer sharpened.Close()
	gocv.AddWeighted(roi, 1.0+float64(sharpness), blurred, -float64(sharpness), 0, &sharpened)

	// Copy back to frame
	sharpened.CopyTo(&roi)
}

// Helper functions
func (b *Blender) pointsToImagePoints(points []detector.Point) []image.Point {
	result := make([]image.Point, len(points))
	for i, p := range points {
		result[i] = image.Pt(int(p.X), int(p.Y))
	}
	return result
}

func (b *Blender) matToPoints(mat gocv.Mat) []image.Point {
	rows := mat.Rows()
	points := make([]image.Point, rows)
	for i := 0; i < rows; i++ {
		// ConvexHull returns indices, get actual point coordinates
		points[i] = image.Pt(int(mat.GetFloatAt(i, 0)), int(mat.GetFloatAt(i, 1)))
	}
	return points
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Close releases blender resources
func (b *Blender) Close() {
	b.erosionKernel.Close()
	b.dilationKernel.Close()
}
