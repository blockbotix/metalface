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

// BlendFaceEnhanced performs enhanced blending using insightface-style mask erosion
// Key insight from insightface: erode mask by ~10% of face size before blurring
// This exposes more original face at boundaries, improving lip sync
func (b *Blender) BlendFaceEnhanced(swappedFace gocv.Mat, frame *gocv.Mat, transform gocv.Mat,
	face *detector.Face, enableMouthMask, enableColorTransfer bool, sharpness float32) {

	// Determine face size based on input
	// Could be: 128 (raw inswapper), 256 (GPEN-256 enhanced), 512 (GPEN-512/simswap)
	faceSize := swappedFace.Rows()

	// For 128x128 faces (raw inswapper without enhancement), upscale for better quality warping
	// For enhanced faces (256/512), use directly - they're already high quality
	// NOTE: We do NOT blur the face pixels - insightface blurs the MASK for smooth transitions
	faceToWarp := swappedFace
	upscaledFace := gocv.NewMat()
	workingSize := faceSize // The size we'll work with for warping
	if faceSize == 128 {
		// Upscale without blurring face pixels - mask blur handles smooth transitions
		gocv.Resize(swappedFace, &upscaledFace, image.Pt(256, 256), 0, 0, gocv.InterpolationLanczos4)
		faceToWarp = upscaledFace
		workingSize = 256
	}
	defer upscaledFace.Close()

	// Inverse warp the swapped face to original frame coordinates
	invTransform := gocv.NewMat()
	gocv.InvertAffineTransform(transform, &invTransform)
	defer invTransform.Close()

	// Scale the inverse transform based on the ratio between working size and original 128
	// The transform was computed for 128x128 alignment
	scaledInvTransform := invTransform.Clone()
	defer scaledInvTransform.Close()
	if workingSize != 128 {
		// Scale factor: 128 / workingSize
		// e.g., for 256: scale = 0.5, for 512: scale = 0.25
		scale := 128.0 / float64(workingSize)
		scaledInvTransform.SetDoubleAt(0, 0, invTransform.GetDoubleAt(0, 0)*scale)
		scaledInvTransform.SetDoubleAt(0, 1, invTransform.GetDoubleAt(0, 1)*scale)
		scaledInvTransform.SetDoubleAt(1, 0, invTransform.GetDoubleAt(1, 0)*scale)
		scaledInvTransform.SetDoubleAt(1, 1, invTransform.GetDoubleAt(1, 1)*scale)
		// Translation stays the same
	}

	frameSize := image.Pt(frame.Cols(), frame.Rows())

	// Use Lanczos4 interpolation for high quality warping
	warpedFace := gocv.NewMat()
	gocv.WarpAffineWithParams(faceToWarp, &warpedFace, scaledInvTransform, frameSize,
		gocv.InterpolationLanczos4, gocv.BorderConstant, color.RGBA{0, 0, 0, 0})
	defer warpedFace.Close()

	// Create elliptical mask on the aligned face BEFORE warping
	// This produces smooth edges after warping (unlike a square which shows rotated corners)
	// The ellipse is centered on the aligned face and covers most of it
	smallMask := gocv.NewMatWithSize(workingSize, workingSize, gocv.MatTypeCV8U)
	defer smallMask.Close()
	// Draw ellipse centered in the aligned face space
	// Use slightly smaller than full size to avoid edge artifacts
	centerPt := image.Pt(workingSize/2, workingSize/2)
	// Horizontal axis slightly wider for natural face shape
	axes := image.Pt(int(float64(workingSize)*0.45), int(float64(workingSize)*0.48))
	gocv.Ellipse(&smallMask, centerPt, axes, 0, 0, 360,
		color.RGBA{R: 255, G: 255, B: 255, A: 255}, -1)

	// Warp the mask using the same scaled inverse transform
	warpedMask := gocv.NewMat()
	defer warpedMask.Close()
	gocv.WarpAffine(smallMask, &warpedMask, scaledInvTransform, frameSize)

	// Threshold to clean up interpolation artifacts (insightface: img_white[img_white>20] = 255)
	gocv.Threshold(warpedMask, &warpedMask, 20, 255, gocv.ThresholdBinary)

	// Calculate mask size for erosion (insightface style)
	// Find bounding box of mask to determine erosion amount
	maskPoints := gocv.FindContours(warpedMask, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	maskSize := faceSize // default
	if maskPoints.Size() > 0 {
		rect := gocv.BoundingRect(maskPoints.At(0))
		maskSize = int(float64(rect.Dx()*rect.Dy()) / float64(rect.Dx()+rect.Dy()) * 2) // approximate sqrt(h*w)
	}
	maskPoints.Close()

	// Balanced erosion: ~7% of face size
	// Original insightface uses ~10% which cuts mouth, 5% shows edges
	k := maskSize / 14
	if k < 7 {
		k = 7
	}
	// Make kernel size odd for OpenCV
	if k%2 == 0 {
		k++
	}
	erodeKernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(k, k))
	defer erodeKernel.Close()
	gocv.Erode(warpedMask, &warpedMask, erodeKernel)

	// Gaussian blur for soft edges - insightface uses k = max(mask_size//20, 5), blur = 2*k+1
	// We use slightly larger blur for smoother transitions since we don't blur face pixels
	blurK := maskSize / 15 // More aggressive blur (was /20)
	if blurK < 7 {
		blurK = 7
	}
	blurSize := 2*blurK + 1 // ensure odd
	gocv.GaussianBlur(warpedMask, &warpedMask, image.Pt(blurSize, blurSize), 0, 0, gocv.BorderDefault)

	// Use alpha blending
	b.alphaBlend(&warpedFace, frame, warpedMask)

	// Apply sharpening if enabled (disabled for now - can add artifacts)
	if sharpness > 0 {
		b.applySharpening(frame, face, sharpness)
	}
}

// createMaskFromContent creates a mask from actual face pixels in the warped face
// Uses higher threshold to exclude dark edges/artifacts
func (b *Blender) createMaskFromContent(warpedFace gocv.Mat) gocv.Mat {
	// Convert to grayscale
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(warpedFace, &gray, gocv.ColorBGRToGray)

	// Medium threshold (50) - balanced between face pixels and edges
	mask := gocv.NewMat()
	gocv.Threshold(gray, &mask, 50, 255, gocv.ThresholdBinary)

	// Moderate erosion (13x13) for edge cleanup
	erodeKernel := gocv.GetStructuringElement(gocv.MorphEllipse, image.Pt(13, 13))
	defer erodeKernel.Close()
	gocv.Erode(mask, &mask, erodeKernel)

	return mask
}

// createConvexHullMask creates a mask from the convex hull of 106 landmarks
func (b *Blender) createConvexHullMask(height, width int, landmarks *detector.Landmarks106) gocv.Mat {
	mask := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)

	// Get face outline indices (0-32 for chin to ears)
	outlineIndices := detector.GetFaceOutlineIndices()
	points := landmarks.GetPoints(outlineIndices)
	imgPoints := b.pointsToImagePoints(points)

	// Use our own convex hull implementation that returns actual points
	hullPoints := detector.ConvexHull(points)
	hullImgPoints := b.pointsToImagePoints(hullPoints)

	// Draw filled convex hull
	if len(hullImgPoints) >= 3 {
		ptsVec := gocv.NewPointsVectorFromPoints([][]image.Point{hullImgPoints})
		defer ptsVec.Close()
		gocv.FillPoly(&mask, ptsVec, color.RGBA{R: 255, G: 255, B: 255, A: 255})
	} else {
		// Fallback: just draw polygon from outline points
		if len(imgPoints) >= 3 {
			ptsVec := gocv.NewPointsVectorFromPoints([][]image.Point{imgPoints})
			defer ptsVec.Close()
			gocv.FillPoly(&mask, ptsVec, color.RGBA{R: 255, G: 255, B: 255, A: 255})
		}
	}

	return mask
}

// createEllipseMaskFrom106 creates an elliptical mask from 106 landmarks (current frame)
func (b *Blender) createEllipseMaskFrom106(height, width int, landmarks *detector.Landmarks106) gocv.Mat {
	mask := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)

	// Calculate bounding box from face outline (indices 0-32)
	minX, minY := float32(width), float32(height)
	maxX, maxY := float32(0), float32(0)
	for i := 0; i <= 32; i++ {
		if landmarks[i].X < minX {
			minX = landmarks[i].X
		}
		if landmarks[i].X > maxX {
			maxX = landmarks[i].X
		}
		if landmarks[i].Y < minY {
			minY = landmarks[i].Y
		}
		if landmarks[i].Y > maxY {
			maxY = landmarks[i].Y
		}
	}

	// Face center and size from bounding box
	centerX := (minX + maxX) / 2
	// Shift center up - use 40% from top instead of 50% to avoid chin area
	centerY := minY + (maxY-minY)*0.4
	faceWidth := (maxX - minX) * 0.85  // Slightly less than full width
	faceHeight := (maxY - minY) * 0.7  // Reduced height, centered higher

	gocv.Ellipse(&mask,
		image.Pt(int(centerX), int(centerY)),
		image.Pt(int(faceWidth/2), int(faceHeight/2)),
		0, 0, 360,
		color.RGBA{R: 255, G: 255, B: 255, A: 255},
		-1,
	)

	return mask
}

// createEllipseMask creates an elliptical mask from 5-point landmarks (fallback)
func (b *Blender) createEllipseMask(height, width int, landmarks detector.Landmarks) gocv.Mat {
	mask := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)

	// Face center - shift up slightly to avoid throat area
	centerX := (landmarks.LeftEye.X + landmarks.RightEye.X) / 2
	centerY := (landmarks.LeftEye.Y + landmarks.RightEye.Y + landmarks.Nose.Y) / 3

	// Face size based on eye distance - reduce height to avoid black throat area
	eyeDist := landmarks.RightEye.X - landmarks.LeftEye.X
	faceWidth := eyeDist * 2.2
	faceHeight := eyeDist * 2.5 // Reduced from 3.0 to avoid throat

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

// alphaBlend performs proper alpha blending: result = src * alpha + dst * (1 - alpha)
func (b *Blender) alphaBlend(src, dst *gocv.Mat, mask gocv.Mat) {
	// Convert mask to 3 channels
	mask3ch := gocv.NewMat()
	defer mask3ch.Close()
	gocv.CvtColor(mask, &mask3ch, gocv.ColorGrayToBGR)

	// Convert to float32 for blending
	srcFloat := gocv.NewMat()
	defer srcFloat.Close()
	dstFloat := gocv.NewMat()
	defer dstFloat.Close()
	maskFloat := gocv.NewMat()
	defer maskFloat.Close()

	src.ConvertTo(&srcFloat, gocv.MatTypeCV32FC3)
	dst.ConvertTo(&dstFloat, gocv.MatTypeCV32FC3)
	mask3ch.ConvertTo(&maskFloat, gocv.MatTypeCV32FC3)

	// Normalize mask to 0-1 by dividing by 255
	maskFloat.DivideFloat(255.0)

	// Calculate inverse mask (1 - alpha)
	invMask := gocv.NewMat()
	defer invMask.Close()
	ones := gocv.NewMatWithSizeFromScalar(gocv.NewScalar(1.0, 1.0, 1.0, 0), mask.Rows(), mask.Cols(), gocv.MatTypeCV32FC3)
	defer ones.Close()
	gocv.Subtract(ones, maskFloat, &invMask)

	// src * alpha
	srcWeighted := gocv.NewMat()
	defer srcWeighted.Close()
	gocv.Multiply(srcFloat, maskFloat, &srcWeighted)

	// dst * (1 - alpha)
	dstWeighted := gocv.NewMat()
	defer dstWeighted.Close()
	gocv.Multiply(dstFloat, invMask, &dstWeighted)

	// result = src * alpha + dst * (1 - alpha)
	resultFloat := gocv.NewMat()
	defer resultFloat.Close()
	gocv.Add(srcWeighted, dstWeighted, &resultFloat)

	// Convert back to uint8
	resultFloat.ConvertTo(dst, gocv.MatTypeCV8UC3)
}

// Helper functions
func (b *Blender) pointsToImagePoints(points []detector.Point) []image.Point {
	result := make([]image.Point, len(points))
	for i, p := range points {
		result[i] = image.Pt(int(p.X), int(p.Y))
	}
	return result
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
