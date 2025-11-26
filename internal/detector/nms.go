package detector

import "sort"

// nms performs Non-Maximum Suppression on detected faces
func nms(faces []Face, iouThreshold float32) []Face {
	if len(faces) == 0 {
		return faces
	}

	// Sort by score (descending)
	sort.Slice(faces, func(i, j int) bool {
		return faces[i].Score > faces[j].Score
	})

	keep := make([]bool, len(faces))
	for i := range keep {
		keep[i] = true
	}

	for i := 0; i < len(faces); i++ {
		if !keep[i] {
			continue
		}
		for j := i + 1; j < len(faces); j++ {
			if !keep[j] {
				continue
			}
			if iou(faces[i].BoundingBox, faces[j].BoundingBox) > iouThreshold {
				keep[j] = false
			}
		}
	}

	result := make([]Face, 0, len(faces))
	for i, face := range faces {
		if keep[i] {
			result = append(result, face)
		}
	}

	return result
}

// iou calculates Intersection over Union of two bounding boxes
func iou(a, b BoundingBox) float32 {
	// Intersection
	x1 := max32(a.X1, b.X1)
	y1 := max32(a.Y1, b.Y1)
	x2 := min32(a.X2, b.X2)
	y2 := min32(a.Y2, b.Y2)

	if x1 >= x2 || y1 >= y2 {
		return 0
	}

	intersection := (x2 - x1) * (y2 - y1)
	union := a.Area() + b.Area() - intersection

	if union <= 0 {
		return 0
	}

	return intersection / union
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func min32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}
