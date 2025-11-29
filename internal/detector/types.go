package detector

// Point represents a 2D point
type Point struct {
	X, Y float32
}

// BoundingBox represents a face bounding box
type BoundingBox struct {
	X1, Y1 float32 // top-left
	X2, Y2 float32 // bottom-right
}

// Width returns box width
func (b BoundingBox) Width() float32 {
	return b.X2 - b.X1
}

// Height returns box height
func (b BoundingBox) Height() float32 {
	return b.Y2 - b.Y1
}

// Center returns box center point
func (b BoundingBox) Center() Point {
	return Point{
		X: (b.X1 + b.X2) / 2,
		Y: (b.Y1 + b.Y2) / 2,
	}
}

// Area returns box area
func (b BoundingBox) Area() float32 {
	return b.Width() * b.Height()
}

// Landmarks represents 5 facial landmark points
type Landmarks struct {
	LeftEye    Point // index 0
	RightEye   Point // index 1
	Nose       Point // index 2
	LeftMouth  Point // index 3
	RightMouth Point // index 4
}

// AsSlice returns landmarks as a flat slice [x0,y0,x1,y1,...]
func (l Landmarks) AsSlice() []float32 {
	return []float32{
		l.LeftEye.X, l.LeftEye.Y,
		l.RightEye.X, l.RightEye.Y,
		l.Nose.X, l.Nose.Y,
		l.LeftMouth.X, l.LeftMouth.Y,
		l.RightMouth.X, l.RightMouth.Y,
	}
}

// Landmarks106 represents 106 facial landmark points from insightface
type Landmarks106 [106]Point

// GetFivePoint extracts 5-point landmarks from 106-point landmarks
// (kept for masking/debug; alignment uses SCRFD 5-point)
func (l *Landmarks106) GetFivePoint() Landmarks {
	// Indices 33-42: one eye region (average of 10 points)
	var eye1X, eye1Y float32
	for i := 33; i <= 42; i++ {
		eye1X += l[i].X
		eye1Y += l[i].Y
	}
	eye1X /= 10
	eye1Y /= 10

	// Indices 87-96: other eye region (average of 10 points)
	var eye2X, eye2Y float32
	for i := 87; i <= 96; i++ {
		eye2X += l[i].X
		eye2Y += l[i].Y
	}
	eye2X /= 10
	eye2Y /= 10

	// Mouth corners
	mouth1 := l[52]
	mouth2 := l[61]

	return Landmarks{
		LeftEye:    Point{X: eye1X, Y: eye1Y},
		RightEye:   Point{X: eye2X, Y: eye2Y},
		Nose:       l[86],
		LeftMouth:  mouth1,
		RightMouth: mouth2,
	}
}

// BoundingBox computes tight bounding box around all 106 points
func (l *Landmarks106) BoundingBox() BoundingBox {
	if len(l) == 0 {
		return BoundingBox{}
	}
	minX, minY := l[0].X, l[0].Y
	maxX, maxY := l[0].X, l[0].Y
	for i := 1; i < len(l); i++ {
		if l[i].X < minX {
			minX = l[i].X
		}
		if l[i].X > maxX {
			maxX = l[i].X
		}
		if l[i].Y < minY {
			minY = l[i].Y
		}
		if l[i].Y > maxY {
			maxY = l[i].Y
		}
	}
	return BoundingBox{X1: minX, Y1: minY, X2: maxX, Y2: maxY}
}

// Face represents a detected face
type Face struct {
	BoundingBox  BoundingBox
	Landmarks    Landmarks     // 5-point from SCRFD
	Landmarks106 *Landmarks106 // 106-point from 2d106det (optional)
	Score        float32
}
