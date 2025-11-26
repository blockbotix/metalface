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

// Face represents a detected face
type Face struct {
	BoundingBox BoundingBox
	Landmarks   Landmarks
	Score       float32
}
