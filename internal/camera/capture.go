package camera

import (
	"fmt"
	"sync"

	"gocv.io/x/gocv"
)

// Capture manages webcam capture
type Capture struct {
	webcam    *gocv.VideoCapture
	deviceID  int
	targetFPS int
	width     int
	height    int
	mu        sync.Mutex
}

// NewCapture creates a new camera capture from device with default 720p resolution
func NewCapture(deviceID int, targetFPS int) (*Capture, error) {
	return NewCaptureWithResolution(deviceID, targetFPS, 1280, 720)
}

// NewCaptureWithResolution creates a new camera capture with specified resolution
func NewCaptureWithResolution(deviceID int, targetFPS int, width, height int) (*Capture, error) {
	webcam, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		return nil, fmt.Errorf("failed to open camera %d: %w", deviceID, err)
	}

	// Set camera properties
	webcam.Set(gocv.VideoCaptureFrameWidth, float64(width))
	webcam.Set(gocv.VideoCaptureFrameHeight, float64(height))
	webcam.Set(gocv.VideoCaptureFPS, float64(targetFPS))

	// Get actual dimensions (camera may not support requested resolution)
	actualWidth := int(webcam.Get(gocv.VideoCaptureFrameWidth))
	actualHeight := int(webcam.Get(gocv.VideoCaptureFrameHeight))

	return &Capture{
		webcam:    webcam,
		deviceID:  deviceID,
		targetFPS: targetFPS,
		width:     actualWidth,
		height:    actualHeight,
	}, nil
}

// Read captures a frame into the provided Mat
func (c *Capture) Read(frame *gocv.Mat) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.webcam == nil {
		return false
	}

	return c.webcam.Read(frame)
}

// Width returns frame width
func (c *Capture) Width() int {
	return c.width
}

// Height returns frame height
func (c *Capture) Height() int {
	return c.height
}

// Close releases the camera
func (c *Capture) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.webcam != nil {
		err := c.webcam.Close()
		c.webcam = nil
		return err
	}
	return nil
}
