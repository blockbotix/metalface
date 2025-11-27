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

// NewCapture creates a new camera capture from device
func NewCapture(deviceID int, targetFPS int) (*Capture, error) {
	webcam, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		return nil, fmt.Errorf("failed to open camera %d: %w", deviceID, err)
	}

	// Set camera properties - use 720p for better performance
	webcam.Set(gocv.VideoCaptureFrameWidth, 1280)
	webcam.Set(gocv.VideoCaptureFrameHeight, 720)
	webcam.Set(gocv.VideoCaptureFPS, float64(targetFPS))

	// Get actual dimensions
	width := int(webcam.Get(gocv.VideoCaptureFrameWidth))
	height := int(webcam.Get(gocv.VideoCaptureFrameHeight))

	return &Capture{
		webcam:    webcam,
		deviceID:  deviceID,
		targetFPS: targetFPS,
		width:     width,
		height:    height,
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
