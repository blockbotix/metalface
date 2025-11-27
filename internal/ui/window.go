package ui

import (
	"fmt"
	"image"
	"image/color"
	"time"

	"gocv.io/x/gocv"
)

// Window manages the preview display
type Window struct {
	window     *gocv.Window
	name       string
	lastFrame  time.Time
	frameCount int
	fps        float64
}

// NewWindow creates a new preview window
func NewWindow(name string) *Window {
	window := gocv.NewWindow(name)
	// Force window to appear on macOS
	window.ResizeWindow(1280, 720)
	window.MoveWindow(100, 100)
	return &Window{
		window:    window,
		name:      name,
		lastFrame: time.Now(),
	}
}

// Show displays a frame and updates FPS counter
func (w *Window) Show(frame *gocv.Mat) {
	w.frameCount++
	now := time.Now()

	// Calculate FPS every second
	elapsed := now.Sub(w.lastFrame)
	if elapsed >= time.Second {
		w.fps = float64(w.frameCount) / elapsed.Seconds()
		w.frameCount = 0
		w.lastFrame = now
	}

	// Draw FPS on frame
	fpsText := fmt.Sprintf("FPS: %.1f", w.fps)
	gocv.PutText(frame, fpsText, image.Pt(10, 30),
		gocv.FontHersheyPlain, 2, color.RGBA{R: 0, G: 255, B: 0, A: 255}, 2)

	w.window.IMShow(*frame)
}

// WaitKey waits for key press, returns key code or -1
func (w *Window) WaitKey(delayMs int) int {
	return w.window.WaitKey(delayMs)
}

// FPS returns current frames per second
func (w *Window) FPS() float64 {
	return w.fps
}

// Close closes the window
func (w *Window) Close() error {
	if w.window != nil {
		return w.window.Close()
	}
	return nil
}
