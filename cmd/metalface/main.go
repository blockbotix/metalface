package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"os"
	"os/signal"
	"runtime"
	"syscall"

	"gocv.io/x/gocv"

	"github.com/dudu/metalface/internal/camera"
	"github.com/dudu/metalface/internal/pipeline"
	"github.com/dudu/metalface/internal/ui"
)

func init() {
	// Lock the main goroutine to the main OS thread.
	// This is required on macOS for OpenCV's highgui (window creation).
	runtime.LockOSThread()
}

type Config struct {
	SourceImage string
	CameraIndex int
	Enhance     bool
	VirtualCam  bool
	Preview     bool
	TargetFPS   int
	Backend     string
}

func main() {
	config := parseFlags()

	if config.SourceImage == "" {
		fmt.Fprintln(os.Stderr, "Error: --source flag is required")
		flag.Usage()
		os.Exit(1)
	}

	if err := run(config); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func parseFlags() Config {
	config := Config{}

	flag.StringVar(&config.SourceImage, "source", "", "Source face image (required)")
	flag.StringVar(&config.SourceImage, "s", "", "Source face image (shorthand)")
	flag.IntVar(&config.CameraIndex, "camera", 0, "Camera device index")
	flag.IntVar(&config.CameraIndex, "c", 0, "Camera device index (shorthand)")
	flag.BoolVar(&config.Enhance, "enhance", false, "Enable face enhancement")
	flag.BoolVar(&config.Enhance, "e", false, "Enable face enhancement (shorthand)")
	flag.BoolVar(&config.VirtualCam, "vcam", false, "Output to virtual camera")
	flag.BoolVar(&config.VirtualCam, "v", false, "Output to virtual camera (shorthand)")
	flag.BoolVar(&config.Preview, "preview", true, "Show preview window")
	flag.BoolVar(&config.Preview, "p", true, "Show preview window (shorthand)")
	flag.IntVar(&config.TargetFPS, "fps", 30, "Target frames per second")
	flag.StringVar(&config.Backend, "backend", "onnx", "Inference backend: onnx or coreml")
	flag.StringVar(&config.Backend, "b", "onnx", "Inference backend (shorthand)")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "MetalFace - Real-time face swapping for Apple Silicon\n\n")
		fmt.Fprintf(os.Stderr, "Usage: metalface [options]\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  metalface --source face.jpg\n")
		fmt.Fprintf(os.Stderr, "  metalface --source face.jpg --backend coreml\n")
		fmt.Fprintf(os.Stderr, "  metalface --source face.jpg --enhance --vcam\n")
	}

	flag.Parse()
	return config
}

func run(config Config) error {
	fmt.Println("MetalFace starting...")

	// Validate backend
	backend := pipeline.Backend(config.Backend)
	if backend != pipeline.BackendONNX && backend != pipeline.BackendCoreML {
		return fmt.Errorf("invalid backend: %s (use 'onnx' or 'coreml')", config.Backend)
	}

	// Configure model paths based on backend
	var scrfdPath, arcfacePath, inswapperPath, gfpganPath string
	if backend == pipeline.BackendCoreML {
		scrfdPath = "converted_coreml/scrfd_10g.mlpackage"
		arcfacePath = "converted_coreml/arcface.mlpackage"
		inswapperPath = "converted_coreml/inswapper.mlpackage"
		gfpganPath = "" // GFPGAN not yet converted to CoreML
	} else {
		scrfdPath = "models/scrfd_10g.onnx"
		arcfacePath = "models/arcface.onnx"
		inswapperPath = "models/inswapper.onnx"
		gfpganPath = "models/gfpgan_1.4.onnx"
	}

	// Create pipeline config
	pipelineConfig := pipeline.Config{
		SCRFDModelPath:      scrfdPath,
		Landmark106Path:     "", // Disabled - 5-point works better for mask alignment
		ArcFaceModelPath:    arcfacePath,
		InswapperModelPath:  inswapperPath,
		GFPGANModelPath:     gfpganPath,
		SourceImagePath:     config.SourceImage,
		DetectionSize:       640,
		ConfThreshold:       0.5,
		NMSThreshold:        0.4,
		BlurSize:            31, // Increased for better feathering (like Deep-Live-Cam)
		EnableMouthMask:     false,
		EnableColorTransfer: true,                                       // Enable LAB color transfer
		EnableEnhancer:      config.Enhance && backend != pipeline.BackendCoreML, // GFPGAN only on ONNX for now
		Sharpness:           0,                                          // Disable for now, can cause artifacts
		Backend:             backend,
	}

	// Initialize pipeline
	fmt.Printf("Loading models (backend: %s)...\n", backend)
	p, err := pipeline.New(pipelineConfig)
	if err != nil {
		return fmt.Errorf("failed to create pipeline: %w", err)
	}
	defer p.Close()
	fmt.Println("Models loaded successfully")

	// Initialize camera
	fmt.Printf("Opening camera %d...\n", config.CameraIndex)
	cam, err := camera.NewCapture(config.CameraIndex, config.TargetFPS)
	if err != nil {
		return fmt.Errorf("failed to open camera: %w", err)
	}
	defer cam.Close()
	fmt.Printf("Camera opened: %dx%d\n", cam.Width(), cam.Height())

	// Create preview window
	var window *ui.Window
	if config.Preview {
		window = ui.NewWindow("MetalFace")
		defer window.Close()
	}

	// Handle signals for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Main loop
	frame := gocv.NewMat()
	defer frame.Close()

	fmt.Println("\nRunning... Press 'q' to quit")

	for {
		select {
		case <-sigChan:
			fmt.Println("\nShutting down...")
			return nil
		default:
		}

		// Capture frame
		if !cam.Read(&frame) {
			continue
		}
		if frame.Empty() {
			continue
		}

		// Process frame
		if err := p.Process(&frame); err != nil {
			fmt.Printf("Warning: %v\n", err)
		}

		// Display timing
		timing := p.LastTiming()
		if timing.Total > 0 {
			fps := 1000.0 / float64(timing.Total.Milliseconds())
			timingText := fmt.Sprintf("D:%.0fms S:%.0fms T:%.0fms (%.1f FPS)",
				float64(timing.Detection.Milliseconds()),
				float64(timing.Swap.Milliseconds()),
				float64(timing.Total.Milliseconds()),
				fps)
			gocv.PutText(&frame, timingText, image.Pt(10, 60),
				gocv.FontHersheyPlain, 1.5, color.RGBA{R: 0, G: 255, B: 0, A: 255}, 2)
			// Print timing to console
			fmt.Printf("\rD:%3.0fms S:%3.0fms T:%3.0fms (%.1f FPS)  ",
				float64(timing.Detection.Milliseconds()),
				float64(timing.Swap.Milliseconds()),
				float64(timing.Total.Milliseconds()),
				fps)
		}

		// Show preview
		if window != nil {
			window.Show(&frame)
			// WaitKey must be called to process window events on macOS
			// Use longer delay to ensure window renders properly
			key := window.WaitKey(10)
			if key == 'q' || key == 27 { // 'q' or ESC
				fmt.Println("\nQuitting...")
				return nil
			}
		}
	}
}
