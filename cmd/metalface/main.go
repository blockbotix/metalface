package main

import (
	"flag"
	"fmt"
	"os"
)

type Config struct {
	SourceImage string
	CameraIndex int
	Enhance     bool
	VirtualCam  bool
	Preview     bool
	TargetFPS   int
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

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "MetalFace - Real-time face swapping for Apple Silicon\n\n")
		fmt.Fprintf(os.Stderr, "Usage: metalface [options]\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  metalface --source face.jpg\n")
		fmt.Fprintf(os.Stderr, "  metalface --source face.jpg --enhance --vcam\n")
	}

	flag.Parse()
	return config
}

func run(config Config) error {
	fmt.Println("MetalFace starting...")
	fmt.Printf("Source image: %s\n", config.SourceImage)
	fmt.Printf("Camera index: %d\n", config.CameraIndex)
	fmt.Printf("Enhancement: %v\n", config.Enhance)
	fmt.Printf("Virtual cam: %v\n", config.VirtualCam)
	fmt.Printf("Preview: %v\n", config.Preview)
	fmt.Printf("Target FPS: %d\n", config.TargetFPS)

	// TODO: Implement pipeline
	// 1. Initialize camera capture
	// 2. Load source face and extract embedding
	// 3. Initialize face detector
	// 4. Initialize face swapper
	// 5. Initialize enhancer (if enabled)
	// 6. Initialize virtual camera (if enabled)
	// 7. Run processing loop

	fmt.Println("\nPipeline not yet implemented. See CLAUDE.md for development phases.")
	return nil
}
