package main

import (
	"fmt"
	"os"
	"time"

	"github.com/dudu/metalface/internal/coreml"
)

func main() {
	fmt.Println("CoreML Native Test")
	fmt.Println("==================")

	// Initialize CoreML
	if err := coreml.InitializeCoreML(); err != nil {
		fmt.Printf("Failed to init CoreML: %v\n", err)
		os.Exit(1)
	}
	defer coreml.ShutdownCoreML()

	// Test loading SCRFD model
	fmt.Println("\nLoading SCRFD detector...")
	scrfdSession, err := coreml.NewSession(
		"converted_coreml/scrfd_10g.mlpackage",
		[]string{"input.1"},
		[]string{"448", "471", "494", "451", "474", "497", "454", "477", "500"},
	)
	if err != nil {
		fmt.Printf("Failed to load SCRFD: %v\n", err)
		os.Exit(1)
	}
	defer scrfdSession.Destroy()

	// Test inference with dummy data
	fmt.Println("\nTesting SCRFD inference...")
	inputSize := 1 * 3 * 640 * 640
	inputData := make([]float32, inputSize)
	inputShape := []int64{1, 3, 640, 640}

	// Warm up
	for i := 0; i < 3; i++ {
		_, err := scrfdSession.Run(inputData, inputShape, 16800*1)
		if err != nil {
			fmt.Printf("Warmup inference failed: %v\n", err)
		}
	}

	// Benchmark
	iterations := 10
	start := time.Now()
	for i := 0; i < iterations; i++ {
		_, err := scrfdSession.Run(inputData, inputShape, 16800*1)
		if err != nil {
			fmt.Printf("Inference %d failed: %v\n", i, err)
		}
	}
	elapsed := time.Since(start)
	avgMs := float64(elapsed.Milliseconds()) / float64(iterations)
	fps := 1000.0 / avgMs
	fmt.Printf("SCRFD: %.1f ms avg (%.1f FPS)\n", avgMs, fps)

	// Test ArcFace
	fmt.Println("\nLoading ArcFace encoder...")
	arcfaceSession, err := coreml.NewSession(
		"converted_coreml/arcface.mlpackage",
		[]string{"input.1"},
		[]string{"683"},
	)
	if err != nil {
		fmt.Printf("Failed to load ArcFace: %v\n", err)
	} else {
		defer arcfaceSession.Destroy()

		arcfaceInput := make([]float32, 1*3*112*112)
		arcfaceShape := []int64{1, 3, 112, 112}

		// Warm up
		for i := 0; i < 3; i++ {
			arcfaceSession.Run(arcfaceInput, arcfaceShape, 512)
		}

		// Benchmark
		start = time.Now()
		for i := 0; i < iterations; i++ {
			arcfaceSession.Run(arcfaceInput, arcfaceShape, 512)
		}
		elapsed = time.Since(start)
		avgMs = float64(elapsed.Milliseconds()) / float64(iterations)
		fps = 1000.0 / avgMs
		fmt.Printf("ArcFace: %.1f ms avg (%.1f FPS)\n", avgMs, fps)
	}

	// Test Inswapper
	fmt.Println("\nLoading Inswapper...")
	inswapperSession, err := coreml.NewSession(
		"converted_coreml/inswapper.mlpackage",
		[]string{"target", "source"},
		[]string{"output"},
	)
	if err != nil {
		fmt.Printf("Failed to load Inswapper: %v\n", err)
	} else {
		defer inswapperSession.Destroy()

		// Multi-input test
		targetInput := make([]float32, 1*3*128*128)
		sourceInput := make([]float32, 1*512)
		inputs := [][]float32{targetInput, sourceInput}
		shapes := [][]int64{{1, 3, 128, 128}, {1, 512}}

		// Warm up
		for i := 0; i < 3; i++ {
			inswapperSession.RunMulti(inputs, shapes, 1*3*128*128)
		}

		// Benchmark
		start = time.Now()
		for i := 0; i < iterations; i++ {
			inswapperSession.RunMulti(inputs, shapes, 1*3*128*128)
		}
		elapsed = time.Since(start)
		avgMs = float64(elapsed.Milliseconds()) / float64(iterations)
		fps = 1000.0 / avgMs
		fmt.Printf("Inswapper: %.1f ms avg (%.1f FPS)\n", avgMs, fps)
	}

	fmt.Println("\nCoreML test complete!")
}
