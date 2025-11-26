package main

import (
	"fmt"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: orttest <model.onnx>")
		fmt.Println("\nThis tool tests if ONNX Runtime can load a model.")
		os.Exit(1)
	}

	modelPath := os.Args[1]
	fmt.Printf("Testing ONNX model: %s\n", modelPath)

	// Check if file exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Printf("Error: File not found: %s\n", modelPath)
		os.Exit(1)
	}

	// Set the shared library path for onnxruntime
	ort.SetSharedLibraryPath("/opt/homebrew/lib/libonnxruntime.dylib")

	// Initialize ONNX Runtime
	fmt.Println("Initializing ONNX Runtime...")
	err := ort.InitializeEnvironment()
	if err != nil {
		fmt.Printf("❌ Failed to initialize ONNX Runtime: %v\n", err)
		fmt.Println("\nYou may need to install ONNX Runtime:")
		fmt.Println("  brew install onnxruntime")
		os.Exit(1)
	}
	defer ort.DestroyEnvironment()

	fmt.Println("✓ ONNX Runtime initialized")

	// Get model input/output info
	fmt.Println("\nGetting model info...")
	inputs, outputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		fmt.Printf("❌ Failed to get model info: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("\n✅ SUCCESS! Model loaded successfully.")

	fmt.Printf("\nInputs (%d):\n", len(inputs))
	for name, info := range inputs {
		fmt.Printf("  %s: shape=%v, type=%v\n", name, info.Dimensions, info.DataType)
	}

	fmt.Printf("\nOutputs (%d):\n", len(outputs))
	for name, info := range outputs {
		fmt.Printf("  %s: shape=%v, type=%v\n", name, info.Dimensions, info.DataType)
	}

	// Get metadata
	fmt.Println("\nMetadata:")
	metadata, err := ort.GetModelMetadata(modelPath)
	if err != nil {
		fmt.Printf("  (Could not read metadata: %v)\n", err)
	} else {
		if producer, err := metadata.GetProducerName(); err == nil {
			fmt.Printf("  Producer: %s\n", producer)
		}
		if version, err := metadata.GetVersion(); err == nil {
			fmt.Printf("  Version: %d\n", version)
		}
		if domain, err := metadata.GetDomain(); err == nil {
			fmt.Printf("  Domain: %s\n", domain)
		}
		if desc, err := metadata.GetDescription(); err == nil {
			fmt.Printf("  Description: %s\n", desc)
		}
		metadata.Destroy()
	}
}
