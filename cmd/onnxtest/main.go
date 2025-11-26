package main

import (
	"fmt"
	"os"

	"github.com/tsawler/go-metal/checkpoints"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: onnxtest <model.onnx>")
		fmt.Println("\nThis tool tests if go-metal can load an ONNX model.")
		fmt.Println("\nDownload models first:")
		fmt.Println("  ./scripts/download_models.sh")
		fmt.Println("\nThen test:")
		fmt.Println("  go run ./cmd/onnxtest models/scrfd_10g.onnx")
		os.Exit(1)
	}

	modelPath := os.Args[1]
	fmt.Printf("Testing ONNX model: %s\n", modelPath)

	// Check if file exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Printf("Error: File not found: %s\n", modelPath)
		fmt.Println("\nRun ./scripts/download_models.sh to download models")
		os.Exit(1)
	}

	// Try to import the ONNX model
	fmt.Println("Attempting to import with go-metal...")
	importer := checkpoints.NewONNXImporter()
	checkpoint, err := importer.ImportFromONNX(modelPath)
	if err != nil {
		fmt.Printf("\n❌ FAILED to import ONNX model:\n%v\n", err)
		fmt.Println("\nThis likely means the model uses unsupported operations.")
		fmt.Println("go-metal only supports: Conv, MatMul, Add, Relu, LeakyRelu,")
		fmt.Println("Sigmoid, Tanh, BatchNorm, Dropout, Softmax, Flatten")
		os.Exit(1)
	}

	fmt.Println("\n✅ SUCCESS! Model imported successfully.")
	fmt.Printf("\nModel details:\n")
	fmt.Printf("  Layers: %d\n", len(checkpoint.ModelSpec.Layers))
	fmt.Printf("  Weights: %d tensors\n", len(checkpoint.Weights))

	// Print layer info
	fmt.Println("\nLayers:")
	for i, layer := range checkpoint.ModelSpec.Layers {
		fmt.Printf("  %d: %s (%s)\n", i+1, layer.Name, layer.Type)
	}
}
