package inference

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	initialized bool
	initMu      sync.Mutex
)

// Initialize sets up ONNX Runtime environment (call once at startup)
func Initialize() error {
	initMu.Lock()
	defer initMu.Unlock()

	if initialized {
		return nil
	}

	// Use official ONNX Runtime with CoreML support
	ort.SetSharedLibraryPath("lib/libonnxruntime.dylib")

	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("failed to initialize ONNX Runtime: %w", err)
	}

	initialized = true
	return nil
}

// Shutdown cleans up ONNX Runtime environment
func Shutdown() error {
	initMu.Lock()
	defer initMu.Unlock()

	if !initialized {
		return nil
	}

	if err := ort.DestroyEnvironment(); err != nil {
		return err
	}

	initialized = false
	return nil
}

// Session wraps an ONNX Runtime inference session
type Session struct {
	session     *ort.DynamicAdvancedSession
	modelPath   string
	inputNames  []string
	outputNames []string
}

// NewSession creates a new inference session from an ONNX model with CoreML acceleration
func NewSession(modelPath string, inputNames, outputNames []string) (*Session, error) {
	if !initialized {
		return nil, fmt.Errorf("ONNX Runtime not initialized, call Initialize() first")
	}

	// Create session options with CoreML execution provider
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer options.Destroy()

	// Enable CoreML execution provider for GPU acceleration on macOS
	// Flag 0 = default settings, use Neural Engine + GPU
	err = options.AppendExecutionProviderCoreML(0)
	if err != nil {
		// CoreML not available, continue with CPU
		fmt.Printf("    [CPU] %s - CoreML failed: %v\n", modelPath, err)
	} else {
		fmt.Printf("    [CoreML] %s\n", modelPath)
	}

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		inputNames,
		outputNames,
		options,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create session for %s: %w", modelPath, err)
	}

	return &Session{
		session:     session,
		modelPath:   modelPath,
		inputNames:  inputNames,
		outputNames: outputNames,
	}, nil
}

// Run executes inference with the given inputs
func (s *Session) Run(inputs []ort.Value, outputs []ort.Value) error {
	return s.session.Run(inputs, outputs)
}

// Destroy releases session resources
func (s *Session) Destroy() error {
	if s.session != nil {
		return s.session.Destroy()
	}
	return nil
}

// CreateTensor creates a float32 tensor with the given shape and data
func CreateTensor[T ort.TensorData](shape []int64, data []T) (*ort.Tensor[T], error) {
	return ort.NewTensor(ort.NewShape(shape...), data)
}

// CreateEmptyTensor creates an uninitialized tensor for output
func CreateEmptyTensor[T ort.TensorData](shape []int64) (*ort.Tensor[T], error) {
	size := int64(1)
	for _, dim := range shape {
		size *= dim
	}
	data := make([]T, size)
	return ort.NewTensor(ort.NewShape(shape...), data)
}
