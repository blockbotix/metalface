package coreml

import (
	"fmt"
	"sync"
)

var (
	initialized bool
	initMu      sync.Mutex
)

// Session wraps a CoreML inference session
type Session struct {
	model       *Model
	modelPath   string
	inputNames  []string
	outputNames []string
}

// InitializeCoreML sets up CoreML environment (call once at startup)
func InitializeCoreML() error {
	initMu.Lock()
	defer initMu.Unlock()

	if initialized {
		return nil
	}

	if err := Initialize(); err != nil {
		return fmt.Errorf("failed to initialize CoreML: %w", err)
	}

	initialized = true
	fmt.Println("  CoreML initialized (native)")
	return nil
}

// ShutdownCoreML cleans up CoreML environment
func ShutdownCoreML() error {
	initMu.Lock()
	defer initMu.Unlock()

	if !initialized {
		return nil
	}

	Shutdown()
	initialized = false
	return nil
}

// NewSession creates a new CoreML inference session
func NewSession(modelPath string, inputNames, outputNames []string) (*Session, error) {
	if !initialized {
		return nil, fmt.Errorf("CoreML not initialized, call InitializeCoreML() first")
	}

	model, err := LoadModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load model %s: %w", modelPath, err)
	}

	fmt.Printf("    [CoreML Native] %s\n", modelPath)

	return &Session{
		model:       model,
		modelPath:   modelPath,
		inputNames:  inputNames,
		outputNames: outputNames,
	}, nil
}

// Run executes inference with a single input
func (s *Session) Run(inputData []float32, inputShape []int64, outputSize int) ([]float32, error) {
	return s.model.RunInference(inputData, inputShape, outputSize)
}

// RunMulti executes inference with multiple inputs
func (s *Session) RunMulti(inputs [][]float32, shapes [][]int64, outputSize int) ([]float32, error) {
	return s.model.RunInferenceMulti(s.inputNames, inputs, shapes, outputSize)
}

// RunMultiOutput executes inference and concatenates multiple outputs in the specified order
func (s *Session) RunMultiOutput(inputData []float32, inputShape []int64, outputNames []string, outputSize int) ([]float32, error) {
	return s.model.RunInferenceMultiOutput(inputData, inputShape, outputNames, outputSize)
}

// Destroy releases session resources
func (s *Session) Destroy() error {
	if s.model != nil {
		return s.model.Close()
	}
	return nil
}
