package coreml

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Foundation -framework CoreML
#include "coreml_wrapper.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

// Model represents a CoreML model
type Model struct {
	handle C.CoreMLModelHandle
}

// Initialize initializes CoreML (call once at startup)
func Initialize() error {
	if C.coreml_init() != 0 {
		return fmt.Errorf("failed to initialize CoreML: %s", C.GoString(C.coreml_get_error()))
	}
	return nil
}

// Shutdown cleans up CoreML
func Shutdown() {
	C.coreml_shutdown()
}

// LoadModel loads a CoreML model from an .mlpackage path
func LoadModel(modelPath string) (*Model, error) {
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	handle := C.coreml_load_model(cPath)
	if handle == nil {
		return nil, fmt.Errorf("failed to load model %s: %s", modelPath, C.GoString(C.coreml_get_error()))
	}

	return &Model{handle: handle}, nil
}

// RunInference runs inference with a single input
func (m *Model) RunInference(inputData []float32, inputShape []int64, outputSize int) ([]float32, error) {
	if m.handle == nil {
		return nil, fmt.Errorf("model not loaded")
	}

	// Prepare output buffer
	output := make([]float32, outputSize)

	// Convert shapes to C types
	cInputShape := make([]C.int64_t, len(inputShape))
	for i, s := range inputShape {
		cInputShape[i] = C.int64_t(s)
	}

	result := C.coreml_run_inference(
		m.handle,
		(*C.float)(unsafe.Pointer(&inputData[0])),
		(*C.int64_t)(unsafe.Pointer(&cInputShape[0])),
		C.int(len(inputShape)),
		(*C.float)(unsafe.Pointer(&output[0])),
		C.size_t(outputSize),
	)

	if result != 0 {
		return nil, fmt.Errorf("inference failed: %s", C.GoString(C.coreml_get_error()))
	}

	return output, nil
}

// RunInferenceMulti runs inference with multiple inputs
func (m *Model) RunInferenceMulti(inputs [][]float32, shapes [][]int64, outputSize int) ([]float32, error) {
	if m.handle == nil {
		return nil, fmt.Errorf("model not loaded")
	}

	numInputs := len(inputs)
	if numInputs == 0 {
		return nil, fmt.Errorf("no inputs provided")
	}

	// Pin Go memory to prevent GC from moving it during C call
	var pinner runtime.Pinner
	defer pinner.Unpin()

	// Allocate C arrays to avoid CGo pointer rules violation
	inputDataPtrs := C.malloc(C.size_t(numInputs) * C.size_t(unsafe.Sizeof(uintptr(0))))
	shapePtrs := C.malloc(C.size_t(numInputs) * C.size_t(unsafe.Sizeof(uintptr(0))))
	ndimsArr := C.malloc(C.size_t(numInputs) * C.size_t(unsafe.Sizeof(C.int(0))))
	defer C.free(inputDataPtrs)
	defer C.free(shapePtrs)
	defer C.free(ndimsArr)

	// Allocate shape arrays in C memory
	cShapePtrs := make([]unsafe.Pointer, numInputs)
	for i := 0; i < numInputs; i++ {
		cShapePtrs[i] = C.malloc(C.size_t(len(shapes[i])) * C.size_t(unsafe.Sizeof(C.int64_t(0))))
		defer C.free(cShapePtrs[i])
	}

	for i := 0; i < numInputs; i++ {
		// Pin the input data slice
		pinner.Pin(&inputs[i][0])

		// Set input data pointer
		*(**C.float)(unsafe.Pointer(uintptr(inputDataPtrs) + uintptr(i)*unsafe.Sizeof(uintptr(0)))) =
			(*C.float)(unsafe.Pointer(&inputs[i][0]))

		// Copy shape to C memory and set shape pointer
		cShape := (*[8]C.int64_t)(cShapePtrs[i])
		for j, s := range shapes[i] {
			cShape[j] = C.int64_t(s)
		}
		*(**C.int64_t)(unsafe.Pointer(uintptr(shapePtrs) + uintptr(i)*unsafe.Sizeof(uintptr(0)))) =
			(*C.int64_t)(cShapePtrs[i])

		// Set ndims
		*(*C.int)(unsafe.Pointer(uintptr(ndimsArr) + uintptr(i)*unsafe.Sizeof(C.int(0)))) = C.int(len(shapes[i]))
	}

	// Prepare output buffer
	output := make([]float32, outputSize)
	pinner.Pin(&output[0])

	result := C.coreml_run_inference_multi(
		m.handle,
		(**C.float)(inputDataPtrs),
		(**C.int64_t)(shapePtrs),
		(*C.int)(ndimsArr),
		C.int(numInputs),
		(*C.float)(unsafe.Pointer(&output[0])),
		C.size_t(outputSize),
	)

	if result != 0 {
		return nil, fmt.Errorf("inference failed: %s", C.GoString(C.coreml_get_error()))
	}

	return output, nil
}

// Close releases the model
func (m *Model) Close() error {
	if m.handle != nil {
		C.coreml_release_model(m.handle)
		m.handle = nil
	}
	return nil
}
