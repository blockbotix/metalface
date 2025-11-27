// coreml_wrapper.h - C interface for CoreML inference
#ifndef COREML_WRAPPER_H
#define COREML_WRAPPER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to CoreML model
typedef void* CoreMLModelHandle;

// Initialize CoreML (call once at startup)
int coreml_init(void);

// Load a CoreML model from .mlpackage path
// Returns handle on success, NULL on failure
CoreMLModelHandle coreml_load_model(const char* model_path);

// Run inference on a model
// input_data: float array in NCHW format
// input_shape: array of 4 ints [N, C, H, W]
// output_data: pre-allocated float array for output
// output_size: size of output array
// Returns 0 on success, non-zero on error
int coreml_run_inference(
    CoreMLModelHandle handle,
    const float* input_data,
    const int64_t* input_shape,
    int num_inputs,
    float* output_data,
    size_t output_size
);

// Run inference with multiple inputs (for inswapper: face + embedding)
int coreml_run_inference_multi(
    CoreMLModelHandle handle,
    const float** input_data_array,
    const int64_t** input_shapes,
    const int* input_ndims,
    int num_inputs,
    float* output_data,
    size_t output_size
);

// Release a model
void coreml_release_model(CoreMLModelHandle handle);

// Shutdown CoreML
void coreml_shutdown(void);

// Get last error message
const char* coreml_get_error(void);

#ifdef __cplusplus
}
#endif

#endif // COREML_WRAPPER_H
