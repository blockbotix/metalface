// coreml_wrapper.m - Objective-C implementation of CoreML inference
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import "coreml_wrapper.h"

static NSString* lastError = nil;

// Model wrapper class
@interface CoreMLModelWrapper : NSObject
@property (nonatomic, strong) MLModel* model;
@property (nonatomic, strong) NSArray<NSString*>* inputNames;
@property (nonatomic, strong) NSArray<NSString*>* outputNames;
@end

@implementation CoreMLModelWrapper
@end

int coreml_init(void) {
    // CoreML doesn't require explicit initialization
    return 0;
}

CoreMLModelHandle coreml_load_model(const char* model_path) {
    @autoreleasepool {
        NSString* path = [NSString stringWithUTF8String:model_path];
        NSURL* modelURL = [NSURL fileURLWithPath:path];

        NSError* error = nil;

        // Compile the model if it's an mlpackage
        NSURL* compiledURL = [MLModel compileModelAtURL:modelURL error:&error];
        if (error) {
            lastError = [NSString stringWithFormat:@"Failed to compile model: %@", error.localizedDescription];
            NSLog(@"CoreML compile error: %@", lastError);
            return NULL;
        }

        // Configure for GPU only (ANE can cause fallback issues with some models)
        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndGPU; // CPU + GPU only, avoid ANE fallback issues

        // Load the compiled model
        MLModel* model = [MLModel modelWithContentsOfURL:compiledURL configuration:config error:&error];
        if (error) {
            lastError = [NSString stringWithFormat:@"Failed to load model: %@", error.localizedDescription];
            NSLog(@"CoreML load error: %@", lastError);
            return NULL;
        }

        // Get input/output names
        MLModelDescription* desc = model.modelDescription;
        NSArray* inputNames = [desc.inputDescriptionsByName allKeys];
        NSArray* outputNames = [desc.outputDescriptionsByName allKeys];

        NSLog(@"CoreML model loaded: %@", path);
        NSLog(@"  Inputs: %@", inputNames);
        NSLog(@"  Outputs: %@", outputNames);

        // Create wrapper
        CoreMLModelWrapper* wrapper = [[CoreMLModelWrapper alloc] init];
        wrapper.model = model;
        wrapper.inputNames = inputNames;
        wrapper.outputNames = outputNames;

        // Return retained pointer
        return (__bridge_retained void*)wrapper;
    }
}

int coreml_run_inference(
    CoreMLModelHandle handle,
    const float* input_data,
    const int64_t* input_shape,
    int num_dims,
    float* output_data,
    size_t output_size
) {
    @autoreleasepool {
        CoreMLModelWrapper* wrapper = (__bridge CoreMLModelWrapper*)handle;
        if (!wrapper || !wrapper.model) {
            lastError = @"Invalid model handle";
            return -1;
        }

        NSError* error = nil;

        // Get first input name
        NSString* inputName = wrapper.inputNames.firstObject;
        if (!inputName) {
            lastError = @"No input found in model";
            return -1;
        }

        // Calculate total size
        int64_t totalSize = 1;
        NSMutableArray* shapeArray = [NSMutableArray array];
        for (int i = 0; i < num_dims; i++) {
            totalSize *= input_shape[i];
            [shapeArray addObject:@(input_shape[i])];
        }

        // Create MLMultiArray for input
        MLMultiArray* inputArray = [[MLMultiArray alloc] initWithShape:shapeArray
                                                              dataType:MLMultiArrayDataTypeFloat32
                                                                 error:&error];
        if (error) {
            lastError = [NSString stringWithFormat:@"Failed to create input array: %@", error.localizedDescription];
            return -1;
        }

        // Copy input data
        float* inputPtr = (float*)inputArray.dataPointer;
        memcpy(inputPtr, input_data, totalSize * sizeof(float));

        // Create feature provider
        MLDictionaryFeatureProvider* inputFeatures = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{inputName: inputArray} error:&error];
        if (error) {
            lastError = [NSString stringWithFormat:@"Failed to create feature provider: %@", error.localizedDescription];
            return -1;
        }

        // Run prediction
        id<MLFeatureProvider> output = [wrapper.model predictionFromFeatures:inputFeatures error:&error];
        if (error) {
            lastError = [NSString stringWithFormat:@"Prediction failed: %@", error.localizedDescription];
            return -1;
        }

        // Get output
        NSString* outputName = wrapper.outputNames.firstObject;
        MLFeatureValue* outputValue = [output featureValueForName:outputName];
        MLMultiArray* outputArray = outputValue.multiArrayValue;

        if (!outputArray) {
            lastError = @"No output array in result";
            return -1;
        }

        // Copy output - must use subscript access for proper GPU->CPU sync
        size_t copySize = MIN(output_size, (size_t)outputArray.count);
        NSArray<NSNumber*>* shape = outputArray.shape;
        NSInteger numDims = shape.count;

        if (numDims == 2) {
            // 2D array (N, C) - embeddings like ArcFace
            NSInteger dim0 = [shape[0] integerValue];
            NSInteger dim1 = [shape[1] integerValue];
            size_t outIdx = 0;
            for (NSInteger i0 = 0; i0 < dim0 && outIdx < copySize; i0++) {
                for (NSInteger i1 = 0; i1 < dim1 && outIdx < copySize; i1++) {
                    output_data[outIdx++] = [outputArray[@[@(i0), @(i1)]] floatValue];
                }
            }
        } else {
            for (size_t j = 0; j < copySize; j++) {
                output_data[j] = [[outputArray objectAtIndexedSubscript:j] floatValue];
            }
        }

        return 0;
    }
}

int coreml_run_inference_multi(
    CoreMLModelHandle handle,
    const char** input_names,
    const float** input_data_array,
    const int64_t** input_shapes,
    const int* input_ndims,
    int num_inputs,
    float* output_data,
    size_t output_size
) {
    @autoreleasepool {
        CoreMLModelWrapper* wrapper = (__bridge CoreMLModelWrapper*)handle;
        if (!wrapper || !wrapper.model) {
            lastError = @"Invalid model handle";
            return -1;
        }

        NSError* error = nil;
        NSMutableDictionary* inputDict = [NSMutableDictionary dictionary];

        // Create input arrays for each input using the provided input names
        for (int inputIdx = 0; inputIdx < num_inputs; inputIdx++) {
            NSString* inputName = [NSString stringWithUTF8String:input_names[inputIdx]];
            const float* data = input_data_array[inputIdx];
            const int64_t* shape = input_shapes[inputIdx];
            int ndims = input_ndims[inputIdx];

            // Calculate total size and create shape array
            int64_t totalSize = 1;
            NSMutableArray* shapeArray = [NSMutableArray array];
            for (int i = 0; i < ndims; i++) {
                totalSize *= shape[i];
                [shapeArray addObject:@(shape[i])];
            }

            // Create MLMultiArray
            MLMultiArray* inputArray = [[MLMultiArray alloc] initWithShape:shapeArray
                                                                  dataType:MLMultiArrayDataTypeFloat32
                                                                     error:&error];
            if (error) {
                lastError = [NSString stringWithFormat:@"Failed to create input array %d: %@", inputIdx, error.localizedDescription];
                return -1;
            }

            // Copy data
            float* inputPtr = (float*)inputArray.dataPointer;
            memcpy(inputPtr, data, totalSize * sizeof(float));

            inputDict[inputName] = inputArray;
        }

        // Create feature provider
        MLDictionaryFeatureProvider* inputFeatures = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:inputDict error:&error];
        if (error) {
            lastError = [NSString stringWithFormat:@"Failed to create feature provider: %@", error.localizedDescription];
            return -1;
        }

        // Run prediction
        id<MLFeatureProvider> output = [wrapper.model predictionFromFeatures:inputFeatures error:&error];
        if (error) {
            lastError = [NSString stringWithFormat:@"Prediction failed: %@", error.localizedDescription];
            return -1;
        }

        // Get output
        NSString* outputName = wrapper.outputNames.firstObject;
        MLFeatureValue* outputValue = [output featureValueForName:outputName];
        MLMultiArray* outputArray = outputValue.multiArrayValue;

        if (!outputArray) {
            lastError = @"No output array in result";
            return -1;
        }

        // Copy output - must use subscript access for proper GPU->CPU sync
        size_t copySize = MIN(output_size, (size_t)outputArray.count);
        NSArray<NSNumber*>* shape = outputArray.shape;
        NSInteger numDims = shape.count;

        if (numDims == 4) {
            // 4D array (N, C, H, W) - inswapper output
            NSInteger dim0 = [shape[0] integerValue];
            NSInteger dim1 = [shape[1] integerValue];
            NSInteger dim2 = [shape[2] integerValue];
            NSInteger dim3 = [shape[3] integerValue];
            size_t outIdx = 0;
            for (NSInteger i0 = 0; i0 < dim0 && outIdx < copySize; i0++) {
                for (NSInteger i1 = 0; i1 < dim1 && outIdx < copySize; i1++) {
                    for (NSInteger i2 = 0; i2 < dim2 && outIdx < copySize; i2++) {
                        for (NSInteger i3 = 0; i3 < dim3 && outIdx < copySize; i3++) {
                            output_data[outIdx++] = [outputArray[@[@(i0), @(i1), @(i2), @(i3)]] floatValue];
                        }
                    }
                }
            }
        } else if (numDims == 2) {
            NSInteger dim0 = [shape[0] integerValue];
            NSInteger dim1 = [shape[1] integerValue];
            size_t outIdx = 0;
            for (NSInteger i0 = 0; i0 < dim0 && outIdx < copySize; i0++) {
                for (NSInteger i1 = 0; i1 < dim1 && outIdx < copySize; i1++) {
                    output_data[outIdx++] = [outputArray[@[@(i0), @(i1)]] floatValue];
                }
            }
        } else {
            for (size_t j = 0; j < copySize; j++) {
                output_data[j] = [[outputArray objectAtIndexedSubscript:j] floatValue];
            }
        }

        return 0;
    }
}

int coreml_run_inference_multi_output(
    CoreMLModelHandle handle,
    const float* input_data,
    const int64_t* input_shape,
    int num_dims,
    const char** output_names,
    int num_outputs,
    float* output_data,
    size_t output_size
) {
    @autoreleasepool {
        CoreMLModelWrapper* wrapper = (__bridge CoreMLModelWrapper*)handle;
        if (!wrapper || !wrapper.model) {
            lastError = @"Invalid model handle";
            return -1;
        }

        NSError* error = nil;

        // Get first input name
        NSString* inputName = wrapper.inputNames.firstObject;
        if (!inputName) {
            lastError = @"No input found in model";
            return -1;
        }

        // Calculate total size
        int64_t totalSize = 1;
        NSMutableArray* shapeArray = [NSMutableArray array];
        for (int i = 0; i < num_dims; i++) {
            totalSize *= input_shape[i];
            [shapeArray addObject:@(input_shape[i])];
        }

        // Create MLMultiArray for input
        MLMultiArray* inputArray = [[MLMultiArray alloc] initWithShape:shapeArray
                                                              dataType:MLMultiArrayDataTypeFloat32
                                                                 error:&error];
        if (error) {
            lastError = [NSString stringWithFormat:@"Failed to create input array: %@", error.localizedDescription];
            return -1;
        }

        // Copy input data
        float* inputPtr = (float*)inputArray.dataPointer;
        memcpy(inputPtr, input_data, totalSize * sizeof(float));

        // Create feature provider
        MLDictionaryFeatureProvider* inputFeatures = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{inputName: inputArray} error:&error];
        if (error) {
            lastError = [NSString stringWithFormat:@"Failed to create feature provider: %@", error.localizedDescription];
            return -1;
        }

        // Run prediction
        id<MLFeatureProvider> output = [wrapper.model predictionFromFeatures:inputFeatures error:&error];
        if (error) {
            lastError = [NSString stringWithFormat:@"Prediction failed: %@", error.localizedDescription];
            return -1;
        }

        // Concatenate outputs in specified order
        size_t offset = 0;
        for (int i = 0; i < num_outputs && offset < output_size; i++) {
            NSString* outputName = [NSString stringWithUTF8String:output_names[i]];
            MLFeatureValue* outputValue = [output featureValueForName:outputName];

            if (!outputValue) {
                lastError = [NSString stringWithFormat:@"Output '%@' not found", outputName];
                return -1;
            }

            MLMultiArray* outputArray = outputValue.multiArrayValue;
            if (!outputArray) {
                lastError = [NSString stringWithFormat:@"Output '%@' is not an array", outputName];
                return -1;
            }

            size_t copySize = MIN((size_t)outputArray.count, output_size - offset);

            // Copy output using getBytesWithHandler for CPU sync, with stride-aware access
            NSArray<NSNumber*>* shape = outputArray.shape;
            NSArray<NSNumber*>* strides = outputArray.strides;
            NSInteger numDims = shape.count;
            __block size_t actualCopySize = copySize;

            // Extract shape and stride values outside block
            NSInteger dim0 = numDims > 0 ? [shape[0] integerValue] : 0;
            NSInteger dim1 = numDims > 1 ? [shape[1] integerValue] : 1;
            NSInteger stride0 = strides.count > 0 ? [strides[0] integerValue] : 1;
            NSInteger stride1 = strides.count > 1 ? [strides[1] integerValue] : 1;

            // Use subscript access for all copying - getBytesWithHandler doesn't sync GPU data
            if (numDims == 2) {
                // 2D array (N, C)
                size_t outIdx = 0;
                for (NSInteger i0 = 0; i0 < dim0 && outIdx < actualCopySize; i0++) {
                    for (NSInteger i1 = 0; i1 < dim1 && outIdx < actualCopySize; i1++) {
                        output_data[offset + outIdx] = [outputArray[@[@(i0), @(i1)]] floatValue];
                        outIdx++;
                    }
                }
            } else if (numDims == 1) {
                // 1D array
                for (size_t j = 0; j < copySize; j++) {
                    output_data[offset + j] = [outputArray[@[@(j)]] floatValue];
                }
            } else {
                // Fallback - linear access
                for (size_t j = 0; j < copySize; j++) {
                    output_data[offset + j] = [[outputArray objectAtIndexedSubscript:j] floatValue];
                }
            }

            offset += copySize;
        }

        return 0;
    }
}

void coreml_release_model(CoreMLModelHandle handle) {
    if (handle) {
        CoreMLModelWrapper* wrapper = (__bridge_transfer CoreMLModelWrapper*)handle;
        wrapper = nil; // ARC will release
    }
}

void coreml_shutdown(void) {
    // Nothing to do
}

const char* coreml_get_error(void) {
    if (lastError) {
        return [lastError UTF8String];
    }
    return "No error";
}
