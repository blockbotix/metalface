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

        // Copy output data using getBytesWithHandler to ensure CPU sync
        size_t copySize = MIN(output_size, (size_t)outputArray.count);
        NSArray<NSNumber*>* strides = outputArray.strides;
        if (strides.count > 0) {
            [outputArray getBytesWithHandler:^(const void * _Nonnull bytes, NSInteger size) {
                memcpy(output_data, bytes, MIN(copySize * sizeof(float), (size_t)size));
            }];
        } else {
            float* outputPtr = (float*)outputArray.dataPointer;
            memcpy(output_data, outputPtr, copySize * sizeof(float));
        }

        return 0;
    }
}

int coreml_run_inference_multi(
    CoreMLModelHandle handle,
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

        // Create input arrays for each input
        for (int inputIdx = 0; inputIdx < num_inputs && inputIdx < (int)wrapper.inputNames.count; inputIdx++) {
            NSString* inputName = wrapper.inputNames[inputIdx];
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

        // Copy output data using getBytesWithHandler to ensure CPU sync
        size_t copySize = MIN(output_size, (size_t)outputArray.count);
        NSArray<NSNumber*>* strides = outputArray.strides;
        if (strides.count > 0) {
            [outputArray getBytesWithHandler:^(const void * _Nonnull bytes, NSInteger size) {
                memcpy(output_data, bytes, MIN(copySize * sizeof(float), (size_t)size));
            }];
        } else {
            float* outputPtr = (float*)outputArray.dataPointer;
            memcpy(output_data, outputPtr, copySize * sizeof(float));
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
