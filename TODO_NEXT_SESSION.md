# Next Session TODO - CoreML Pipeline Fix

## Current Status
- CoreML backend is WORKING - runs on GPU at ~20 FPS
- Face detection works (subscript-based data access fix applied)
- Face swap executes successfully
- **ISSUE**: Black oval mask appears over face instead of proper blending

## Problem to Fix
The blending mask is showing as a solid black oval instead of properly blending the swapped face onto the target frame. This is likely in the blender code.

## Likely Causes
1. **Color transfer issue** - The swapped face colors may not match
2. **Mask inversion** - The blend mask might be inverted (masking the face instead of revealing it)
3. **Alpha blending** - The alpha values might be wrong in the blend operation
4. **Transform matrix** - The inverse transform to place the swapped face back might have issues

## Files to Investigate
- `internal/swapper/blend.go` - The `BlendFaceEnhanced` function
- `internal/swapper/inswapper_coreml.go` - Check if output format differs from ONNX version
- `internal/coreml/coreml_wrapper.m` - Verify output data is correct

## Quick Debug Steps
1. Run with ONNX backend to confirm blending works: `./metalface --source <image> --backend onnx`
2. Compare the swapped face output between ONNX and CoreML
3. Check if the mask is inverted by temporarily inverting it in blend.go
4. Add debug output to save the swapped face and mask as images

## Key Code Locations
- Blender: `internal/swapper/blend.go:BlendFaceEnhanced()`
- CoreML Inswapper: `internal/swapper/inswapper_coreml.go:Swap()`
- Pipeline swap: `internal/pipeline/pipeline.go:processSwap()`

## What Was Fixed This Session
- `internal/coreml/coreml_wrapper.m`: Changed from `getBytesWithHandler` to subscript access for MLMultiArray data
- This fixed the "scores all zeros" issue where GPU data wasn't syncing to CPU
- The fix uses `outputArray[@[@(i0), @(i1)]]` instead of direct memory access

## Commands
```bash
# Build
go build -o metalface ./cmd/metalface

# Test CoreML (current issue)
./metalface --source "/Users/dudu/Downloads/Official_Presidential_Portrait_of_President_Donald_J._Trump_(2025).jpg" --backend coreml

# Test ONNX (should work correctly for comparison)
./metalface --source "/Users/dudu/Downloads/Official_Presidential_Portrait_of_President_Donald_J._Trump_(2025).jpg" --backend onnx
```

## Completed Tasks
- [x] Create CoreML wrapper (coreml_wrapper.m, coreml_wrapper.h)
- [x] Create CoreML Go bindings (internal/coreml/coreml.go)
- [x] Implement SCRFD CoreML detector
- [x] Implement ArcFace CoreML encoder
- [x] Implement Inswapper CoreML generator
- [x] Fix MLMultiArray GPU->CPU data sync issue
- [x] Integrate into pipeline with --backend flag

## Remaining Tasks
- [ ] Fix black oval blending issue
- [ ] Remove debug logging from coreml_wrapper.m
- [ ] Optimize subscript access (currently slower than direct memory)
- [ ] Add 106-landmark detector CoreML support
- [ ] Add GFPGAN enhancer CoreML support
