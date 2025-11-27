package pipeline

import (
	"fmt"
	"image"
	"time"

	"gocv.io/x/gocv"

	"github.com/dudu/metalface/internal/coreml"
	"github.com/dudu/metalface/internal/detector"
	"github.com/dudu/metalface/internal/enhancer"
	"github.com/dudu/metalface/internal/inference"
	"github.com/dudu/metalface/internal/swapper"
)

// Config holds pipeline configuration
type Config struct {
	SCRFDModelPath      string
	Landmark106Path     string
	ArcFaceModelPath    string
	InswapperModelPath  string
	GFPGANModelPath     string
	SourceImagePath     string
	DetectionSize       int
	ConfThreshold       float32
	NMSThreshold        float32
	BlurSize            int
	EnableMouthMask     bool
	EnableColorTransfer bool
	EnableEnhancer      bool
	Sharpness           float32
	Backend             Backend // "onnx" or "coreml"
}

// Timing holds performance timing information
type Timing struct {
	Detection time.Duration
	Alignment time.Duration
	Embedding time.Duration
	Swap      time.Duration
	Blend     time.Duration
	Total     time.Duration
}

// Pipeline orchestrates the face swap process
type Pipeline struct {
	config          Config
	faceDetector    FaceDetector
	landmarkDet     LandmarkDetector
	encoder         FaceEncoder
	generator       FaceSwapper
	aligner         *swapper.FaceAligner
	blender         *swapper.Blender
	enhancer        *enhancer.GFPGAN
	sourceEmbedding *swapper.Embedding
	lastTiming      Timing
	backend         Backend
	// Parallel detection
	lastFaces  []detector.Face
	detectChan chan detectResult
	detecting  bool
	// Frame interpolation for temporal smoothness
	previousFrame *gocv.Mat
}

type detectResult struct {
	faces    []detector.Face
	err      error
	duration time.Duration
}

// New creates a new face swap pipeline
func New(config Config) (*Pipeline, error) {
	var (
		faceDet     FaceDetector
		landmarkDet LandmarkDetector
		enc         FaceEncoder
		gen         FaceSwapper
		err         error
	)

	// Default to ONNX if not specified
	backend := config.Backend
	if backend == "" {
		backend = BackendONNX
	}

	// Initialize based on backend
	if backend == BackendCoreML {
		fmt.Println("  Initializing CoreML...")
		if err := coreml.InitializeCoreML(); err != nil {
			return nil, fmt.Errorf("failed to initialize CoreML: %w", err)
		}
		fmt.Println("  CoreML initialized")

		// Create CoreML detector
		fmt.Println("  Loading SCRFD detector (CoreML)...")
		faceDet, err = detector.NewSCRFDCoreML(
			config.SCRFDModelPath,
			config.DetectionSize,
			config.ConfThreshold,
			config.NMSThreshold,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create detector: %w", err)
		}
		fmt.Println("  SCRFD detector loaded (CoreML)")

		// Create 106-point landmark detector
		if config.Landmark106Path != "" {
			fmt.Printf("  Loading 106-landmark detector (CoreML) from %s...\n", config.Landmark106Path)
			landmarkDet, err = detector.NewLandmark106CoreML(config.Landmark106Path)
			if err != nil {
				faceDet.Close()
				return nil, fmt.Errorf("failed to create landmark detector: %w", err)
			}
			fmt.Println("  106-landmark detector loaded (CoreML)")
		}

		// Create encoder
		fmt.Println("  Loading ArcFace encoder (CoreML)...")
		enc, err = swapper.NewArcFaceEncoderCoreML(config.ArcFaceModelPath)
		if err != nil {
			faceDet.Close()
			if landmarkDet != nil {
				landmarkDet.Close()
			}
			return nil, fmt.Errorf("failed to create encoder: %w", err)
		}
		fmt.Println("  ArcFace encoder loaded (CoreML)")

		// Create generator
		fmt.Println("  Loading Inswapper generator (CoreML)...")
		gen, err = swapper.NewInswapperCoreML(config.InswapperModelPath)
		if err != nil {
			faceDet.Close()
			if landmarkDet != nil {
				landmarkDet.Close()
			}
			enc.Close()
			return nil, fmt.Errorf("failed to create generator: %w", err)
		}
		fmt.Println("  Inswapper generator loaded (CoreML)")
	} else {
		// ONNX backend
		fmt.Println("  Initializing ONNX Runtime...")
		if err := inference.Initialize(); err != nil {
			return nil, fmt.Errorf("failed to initialize inference: %w", err)
		}
		fmt.Println("  ONNX Runtime initialized")

		// Create ONNX detector
		fmt.Println("  Loading SCRFD detector...")
		faceDet, err = detector.NewSCRFD(
			config.SCRFDModelPath,
			config.DetectionSize,
			config.ConfThreshold,
			config.NMSThreshold,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create detector: %w", err)
		}
		fmt.Println("  SCRFD detector loaded")

		// Create 106-point landmark detector
		if config.Landmark106Path != "" {
			fmt.Printf("  Loading 106-landmark detector from %s...\n", config.Landmark106Path)
			landmarkDet, err = detector.NewLandmark106(config.Landmark106Path)
			if err != nil {
				faceDet.Close()
				return nil, fmt.Errorf("failed to create landmark detector: %w", err)
			}
			fmt.Println("  106-landmark detector loaded")
		}

		// Create encoder
		fmt.Println("  Loading ArcFace encoder...")
		enc, err = swapper.NewArcFaceEncoder(config.ArcFaceModelPath)
		if err != nil {
			faceDet.Close()
			if landmarkDet != nil {
				landmarkDet.Close()
			}
			return nil, fmt.Errorf("failed to create encoder: %w", err)
		}
		fmt.Println("  ArcFace encoder loaded")

		// Create generator
		fmt.Println("  Loading Inswapper generator...")
		gen, err = swapper.NewInswapper(config.InswapperModelPath)
		if err != nil {
			faceDet.Close()
			if landmarkDet != nil {
				landmarkDet.Close()
			}
			enc.Close()
			return nil, fmt.Errorf("failed to create generator: %w", err)
		}
		fmt.Println("  Inswapper generator loaded")
	}

	// Create aligner and blender
	aligner := swapper.NewFaceAligner()
	blender := swapper.NewBlender(config.BlurSize)

	// Create GFPGAN enhancer if enabled (ONNX only for now)
	var enh *enhancer.GFPGAN
	if config.EnableEnhancer && config.GFPGANModelPath != "" && backend == BackendONNX {
		fmt.Println("  Loading GFPGAN enhancer...")
		enh, err = enhancer.NewGFPGAN(config.GFPGANModelPath)
		if err != nil {
			faceDet.Close()
			if landmarkDet != nil {
				landmarkDet.Close()
			}
			enc.Close()
			gen.Close()
			return nil, fmt.Errorf("failed to create enhancer: %w", err)
		}
		fmt.Println("  GFPGAN enhancer loaded")
	}

	p := &Pipeline{
		config:       config,
		faceDetector: faceDet,
		landmarkDet:  landmarkDet,
		encoder:      enc,
		generator:    gen,
		aligner:      aligner,
		blender:      blender,
		enhancer:     enh,
		backend:      backend,
		detectChan:   make(chan detectResult, 1),
	}

	// Load source face
	fmt.Println("  Loading source face...")
	if err := p.loadSourceFace(config.SourceImagePath); err != nil {
		p.Close()
		return nil, fmt.Errorf("failed to load source face: %w", err)
	}
	fmt.Println("  Source face loaded")

	return p, nil
}

// loadSourceFace loads and encodes the source face
func (p *Pipeline) loadSourceFace(imagePath string) error {
	// Load image
	img := gocv.IMRead(imagePath, gocv.IMReadColor)
	if img.Empty() {
		return fmt.Errorf("failed to load image: %s", imagePath)
	}
	defer img.Close()

	// Detect face
	faces, err := p.faceDetector.Detect(img)
	if err != nil {
		return fmt.Errorf("detection failed: %w", err)
	}
	if len(faces) == 0 {
		return fmt.Errorf("no face detected in source image")
	}

	// Use first detected face
	face := faces[0]

	// Align face for ArcFace
	aligned, err := p.aligner.AlignForArcFace(img, face.Landmarks)
	if err != nil {
		return fmt.Errorf("alignment failed: %w", err)
	}
	defer aligned.AlignedFace.Close()
	defer aligned.Transform.Close()

	// Extract embedding
	embedding, err := p.encoder.Extract(aligned.AlignedFace)
	if err != nil {
		return fmt.Errorf("embedding extraction failed: %w", err)
	}

	p.sourceEmbedding = embedding
	return nil
}

// Process performs face swap on a frame - parallel detection, sync swap
func (p *Pipeline) Process(frame *gocv.Mat) error {
	totalStart := time.Now()
	var timing Timing

	// Check if previous detection finished
	select {
	case result := <-p.detectChan:
		if result.err == nil {
			p.lastFaces = result.faces
		}
		timing.Detection = result.duration
		p.detecting = false
	default:
		timing.Detection = 0
	}

	// Start detection for next frame in background
	if !p.detecting {
		p.detecting = true
		frameCopy := frame.Clone()
		go func() {
			start := time.Now()
			faces, err := p.faceDetector.Detect(frameCopy)
			frameCopy.Close()
			p.detectChan <- detectResult{faces: faces, err: err, duration: time.Since(start)}
		}()
	}

	// Perform swap synchronously on current frame using last detected faces
	if len(p.lastFaces) > 0 {
		swapStart := time.Now()
		p.processSwap(frame, p.lastFaces)
		timing.Swap = time.Since(swapStart)

		// Frame interpolation: blend with previous frame for temporal smoothness
		// This reduces flicker (like Deep-Live-Cam's 20% interpolation)
		if p.previousFrame != nil && !p.previousFrame.Empty() {
			// Blend: 80% current + 20% previous
			gocv.AddWeighted(*frame, 0.8, *p.previousFrame, 0.2, 0, frame)
		}

		// Store current frame for next iteration
		if p.previousFrame == nil {
			prevFrame := frame.Clone()
			p.previousFrame = &prevFrame
		} else {
			frame.CopyTo(p.previousFrame)
		}
	}

	timing.Total = time.Since(totalStart)
	p.lastTiming = timing
	return nil
}

// processSwap performs the swap on a frame copy
func (p *Pipeline) processSwap(frame *gocv.Mat, faces []detector.Face) {
	for i := range faces {
		face := &faces[i]

		// Get 106-point landmarks if available
		if p.landmarkDet != nil {
			if err := p.landmarkDet.Detect(*frame, face); err != nil {
				// Landmark detection failed, will use 5-point fallback
			}
		}

		aligned, err := p.aligner.AlignForInswapper(*frame, face.Landmarks)
		if err != nil {
			continue
		}

		swappedFace, err := p.generator.Swap(aligned.AlignedFace, p.sourceEmbedding)
		if err != nil {
			aligned.AlignedFace.Close()
			aligned.Transform.Close()
			continue
		}

		// Apply face enhancement if enabled
		faceToBlend := swappedFace
		if p.enhancer != nil {
			enhanced, err := p.enhancer.Enhance(swappedFace)
			if err == nil {
				// Resize enhanced face back to 128x128 (original swap size)
				resized := gocv.NewMat()
				gocv.Resize(enhanced, &resized, image.Pt(128, 128), 0, 0, gocv.InterpolationLinear)
				enhanced.Close()
				swappedFace.Close()
				faceToBlend = resized
			}
			// If enhancement fails, fall back to unenhanced face
		}

		// Use enhanced blending with 106 landmarks
		p.blender.BlendFaceEnhanced(faceToBlend, frame, aligned.Transform, face,
			p.config.EnableMouthMask, p.config.EnableColorTransfer, p.config.Sharpness)

		faceToBlend.Close()
		aligned.AlignedFace.Close()
		aligned.Transform.Close()
	}
}

// LastTiming returns timing from last Process call
func (p *Pipeline) LastTiming() Timing {
	return p.lastTiming
}

// Close releases pipeline resources
func (p *Pipeline) Close() error {
	var errs []error

	if p.faceDetector != nil {
		if err := p.faceDetector.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if p.landmarkDet != nil {
		if err := p.landmarkDet.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if p.encoder != nil {
		if err := p.encoder.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if p.generator != nil {
		if err := p.generator.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if p.aligner != nil {
		p.aligner.Close()
	}
	if p.blender != nil {
		p.blender.Close()
	}
	if p.enhancer != nil {
		if err := p.enhancer.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if p.previousFrame != nil {
		p.previousFrame.Close()
	}

	// Shutdown the appropriate backend
	if p.backend == BackendCoreML {
		if err := coreml.ShutdownCoreML(); err != nil {
			errs = append(errs, err)
		}
	} else {
		if err := inference.Shutdown(); err != nil {
			errs = append(errs, err)
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("cleanup errors: %v", errs)
	}
	return nil
}
