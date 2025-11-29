package pipeline

import (
	"fmt"
	"image"
	"image/color"
	"os"
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
	SimSwap512ModelPath string // Path to SimSwap 512 model
	EmapPath            string // Path to emap.bin for inswapper embedding transformation
	GFPGANModelPath     string
	GPEN256ModelPath    string // Path to GPEN-BFR-256 model (fast)
	GPEN512ModelPath    string // Path to GPEN-BFR-512 model (balanced)
	SourceImagePath     string
	DetectionSize       int
	ConfThreshold       float32
	NMSThreshold        float32
	BlurSize            int
	EnableMouthMask     bool
	EnableColorTransfer bool
	Sharpness           float32
	Backend             Backend      // "onnx" or "coreml"
	Model               ModelType    // "inswapper" or "simswap512"
	Enhancer            EnhancerType // "gfpgan", "gpen256", "gpen512", or "" for none
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
	enhancer        FaceEnhancer
	emap            *swapper.Emap // Expression map for embedding transformation
	sourceEmbedding *swapper.Embedding
	lastTiming      Timing
	backend         Backend
	modelType       ModelType
	// Parallel detection
	lastFaces  []detector.Face
	detectChan chan detectResult
	detecting  bool
	// Frame interpolation for temporal smoothness
	previousFrame *gocv.Mat
	// Debug overlay
	debugLandmarks bool
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
		debugLM     = os.Getenv("METALFACE_DEBUG_LANDMARKS") == "1"
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

		// Also initialize ONNX Runtime if enhancer is enabled (all enhancers use ONNX)
		if config.Enhancer != EnhancerNone {
			fmt.Println("  Initializing ONNX Runtime (for face enhancer)...")
			if err := inference.Initialize(); err != nil {
				return nil, fmt.Errorf("failed to initialize ONNX Runtime: %w", err)
			}
			fmt.Println("  ONNX Runtime initialized")
		}

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

		// Create generator based on model type
		modelType := config.Model
		if modelType == "" {
			modelType = ModelInswapper // Default to inswapper
		}

		if modelType == ModelSimSwap512 {
			fmt.Println("  Loading SimSwap 512 generator...")
			gen, err = swapper.NewSimSwap512(config.SimSwap512ModelPath)
			if err != nil {
				faceDet.Close()
				if landmarkDet != nil {
					landmarkDet.Close()
				}
				enc.Close()
				return nil, fmt.Errorf("failed to create generator: %w", err)
			}
			fmt.Println("  SimSwap 512 generator loaded")
		} else {
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
	}

	// Create aligner and blender
	aligner := swapper.NewFaceAligner()
	blender := swapper.NewBlender(config.BlurSize)

	// Create face enhancer if enabled
	// Note: All enhancers use ONNX - CoreML conversion fails for these models
	var enh FaceEnhancer
	if config.Enhancer != EnhancerNone {
		switch config.Enhancer {
		case EnhancerGPEN256:
			if config.GPEN256ModelPath != "" {
				fmt.Println("  Loading GPEN-256 enhancer (fast)...")
				enh, err = enhancer.NewGPEN(config.GPEN256ModelPath, enhancer.GPEN256)
				if err != nil {
					faceDet.Close()
					if landmarkDet != nil {
						landmarkDet.Close()
					}
					enc.Close()
					gen.Close()
					return nil, fmt.Errorf("failed to create GPEN-256 enhancer: %w", err)
				}
				fmt.Println("  GPEN-256 enhancer loaded")
			}
		case EnhancerGPEN512:
			if config.GPEN512ModelPath != "" {
				fmt.Println("  Loading GPEN-512 enhancer (balanced)...")
				enh, err = enhancer.NewGPEN(config.GPEN512ModelPath, enhancer.GPEN512)
				if err != nil {
					faceDet.Close()
					if landmarkDet != nil {
						landmarkDet.Close()
					}
					enc.Close()
					gen.Close()
					return nil, fmt.Errorf("failed to create GPEN-512 enhancer: %w", err)
				}
				fmt.Println("  GPEN-512 enhancer loaded")
			}
		case EnhancerGFPGAN:
			if config.GFPGANModelPath != "" {
				fmt.Println("  Loading GFPGAN enhancer (high quality, slow)...")
				enh, err = enhancer.NewGFPGAN(config.GFPGANModelPath)
				if err != nil {
					faceDet.Close()
					if landmarkDet != nil {
						landmarkDet.Close()
					}
					enc.Close()
					gen.Close()
					return nil, fmt.Errorf("failed to create GFPGAN enhancer: %w", err)
				}
				fmt.Println("  GFPGAN enhancer loaded")
			}
		}
	}

	// Load emap for embedding transformation
	var emap *swapper.Emap
	if config.EmapPath != "" {
		fmt.Println("  Loading emap...")
		emap, err = swapper.LoadEmap(config.EmapPath)
		if err != nil {
			faceDet.Close()
			if landmarkDet != nil {
				landmarkDet.Close()
			}
			enc.Close()
			gen.Close()
			return nil, fmt.Errorf("failed to load emap: %w", err)
		}
		fmt.Println("  Emap loaded (512x512 transformation matrix)")
	}

	// Determine model type
	modelType := config.Model
	if modelType == "" {
		modelType = ModelInswapper
	}

	p := &Pipeline{
		config:         config,
		faceDetector:   faceDet,
		landmarkDet:    landmarkDet,
		encoder:        enc,
		generator:      gen,
		aligner:        aligner,
		blender:        blender,
		enhancer:       enh,
		emap:           emap,
		backend:        backend,
		modelType:      modelType,
		detectChan:     make(chan detectResult, 1),
		debugLandmarks: debugLM,
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

	// Apply emap transformation only for inswapper model
	// This transforms ArcFace embedding to inswapper latent space
	// SimSwap uses raw ArcFace embedding directly
	if p.emap != nil && p.modelType == ModelInswapper {
		embedding = p.emap.TransformEmbedding(embedding)
	}

	p.sourceEmbedding = embedding
	return nil
}

// drawDebugLandmarks overlays 5-point (and optionally 106) landmarks for visual inspection
func (p *Pipeline) drawDebugLandmarks(frame *gocv.Mat, lm detector.Landmarks, lm106 *detector.Landmarks106) {
	// 5-point in distinct colors
	colors := []color.RGBA{
		{255, 0, 0, 0},   // left eye
		{0, 255, 0, 0},   // right eye
		{0, 0, 255, 0},   // nose
		{255, 255, 0, 0}, // left mouth
		{0, 255, 255, 0}, // right mouth
	}
	pts := []detector.Point{lm.LeftEye, lm.RightEye, lm.Nose, lm.LeftMouth, lm.RightMouth}
	for i, pt := range pts {
		gocv.Circle(frame, image.Pt(int(pt.X), int(pt.Y)), 3, colors[i], 2)
	}

	// Optional: light gray dots for 106 landmarks to see spread / bbox
	if lm106 != nil {
		gray := color.RGBA{200, 200, 200, 0}
		for i := 0; i < len(*lm106); i++ {
			gocv.Circle(frame, image.Pt(int((*lm106)[i].X), int((*lm106)[i].Y)), 1, gray, 1)
		}
	}
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

		// Frame interpolation disabled for better lip sync responsiveness
		// The 80/20 blend was causing noticeable lag in mouth movements
		// TODO: Consider making this configurable if users want smoother output
		// if p.previousFrame != nil && !p.previousFrame.Empty() {
		// 	gocv.AddWeighted(*frame, 0.95, *p.previousFrame, 0.05, 0, frame)
		// }
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

		// Hybrid approach: use SCRFD eyes (stable ordering) + 106-point mouth/nose (fresh for lip sync)
		// SCRFD 5-point has guaranteed correct left/right ordering from model training
		// 106-point mouth positions are fresh (from current frame) for better lip sync
		landmarks := face.Landmarks // SCRFD 5-point (correctly ordered)
		if face.Landmarks106 != nil {
			fresh := face.Landmarks106.GetFivePoint()
			// Only update mouth and nose from fresh 106-point
			// Keep SCRFD eye positions for stable alignment during head tilts
			landmarks.LeftMouth = fresh.LeftMouth
			landmarks.RightMouth = fresh.RightMouth
			landmarks.Nose = fresh.Nose
		}
		if p.debugLandmarks {
			p.drawDebugLandmarks(frame, landmarks, face.Landmarks106)
		}

		// Align face based on model type
		var aligned *swapper.AlignResult
		var err error
		var faceSize int

		if p.modelType == ModelSimSwap512 {
			aligned, err = p.aligner.AlignForSimSwap512(*frame, landmarks)
			faceSize = 512
		} else {
			aligned, err = p.aligner.AlignForInswapper(*frame, landmarks)
			faceSize = 128
		}
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
				// Resize enhanced face back to original swap size using high-quality interpolation
				resized := gocv.NewMat()
				gocv.Resize(enhanced, &resized, image.Pt(faceSize, faceSize), 0, 0, gocv.InterpolationLanczos4)
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
