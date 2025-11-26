package pipeline

import (
	"fmt"
	"time"

	"gocv.io/x/gocv"

	"github.com/dudu/metalface/internal/detector"
	"github.com/dudu/metalface/internal/inference"
	"github.com/dudu/metalface/internal/swapper"
)

// Config holds pipeline configuration
type Config struct {
	SCRFDModelPath      string
	Landmark106Path     string
	ArcFaceModelPath    string
	InswapperModelPath  string
	SourceImagePath     string
	DetectionSize       int
	ConfThreshold       float32
	NMSThreshold        float32
	BlurSize            int
	EnableMouthMask     bool
	EnableColorTransfer bool
	Sharpness           float32
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
	detector        *detector.SCRFD
	landmark106     *detector.Landmark106
	encoder         *swapper.ArcFaceEncoder
	generator       *swapper.Inswapper
	aligner         *swapper.FaceAligner
	blender         *swapper.Blender
	sourceEmbedding *swapper.Embedding
	lastTiming      Timing
	// Parallel detection
	lastFaces  []detector.Face
	detectChan chan detectResult
	detecting  bool
}

type detectResult struct {
	faces    []detector.Face
	err      error
	duration time.Duration
}

// New creates a new face swap pipeline
func New(config Config) (*Pipeline, error) {
	// Initialize ONNX Runtime
	if err := inference.Initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize inference: %w", err)
	}

	// Create detector
	det, err := detector.NewSCRFD(
		config.SCRFDModelPath,
		config.DetectionSize,
		config.ConfThreshold,
		config.NMSThreshold,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create detector: %w", err)
	}

	// Create 106-point landmark detector
	var lmk106 *detector.Landmark106
	if config.Landmark106Path != "" {
		lmk106, err = detector.NewLandmark106(config.Landmark106Path)
		if err != nil {
			det.Close()
			return nil, fmt.Errorf("failed to create landmark detector: %w", err)
		}
	}

	// Create encoder
	enc, err := swapper.NewArcFaceEncoder(config.ArcFaceModelPath)
	if err != nil {
		det.Close()
		if lmk106 != nil {
			lmk106.Close()
		}
		return nil, fmt.Errorf("failed to create encoder: %w", err)
	}

	// Create generator
	gen, err := swapper.NewInswapper(config.InswapperModelPath)
	if err != nil {
		det.Close()
		if lmk106 != nil {
			lmk106.Close()
		}
		enc.Close()
		return nil, fmt.Errorf("failed to create generator: %w", err)
	}

	// Create aligner and blender
	aligner := swapper.NewFaceAligner()
	blender := swapper.NewBlender(config.BlurSize)

	p := &Pipeline{
		config:      config,
		detector:    det,
		landmark106: lmk106,
		encoder:     enc,
		generator:   gen,
		aligner:     aligner,
		blender:     blender,
		detectChan:  make(chan detectResult, 1),
	}

	// Load source face
	if err := p.loadSourceFace(config.SourceImagePath); err != nil {
		p.Close()
		return nil, fmt.Errorf("failed to load source face: %w", err)
	}

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
	faces, err := p.detector.Detect(img)
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
			faces, err := p.detector.Detect(frameCopy)
			frameCopy.Close()
			p.detectChan <- detectResult{faces: faces, err: err, duration: time.Since(start)}
		}()
	}

	// Perform swap synchronously on current frame using last detected faces
	if len(p.lastFaces) > 0 {
		swapStart := time.Now()
		p.processSwap(frame, p.lastFaces)
		timing.Swap = time.Since(swapStart)
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
		if p.landmark106 != nil {
			if err := p.landmark106.Detect(*frame, face); err != nil {
				// Continue with 5-point landmarks if 106 fails
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

		// Use enhanced blending with 106 landmarks
		p.blender.BlendFaceEnhanced(swappedFace, frame, aligned.Transform, face,
			p.config.EnableMouthMask, p.config.EnableColorTransfer, p.config.Sharpness)

		swappedFace.Close()
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

	if p.detector != nil {
		if err := p.detector.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if p.landmark106 != nil {
		if err := p.landmark106.Close(); err != nil {
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

	if err := inference.Shutdown(); err != nil {
		errs = append(errs, err)
	}

	if len(errs) > 0 {
		return fmt.Errorf("cleanup errors: %v", errs)
	}
	return nil
}
