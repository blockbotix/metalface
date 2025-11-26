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
	SCRFDModelPath     string
	ArcFaceModelPath   string
	InswapperModelPath string
	SourceImagePath    string
	DetectionSize      int
	ConfThreshold      float32
	NMSThreshold       float32
	BlurSize           int
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
	encoder         *swapper.ArcFaceEncoder
	generator       *swapper.Inswapper
	aligner         *swapper.FaceAligner
	blender         *swapper.Blender
	sourceEmbedding *swapper.Embedding
	lastTiming      Timing
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

	// Create encoder
	enc, err := swapper.NewArcFaceEncoder(config.ArcFaceModelPath)
	if err != nil {
		det.Close()
		return nil, fmt.Errorf("failed to create encoder: %w", err)
	}

	// Create generator
	gen, err := swapper.NewInswapper(config.InswapperModelPath)
	if err != nil {
		det.Close()
		enc.Close()
		return nil, fmt.Errorf("failed to create generator: %w", err)
	}

	// Create aligner and blender
	aligner := swapper.NewFaceAligner()
	blender := swapper.NewBlender(config.BlurSize)

	p := &Pipeline{
		config:    config,
		detector:  det,
		encoder:   enc,
		generator: gen,
		aligner:   aligner,
		blender:   blender,
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

// Process performs face swap on a frame
func (p *Pipeline) Process(frame *gocv.Mat) error {
	totalStart := time.Now()
	var timing Timing

	// Detect faces
	detectStart := time.Now()
	faces, err := p.detector.Detect(*frame)
	timing.Detection = time.Since(detectStart)

	if err != nil {
		return fmt.Errorf("detection failed: %w", err)
	}

	// Process each detected face
	for _, face := range faces {
		// Align face for Inswapper
		alignStart := time.Now()
		aligned, err := p.aligner.AlignForInswapper(*frame, face.Landmarks)
		if err != nil {
			continue // Skip this face
		}
		timing.Alignment += time.Since(alignStart)

		// Swap face
		swapStart := time.Now()
		swappedFace, err := p.generator.Swap(aligned.AlignedFace, p.sourceEmbedding)
		if err != nil {
			aligned.AlignedFace.Close()
			aligned.Transform.Close()
			continue // Skip this face
		}
		timing.Swap += time.Since(swapStart)

		// Blend onto frame
		blendStart := time.Now()
		p.blender.BlendFace(swappedFace, *frame, aligned.Transform, face.Landmarks)
		timing.Blend += time.Since(blendStart)

		// Cleanup
		swappedFace.Close()
		aligned.AlignedFace.Close()
		aligned.Transform.Close()
	}

	timing.Total = time.Since(totalStart)
	p.lastTiming = timing

	return nil
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
