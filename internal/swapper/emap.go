package swapper

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// Emap is the 512x512 transformation matrix for inswapper
type Emap [512][512]float32

// LoadEmap loads the emap matrix from a binary file
func LoadEmap(path string) (*Emap, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read emap file: %w", err)
	}

	expectedSize := 512 * 512 * 4 // 512x512 float32
	if len(data) != expectedSize {
		return nil, fmt.Errorf("emap file size mismatch: expected %d, got %d", expectedSize, len(data))
	}

	var emap Emap
	for i := 0; i < 512; i++ {
		for j := 0; j < 512; j++ {
			offset := (i*512 + j) * 4
			bits := binary.LittleEndian.Uint32(data[offset : offset+4])
			emap[i][j] = math.Float32frombits(bits)
		}
	}

	return &emap, nil
}

// TransformEmbedding applies the emap transformation to convert ArcFace embedding
// to the latent space expected by inswapper:
//   latent = embedding @ emap
//   latent = latent / norm(latent)
func (e *Emap) TransformEmbedding(embedding *Embedding) *Embedding {
	var latent Embedding

	// Matrix multiplication: latent = embedding @ emap
	// embedding is 1x512, emap is 512x512, result is 1x512
	for j := 0; j < 512; j++ {
		var sum float32
		for i := 0; i < 512; i++ {
			sum += embedding[i] * e[i][j]
		}
		latent[j] = sum
	}

	// L2 normalize
	var norm float64
	for _, v := range latent {
		norm += float64(v * v)
	}
	norm = math.Sqrt(norm)

	if norm < 1e-10 {
		norm = 1
	}

	for i := range latent {
		latent[i] = latent[i] / float32(norm)
	}

	return &latent
}
