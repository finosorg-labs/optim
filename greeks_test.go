package optim

import (
	"math"
	"testing"
)

func TestGreeksBatch_CallBasic(t *testing.T) {
	S := []float64{100.0}
	K := []float64{100.0}
	T := []float64{1.0}
	r := []float64{0.05}
	sigma := []float64{0.2}
	isCall := []bool{true}
	delta := make([]float64, 1)
	gamma := make([]float64, 1)
	vega := make([]float64, 1)
	theta := make([]float64, 1)
	rho := make([]float64, 1)

	err := GreeksBatch(S, K, T, r, sigma, isCall, delta, gamma, vega, theta, rho)
	if err != nil {
		t.Fatalf("GreeksBatch failed: %v", err)
	}

	if delta[0] <= 0.63 || delta[0] >= 0.64 {
		t.Errorf("Expected delta in (0.63, 0.64), got %f", delta[0])
	}
	if gamma[0] <= 0.018 || gamma[0] >= 0.019 {
		t.Errorf("Expected gamma in (0.018, 0.019), got %f", gamma[0])
	}
	if vega[0] <= 37.0 || vega[0] >= 38.0 {
		t.Errorf("Expected vega in (37, 38), got %f", vega[0])
	}
	if theta[0] >= 0.0 {
		t.Errorf("Expected theta < 0, got %f", theta[0])
	}
	if rho[0] <= 0.0 {
		t.Errorf("Expected rho > 0, got %f", rho[0])
	}
}

func TestGreeksBatch_PutBasic(t *testing.T) {
	S := []float64{100.0}
	K := []float64{100.0}
	T := []float64{1.0}
	r := []float64{0.05}
	sigma := []float64{0.2}
	isCall := []bool{false}
	delta := make([]float64, 1)
	gamma := make([]float64, 1)
	vega := make([]float64, 1)
	theta := make([]float64, 1)
	rho := make([]float64, 1)

	err := GreeksBatch(S, K, T, r, sigma, isCall, delta, gamma, vega, theta, rho)
	if err != nil {
		t.Fatalf("GreeksBatch failed: %v", err)
	}

	if delta[0] >= -0.36 || delta[0] <= -0.37 {
		t.Errorf("Expected delta in (-0.37, -0.36), got %f", delta[0])
	}
	if gamma[0] <= 0.018 || gamma[0] >= 0.019 {
		t.Errorf("Expected gamma in (0.018, 0.019), got %f", gamma[0])
	}
	if vega[0] <= 37.0 || vega[0] >= 38.0 {
		t.Errorf("Expected vega in (37, 38), got %f", vega[0])
	}
	if theta[0] >= 0.0 {
		t.Errorf("Expected theta < 0, got %f", theta[0])
	}
	if rho[0] >= 0.0 {
		t.Errorf("Expected rho < 0, got %f", rho[0])
	}
}

func TestGreeksBatch_Mixed(t *testing.T) {
	S := []float64{100.0, 110.0, 90.0, 100.0}
	K := []float64{100.0, 100.0, 100.0, 100.0}
	T := []float64{1.0, 0.5, 1.0, 0.25}
	r := []float64{0.05, 0.05, 0.05, 0.05}
	sigma := []float64{0.2, 0.25, 0.3, 0.15}
	isCall := []bool{true, true, false, false}
	delta := make([]float64, 4)
	gamma := make([]float64, 4)
	vega := make([]float64, 4)
	theta := make([]float64, 4)
	rho := make([]float64, 4)

	err := GreeksBatch(S, K, T, r, sigma, isCall, delta, gamma, vega, theta, rho)
	if err != nil {
		t.Fatalf("GreeksBatch failed: %v", err)
	}

	if delta[0] <= 0.63 || delta[0] >= 0.64 {
		t.Errorf("Expected delta[0] in (0.63, 0.64), got %f", delta[0])
	}
	if delta[1] <= 0.7 {
		t.Errorf("Expected delta[1] > 0.7, got %f", delta[1])
	}
	if delta[2] >= 0.0 {
		t.Errorf("Expected delta[2] < 0, got %f", delta[2])
	}
	if delta[3] >= 0.0 {
		t.Errorf("Expected delta[3] < 0, got %f", delta[3])
	}

	for i := 0; i < 4; i++ {
		if gamma[i] <= 0.0 {
			t.Errorf("Expected gamma[%d] > 0, got %f", i, gamma[i])
		}
		if vega[i] <= 0.0 {
			t.Errorf("Expected vega[%d] > 0, got %f", i, vega[i])
		}
		if theta[i] >= 0.0 {
			t.Errorf("Expected theta[%d] < 0, got %f", i, theta[i])
		}
	}

	if rho[0] <= 0.0 {
		t.Errorf("Expected rho[0] > 0, got %f", rho[0])
	}
	if rho[1] <= 0.0 {
		t.Errorf("Expected rho[1] > 0, got %f", rho[1])
	}
	if rho[2] >= 0.0 {
		t.Errorf("Expected rho[2] < 0, got %f", rho[2])
	}
	if rho[3] >= 0.0 {
		t.Errorf("Expected rho[3] < 0, got %f", rho[3])
	}
}

func TestGreeksBatch_PartialOutputs(t *testing.T) {
	S := []float64{100.0}
	K := []float64{100.0}
	T := []float64{1.0}
	r := []float64{0.05}
	sigma := []float64{0.2}
	isCall := []bool{true}
	delta := make([]float64, 1)
	vega := make([]float64, 1)

	err := GreeksBatch(S, K, T, r, sigma, isCall, delta, nil, vega, nil, nil)
	if err != nil {
		t.Fatalf("GreeksBatch failed: %v", err)
	}

	if delta[0] <= 0.63 || delta[0] >= 0.64 {
		t.Errorf("Expected delta in (0.63, 0.64), got %f", delta[0])
	}
	if vega[0] <= 37.0 || vega[0] >= 38.0 {
		t.Errorf("Expected vega in (37, 38), got %f", vega[0])
	}
}

func TestGreeksBatch_EmptyInput(t *testing.T) {
	S := []float64{}
	K := []float64{}
	T := []float64{}
	r := []float64{}
	sigma := []float64{}
	isCall := []bool{}
	delta := []float64{}

	err := GreeksBatch(S, K, T, r, sigma, isCall, delta, nil, nil, nil, nil)
	if err == nil {
		t.Error("Expected error for empty input")
	}
}

func TestGreeksBatch_LengthMismatch(t *testing.T) {
	S := []float64{100.0, 110.0}
	K := []float64{100.0}
	T := []float64{1.0}
	r := []float64{0.05}
	sigma := []float64{0.2}
	isCall := []bool{true}
	delta := make([]float64, 1)

	err := GreeksBatch(S, K, T, r, sigma, isCall, delta, nil, nil, nil, nil)
	if err == nil {
		t.Error("Expected error for length mismatch")
	}
}

func TestGreeksBatch_AllNilOutputs(t *testing.T) {
	S := []float64{100.0}
	K := []float64{100.0}
	T := []float64{1.0}
	r := []float64{0.05}
	sigma := []float64{0.2}
	isCall := []bool{true}

	err := GreeksBatch(S, K, T, r, sigma, isCall, nil, nil, nil, nil, nil)
	if err == nil {
		t.Error("Expected error for all nil outputs")
	}
}

func TestGreeksBatch_InvalidSpot(t *testing.T) {
	S := []float64{0.0}
	K := []float64{100.0}
	T := []float64{1.0}
	r := []float64{0.05}
	sigma := []float64{0.2}
	isCall := []bool{true}
	delta := make([]float64, 1)

	err := GreeksBatch(S, K, T, r, sigma, isCall, delta, nil, nil, nil, nil)
	if err == nil {
		t.Error("Expected error for invalid spot price")
	}
}

func TestGreeksBatch_InvalidVolatility(t *testing.T) {
	S := []float64{100.0}
	K := []float64{100.0}
	T := []float64{1.0}
	r := []float64{0.05}
	sigma := []float64{0.0}
	isCall := []bool{true}
	delta := make([]float64, 1)

	err := GreeksBatch(S, K, T, r, sigma, isCall, delta, nil, nil, nil, nil)
	if err == nil {
		t.Error("Expected error for invalid volatility")
	}
}

func TestGreeksBatch_GammaVegaSymmetry(t *testing.T) {
	S := []float64{100.0, 100.0}
	K := []float64{100.0, 100.0}
	T := []float64{1.0, 1.0}
	r := []float64{0.05, 0.05}
	sigma := []float64{0.2, 0.2}
	isCall := []bool{true, false}
	gamma := make([]float64, 2)
	vega := make([]float64, 2)

	err := GreeksBatch(S, K, T, r, sigma, isCall, nil, gamma, vega, nil, nil)
	if err != nil {
		t.Fatalf("GreeksBatch failed: %v", err)
	}

	if math.Abs(gamma[0]-gamma[1]) > 1e-10 {
		t.Errorf("Expected gamma[0] == gamma[1], got %f vs %f", gamma[0], gamma[1])
	}
	if math.Abs(vega[0]-vega[1]) > 1e-10 {
		t.Errorf("Expected vega[0] == vega[1], got %f vs %f", vega[0], vega[1])
	}
}

func TestGreeksBatch_LargeBatch(t *testing.T) {
	n := 1000
	S := make([]float64, n)
	K := make([]float64, n)
	T := make([]float64, n)
	r := make([]float64, n)
	sigma := make([]float64, n)
	isCall := make([]bool, n)
	delta := make([]float64, n)
	gamma := make([]float64, n)
	vega := make([]float64, n)
	theta := make([]float64, n)
	rho := make([]float64, n)

	for i := 0; i < n; i++ {
		S[i] = 100.0 + float64(i%50) - 25.0
		K[i] = 100.0
		T[i] = 0.25 + float64(i%8)*0.125
		r[i] = 0.05
		sigma[i] = 0.2 + float64(i%10)*0.01
		isCall[i] = (i % 2) == 0
	}

	err := GreeksBatch(S, K, T, r, sigma, isCall, delta, gamma, vega, theta, rho)
	if err != nil {
		t.Fatalf("GreeksBatch failed: %v", err)
	}

	for i := 0; i < n; i++ {
		if gamma[i] <= 0.0 {
			t.Errorf("Expected gamma[%d] > 0, got %f", i, gamma[i])
		}
		if vega[i] <= 0.0 {
			t.Errorf("Expected vega[%d] > 0, got %f", i, vega[i])
		}
	}
}

func BenchmarkGreeksBatch_100(b *testing.B) {
	n := 100
	S := make([]float64, n)
	K := make([]float64, n)
	T := make([]float64, n)
	r := make([]float64, n)
	sigma := make([]float64, n)
	isCall := make([]bool, n)
	delta := make([]float64, n)
	gamma := make([]float64, n)
	vega := make([]float64, n)
	theta := make([]float64, n)
	rho := make([]float64, n)

	for i := 0; i < n; i++ {
		S[i] = 100.0
		K[i] = 100.0
		T[i] = 1.0
		r[i] = 0.05
		sigma[i] = 0.2
		isCall[i] = true
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GreeksBatch(S, K, T, r, sigma, isCall, delta, gamma, vega, theta, rho)
	}
}

func BenchmarkGreeksBatch_1000(b *testing.B) {
	n := 1000
	S := make([]float64, n)
	K := make([]float64, n)
	T := make([]float64, n)
	r := make([]float64, n)
	sigma := make([]float64, n)
	isCall := make([]bool, n)
	delta := make([]float64, n)
	gamma := make([]float64, n)
	vega := make([]float64, n)
	theta := make([]float64, n)
	rho := make([]float64, n)

	for i := 0; i < n; i++ {
		S[i] = 100.0
		K[i] = 100.0
		T[i] = 1.0
		r[i] = 0.05
		sigma[i] = 0.2
		isCall[i] = true
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GreeksBatch(S, K, T, r, sigma, isCall, delta, gamma, vega, theta, rho)
	}
}

func BenchmarkGreeksBatch_5000(b *testing.B) {
	n := 5000
	S := make([]float64, n)
	K := make([]float64, n)
	T := make([]float64, n)
	r := make([]float64, n)
	sigma := make([]float64, n)
	isCall := make([]bool, n)
	delta := make([]float64, n)
	gamma := make([]float64, n)
	vega := make([]float64, n)
	theta := make([]float64, n)
	rho := make([]float64, n)

	for i := 0; i < n; i++ {
		S[i] = 100.0
		K[i] = 100.0
		T[i] = 1.0
		r[i] = 0.05
		sigma[i] = 0.2
		isCall[i] = true
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GreeksBatch(S, K, T, r, sigma, isCall, delta, gamma, vega, theta, rho)
	}
}
