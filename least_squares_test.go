package optim

import (
	"math"
	"testing"
)

func TestLeastSquaresBasic(t *testing.T) {
	X := []float64{
		1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 2.0, 3.0, 4.0, 5.0,
	}
	y := []float64{2.0, 4.0, 5.0, 4.0, 5.0}

	beta, err := LeastSquares(X, y, 5, 2, false)
	if err != nil {
		t.Fatalf("LeastSquares failed: %v", err)
	}

	if len(beta) != 2 {
		t.Errorf("Expected 2 coefficients, got %d", len(beta))
	}

	if math.Abs(beta[0]-2.2) > 0.1 {
		t.Errorf("Intercept = %f, want approximately 2.2", beta[0])
	}
	if math.Abs(beta[1]-0.6) > 0.1 {
		t.Errorf("Slope = %f, want approximately 0.6", beta[1])
	}
}

func TestLeastSquaresPerfectFit(t *testing.T) {
	X := []float64{
		1.0, 1.0, 1.0,
		1.0, 2.0, 3.0,
	}
	y := []float64{3.0, 5.0, 7.0}

	beta, err := LeastSquares(X, y, 3, 2, false)
	if err != nil {
		t.Fatalf("LeastSquares failed: %v", err)
	}

	if math.Abs(beta[0]-1.0) > 1e-10 {
		t.Errorf("Intercept = %f, want 1.0", beta[0])
	}
	if math.Abs(beta[1]-2.0) > 1e-10 {
		t.Errorf("Slope = %f, want 2.0", beta[1])
	}
}

func TestLeastSquaresInvalidDimensions(t *testing.T) {
	X := []float64{1.0, 2.0, 3.0, 4.0}
	y := []float64{1.0, 2.0}

	_, err := LeastSquares(X, y, 0, 2, false)
	if err != ErrInvalidDimensions {
		t.Errorf("Expected ErrInvalidDimensions for n=0, got %v", err)
	}

	_, err = LeastSquares(X, y, 2, 0, false)
	if err != ErrInvalidDimensions {
		t.Errorf("Expected ErrInvalidDimensions for p=0, got %v", err)
	}

	_, err = LeastSquares(X, y, 2, 3, false)
	if err != ErrInvalidDimensions {
		t.Errorf("Expected ErrInvalidDimensions for n<p, got %v", err)
	}
}

func TestLeastSquaresRankDeficient(t *testing.T) {
	X := []float64{
		1.0, 1.0, 1.0,
		1.0, 2.0, 3.0,
		2.0, 4.0, 6.0,
	}
	y := []float64{1.0, 2.0, 3.0}

	_, err := LeastSquares(X, y, 3, 3, false)
	if err != ErrRankDeficient {
		t.Errorf("Expected ErrRankDeficient for collinear matrix, got %v", err)
	}
}

func TestLeastSquaresBatch(t *testing.T) {
	X := []float64{
		1.0, 1.0, 1.0,
		1.0, 2.0, 3.0,

		1.0, 1.0, 1.0,
		2.0, 3.0, 4.0,
	}
	y := []float64{
		3.0, 5.0, 7.0,
		4.0, 7.0, 10.0,
	}

	beta, count, err := LeastSquaresBatch(X, y, 3, 2, 2, false)
	if err != nil {
		t.Fatalf("LeastSquaresBatch failed: %v", err)
	}

	if count != 2 {
		t.Errorf("Expected 2 successful regressions, got %d", count)
	}

	if math.Abs(beta[0]-1.0) > 1e-10 {
		t.Errorf("First intercept = %f, want 1.0", beta[0])
	}
	if math.Abs(beta[1]-2.0) > 1e-10 {
		t.Errorf("First slope = %f, want 2.0", beta[1])
	}
	if math.Abs(beta[2]-(-2.0)) > 1e-10 {
		t.Errorf("Second intercept = %f, want -2.0", beta[2])
	}
	if math.Abs(beta[3]-3.0) > 1e-10 {
		t.Errorf("Second slope = %f, want 3.0", beta[3])
	}
}

func TestLeastSquaresExt(t *testing.T) {
	X := []float64{
		1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 2.0, 3.0, 4.0, 5.0,
	}
	y := []float64{2.0, 4.0, 5.0, 4.0, 5.0}

	result, err := LeastSquaresExt(X, y, 5, 2, false, true)
	if err != nil {
		t.Fatalf("LeastSquaresExt failed: %v", err)
	}

	if math.Abs(result.Beta[0]-2.2) > 0.1 {
		t.Errorf("Intercept = %f, want approximately 2.2", result.Beta[0])
	}
	if math.Abs(result.Beta[1]-0.6) > 0.1 {
		t.Errorf("Slope = %f, want approximately 0.6", result.Beta[1])
	}

	if result.RSquared < 0.0 || result.RSquared > 1.0 {
		t.Errorf("RSquared = %f, want in [0, 1]", result.RSquared)
	}

	if result.StdError < 0.0 {
		t.Errorf("StdError = %f, want non-negative", result.StdError)
	}

	if len(result.Residuals) != 5 {
		t.Errorf("Expected 5 residuals, got %d", len(result.Residuals))
	}
}

func TestLeastSquaresExtPerfectFit(t *testing.T) {
	X := []float64{
		1.0, 1.0, 1.0,
		1.0, 2.0, 3.0,
	}
	y := []float64{3.0, 5.0, 7.0}

	result, err := LeastSquaresExt(X, y, 3, 2, false, true)
	if err != nil {
		t.Fatalf("LeastSquaresExt failed: %v", err)
	}

	if math.Abs(result.RSquared-1.0) > 1e-10 {
		t.Errorf("RSquared = %f, want 1.0 for perfect fit", result.RSquared)
	}

	if result.StdError > 1e-10 {
		t.Errorf("StdError = %f, want near zero for perfect fit", result.StdError)
	}

	for i, r := range result.Residuals {
		if math.Abs(r) > 1e-10 {
			t.Errorf("Residual[%d] = %f, want near zero", i, r)
		}
	}
}

func TestLeastSquaresExtNoResiduals(t *testing.T) {
	X := []float64{
		1.0, 1.0, 1.0,
		1.0, 2.0, 3.0,
	}
	y := []float64{3.0, 5.0, 7.0}

	result, err := LeastSquaresExt(X, y, 3, 2, false, false)
	if err != nil {
		t.Fatalf("LeastSquaresExt failed: %v", err)
	}

	if result.Residuals != nil {
		t.Errorf("Expected nil residuals when computeResiduals=false")
	}

	if math.Abs(result.Beta[0]-1.0) > 1e-10 {
		t.Errorf("Intercept = %f, want 1.0", result.Beta[0])
	}
	if math.Abs(result.Beta[1]-2.0) > 1e-10 {
		t.Errorf("Slope = %f, want 2.0", result.Beta[1])
	}
}

func BenchmarkLeastSquaresSmall(b *testing.B) {
	n, p := 100, 5
	X := make([]float64, n*p)
	y := make([]float64, n)

	for i := 0; i < n; i++ {
		for j := 0; j < p; j++ {
			X[j*n+i] = float64(i*p + j + 1)
		}
		y[i] = float64(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Xcopy := make([]float64, len(X))
		copy(Xcopy, X)
		_, _ = LeastSquares(Xcopy, y, n, p, false)
	}
}

func BenchmarkLeastSquaresMedium(b *testing.B) {
	n, p := 1000, 10
	X := make([]float64, n*p)
	y := make([]float64, n)

	for i := 0; i < n; i++ {
		for j := 0; j < p; j++ {
			X[j*n+i] = float64(i*p + j + 1)
		}
		y[i] = float64(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Xcopy := make([]float64, len(X))
		copy(Xcopy, X)
		_, _ = LeastSquares(Xcopy, y, n, p, false)
	}
}

func BenchmarkLeastSquaresBatch(b *testing.B) {
	n, p, batchSize := 50, 3, 100
	X := make([]float64, batchSize*n*p)
	y := make([]float64, batchSize*n)

	for batch := 0; batch < batchSize; batch++ {
		for i := 0; i < n; i++ {
			for j := 0; j < p; j++ {
				X[batch*n*p+j*n+i] = float64(batch*n*p + i*p + j + 1)
			}
			y[batch*n+i] = float64(batch*n + i + 1)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Xcopy := make([]float64, len(X))
		copy(Xcopy, X)
		_, _, _ = LeastSquaresBatch(Xcopy, y, n, p, batchSize, false)
	}
}
