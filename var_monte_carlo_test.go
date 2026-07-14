package optim

import (
	"math"
	"testing"
)

func TestNewVarMonteCarloState(t *testing.T) {
	state, err := NewVarMonteCarloState(10, 1000, 42)
	if err != nil {
		t.Fatalf("NewVarMonteCarloState failed: %v", err)
	}
	defer state.Close()

	if state.dim != 10 {
		t.Errorf("Expected dim=10, got %d", state.dim)
	}
}

func TestNewVarMonteCarloState_Invalid(t *testing.T) {
	_, err := NewVarMonteCarloState(0, 1000, 42)
	if err == nil {
		t.Error("Expected error for dim=0")
	}

	_, err = NewVarMonteCarloState(10, 0, 42)
	if err == nil {
		t.Error("Expected error for nPaths=0")
	}
}

func TestVarMonteCarlo_Basic(t *testing.T) {
	state, err := NewVarMonteCarloState(3, 10000, 42)
	if err != nil {
		t.Fatalf("Failed to create state: %v", err)
	}
	defer state.Close()

	means := []float64{0.001, 0.002, -0.001}
	covMatrix := [][]float64{
		{0.0004, 0.0001, 0.0000},
		{0.0001, 0.0009, 0.0002},
		{0.0000, 0.0002, 0.0006},
	}
	weights := []float64{0.5, 0.3, 0.2}
	confidence := 0.95

	var_, cvar, err := state.VarMonteCarlo(means, covMatrix, weights, confidence)
	if err != nil {
		t.Fatalf("VarMonteCarlo failed: %v", err)
	}

	if cvar > var_ {
		t.Errorf("CVaR should be <= VaR: cvar=%f, var=%f", cvar, var_)
	}
}

func TestVarMonteCarlo_SingleAsset(t *testing.T) {
	state, err := NewVarMonteCarloState(1, 5000, 123)
	if err != nil {
		t.Fatalf("Failed to create state: %v", err)
	}
	defer state.Close()

	means := []float64{0.001}
	covMatrix := [][]float64{{0.0004}}
	weights := []float64{1.0}
	confidence := 0.95

	var_, cvar, err := state.VarMonteCarlo(means, covMatrix, weights, confidence)
	if err != nil {
		t.Fatalf("VarMonteCarlo failed: %v", err)
	}

	if var_ >= means[0] {
		t.Errorf("VaR should be less than mean")
	}

	if cvar >= var_ {
		t.Errorf("CVaR should be less than VaR")
	}
}

func TestVarMonteCarlo_Uncorrelated(t *testing.T) {
	state, err := NewVarMonteCarloState(2, 10000, 456)
	if err != nil {
		t.Fatalf("Failed to create state: %v", err)
	}
	defer state.Close()

	means := []float64{0.0, 0.0}
	covMatrix := [][]float64{
		{1.0, 0.0},
		{0.0, 1.0},
	}
	weights := []float64{0.5, 0.5}
	confidence := 0.95

	var_, cvar, err := state.VarMonteCarlo(means, covMatrix, weights, confidence)
	if err != nil {
		t.Fatalf("VarMonteCarlo failed: %v", err)
	}

	if cvar > var_ {
		t.Errorf("CVaR should be <= VaR")
	}
}

func TestVarMonteCarlo_DimensionMismatch(t *testing.T) {
	state, err := NewVarMonteCarloState(2, 1000, 42)
	if err != nil {
		t.Fatalf("Failed to create state: %v", err)
	}
	defer state.Close()

	means := []float64{0.0, 0.0, 0.0}
	covMatrix := [][]float64{
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
	}
	weights := []float64{0.33, 0.33, 0.34}

	_, _, err = state.VarMonteCarlo(means, covMatrix, weights, 0.95)
	if err == nil {
		t.Error("Expected error for dimension mismatch")
	}
}

func TestVarMonteCarlo_InvalidConfidence(t *testing.T) {
	state, err := NewVarMonteCarloState(2, 1000, 42)
	if err != nil {
		t.Fatalf("Failed to create state: %v", err)
	}
	defer state.Close()

	means := []float64{0.0, 0.0}
	covMatrix := [][]float64{
		{1.0, 0.0},
		{0.0, 1.0},
	}
	weights := []float64{0.5, 0.5}

	_, _, err = state.VarMonteCarlo(means, covMatrix, weights, 0.0)
	if err == nil {
		t.Error("Expected error for confidence=0")
	}

	_, _, err = state.VarMonteCarlo(means, covMatrix, weights, 1.0)
	if err == nil {
		t.Error("Expected error for confidence=1")
	}
}

func TestVarMonteCarlo_NotPositiveDefinite(t *testing.T) {
	state, err := NewVarMonteCarloState(2, 1000, 42)
	if err != nil {
		t.Fatalf("Failed to create state: %v", err)
	}
	defer state.Close()

	means := []float64{0.0, 0.0}
	covMatrix := [][]float64{
		{1.0, 2.0},
		{2.0, 1.0},
	}
	weights := []float64{0.5, 0.5}

	_, _, err = state.VarMonteCarlo(means, covMatrix, weights, 0.95)
	if err == nil {
		t.Error("Expected error for non-positive-definite matrix")
	}
}

func TestVarMonteCarloFromHistory_Basic(t *testing.T) {
	state, err := NewVarMonteCarloState(2, 5000, 789)
	if err != nil {
		t.Fatalf("Failed to create state: %v", err)
	}
	defer state.Close()

	nPeriods := 100
	returns := make([][]float64, 2)
	for i := 0; i < 2; i++ {
		returns[i] = make([]float64, nPeriods)
		for t := 0; t < nPeriods; t++ {
			if i == 0 {
				returns[i][t] = 0.02 * (float64(t%20) - 10.0) / 10.0
			} else {
				returns[i][t] = 0.015 * (float64((t+5)%20) - 10.0) / 10.0
			}
		}
	}

	weights := []float64{0.6, 0.4}
	confidence := 0.95

	var_, cvar, err := state.VarMonteCarloFromHistory(returns, weights, confidence)
	if err != nil {
		t.Fatalf("VarMonteCarloFromHistory failed: %v", err)
	}

	if cvar > var_ {
		t.Errorf("CVaR should be <= VaR")
	}
}

func TestVarMonteCarloFromHistory_InsufficientData(t *testing.T) {
	state, err := NewVarMonteCarloState(10, 1000, 42)
	if err != nil {
		t.Fatalf("Failed to create state: %v", err)
	}
	defer state.Close()

	returns := make([][]float64, 10)
	for i := 0; i < 10; i++ {
		returns[i] = make([]float64, 5)
	}

	weights := make([]float64, 10)
	for i := 0; i < 10; i++ {
		weights[i] = 0.1
	}

	_, _, err = state.VarMonteCarloFromHistory(returns, weights, 0.95)
	if err == nil {
		t.Error("Expected error for insufficient historical data")
	}
}

func TestVarMonteCarloFromHistory_UnevenPeriods(t *testing.T) {
	state, err := NewVarMonteCarloState(2, 1000, 42)
	if err != nil {
		t.Fatalf("Failed to create state: %v", err)
	}
	defer state.Close()

	returns := [][]float64{
		{0.01, 0.02, 0.03},
		{-0.01, 0.01},
	}
	weights := []float64{0.5, 0.5}

	_, _, err = state.VarMonteCarloFromHistory(returns, weights, 0.95)
	if err == nil {
		t.Error("Expected error for uneven periods")
	}
}

func TestVarMonteCarlo_Reproducibility(t *testing.T) {
	state1, _ := NewVarMonteCarloState(3, 1000, 12345)
	defer state1.Close()

	state2, _ := NewVarMonteCarloState(3, 1000, 12345)
	defer state2.Close()

	means := []float64{0.001, 0.002, -0.001}
	covMatrix := [][]float64{
		{0.0004, 0.0001, 0.0000},
		{0.0001, 0.0009, 0.0002},
		{0.0000, 0.0002, 0.0006},
	}
	weights := []float64{0.5, 0.3, 0.2}

	var1, cvar1, _ := state1.VarMonteCarlo(means, covMatrix, weights, 0.95)
	var2, cvar2, _ := state2.VarMonteCarlo(means, covMatrix, weights, 0.95)

	if math.Abs(var1-var2) > 1e-10 {
		t.Errorf("VaR not reproducible: %f vs %f", var1, var2)
	}

	if math.Abs(cvar1-cvar2) > 1e-10 {
		t.Errorf("CVaR not reproducible: %f vs %f", cvar1, cvar2)
	}
}

func TestVarMonteCarloState_Close(t *testing.T) {
	state, err := NewVarMonteCarloState(2, 1000, 42)
	if err != nil {
		t.Fatalf("Failed to create state: %v", err)
	}

	state.Close()

	means := []float64{0.0, 0.0}
	covMatrix := [][]float64{{1.0, 0.0}, {0.0, 1.0}}
	weights := []float64{0.5, 0.5}

	_, _, err = state.VarMonteCarlo(means, covMatrix, weights, 0.95)
	if err == nil {
		t.Error("Expected error when using closed state")
	}
}

func BenchmarkVarMonteCarlo_10x1000(b *testing.B) {
	state, _ := NewVarMonteCarloState(10, 1000, 42)
	defer state.Close()

	means := make([]float64, 10)
	covMatrix := make([][]float64, 10)
	for i := 0; i < 10; i++ {
		means[i] = 0.001
		covMatrix[i] = make([]float64, 10)
		for j := 0; j < 10; j++ {
			if i == j {
				covMatrix[i][j] = 0.0004
			} else {
				covMatrix[i][j] = 0.0001
			}
		}
	}
	weights := make([]float64, 10)
	for i := 0; i < 10; i++ {
		weights[i] = 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		state.VarMonteCarlo(means, covMatrix, weights, 0.95)
	}
}

func BenchmarkVarMonteCarlo_50x5000(b *testing.B) {
	state, _ := NewVarMonteCarloState(50, 5000, 42)
	defer state.Close()

	means := make([]float64, 50)
	covMatrix := make([][]float64, 50)
	for i := 0; i < 50; i++ {
		means[i] = 0.001
		covMatrix[i] = make([]float64, 50)
		for j := 0; j < 50; j++ {
			if i == j {
				covMatrix[i][j] = 0.0004
			} else {
				covMatrix[i][j] = 0.0001
			}
		}
	}
	weights := make([]float64, 50)
	for i := 0; i < 50; i++ {
		weights[i] = 0.02
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		state.VarMonteCarlo(means, covMatrix, weights, 0.95)
	}
}

func TestSetCovariance_Basic(t *testing.T) {
	state, err := NewVarMonteCarloState(3, 5000, 42)
	if err != nil {
		t.Fatalf("Failed to create state: %v", err)
	}
	defer state.Close()

	covMatrix := [][]float64{
		{0.0004, 0.0001, 0.0000},
		{0.0001, 0.0009, 0.0002},
		{0.0000, 0.0002, 0.0006},
	}

	err = state.SetCovariance(covMatrix)
	if err != nil {
		t.Fatalf("SetCovariance failed: %v", err)
	}
}

func TestSetCovariance_NotPositiveDefinite(t *testing.T) {
	state, err := NewVarMonteCarloState(2, 1000, 42)
	if err != nil {
		t.Fatalf("Failed to create state: %v", err)
	}
	defer state.Close()

	// Non-positive definite matrix
	covMatrix := [][]float64{
		{1.0, 2.0},
		{2.0, 1.0},
	}

	err = state.SetCovariance(covMatrix)
	if err == nil {
		t.Error("Expected error for non-positive definite matrix")
	}
}

func TestSetCovariance_InvalidDimensions(t *testing.T) {
	state, err := NewVarMonteCarloState(3, 1000, 42)
	if err != nil {
		t.Fatalf("Failed to create state: %v", err)
	}
	defer state.Close()

	// Wrong dimensions
	covMatrix := [][]float64{
		{0.0004, 0.0001},
		{0.0001, 0.0009},
	}

	err = state.SetCovariance(covMatrix)
	if err == nil {
		t.Error("Expected error for dimension mismatch")
	}
}

func TestVarMonteCarloCached_Basic(t *testing.T) {
	state, err := NewVarMonteCarloState(3, 10000, 42)
	if err != nil {
		t.Fatalf("Failed to create state: %v", err)
	}
	defer state.Close()

	covMatrix := [][]float64{
		{0.0004, 0.0001, 0.0000},
		{0.0001, 0.0009, 0.0002},
		{0.0000, 0.0002, 0.0006},
	}

	err = state.SetCovariance(covMatrix)
	if err != nil {
		t.Fatalf("SetCovariance failed: %v", err)
	}

	means := []float64{0.001, 0.002, -0.001}
	weights := []float64{0.5, 0.3, 0.2}
	confidence := 0.95

	var_, cvar, err := state.VarMonteCarloCached(means, weights, confidence)
	if err != nil {
		t.Fatalf("VarMonteCarloCached failed: %v", err)
	}

	if cvar >= var_ {
		t.Errorf("CVaR should be less than VaR: cvar=%f, var=%f", cvar, var_)
	}
}

func TestVarMonteCarloCached_WithoutSetCovariance(t *testing.T) {
	state, err := NewVarMonteCarloState(3, 1000, 42)
	if err != nil {
		t.Fatalf("Failed to create state: %v", err)
	}
	defer state.Close()

	means := []float64{0.001, 0.002, -0.001}
	weights := []float64{0.5, 0.3, 0.2}

	_, _, err = state.VarMonteCarloCached(means, weights, 0.95)
	if err == nil {
		t.Error("Expected error when calling VarMonteCarloCached without SetCovariance")
	}
}

func TestVarMonteCarloCached_MultipleRuns(t *testing.T) {
	state, err := NewVarMonteCarloState(2, 5000, 42)
	if err != nil {
		t.Fatalf("Failed to create state: %v", err)
	}
	defer state.Close()

	covMatrix := [][]float64{
		{0.0004, 0.0001},
		{0.0001, 0.0009},
	}

	err = state.SetCovariance(covMatrix)
	if err != nil {
		t.Fatalf("SetCovariance failed: %v", err)
	}

	// Run multiple times with different means/weights
	testCases := []struct {
		means   []float64
		weights []float64
	}{
		{[]float64{0.001, 0.002}, []float64{0.6, 0.4}},
		{[]float64{0.002, -0.001}, []float64{0.7, 0.3}},
		{[]float64{-0.001, 0.003}, []float64{0.5, 0.5}},
	}

	for i, tc := range testCases {
		var_, cvar, err := state.VarMonteCarloCached(tc.means, tc.weights, 0.95)
		if err != nil {
			t.Fatalf("VarMonteCarloCached failed for case %d: %v", i, err)
		}

		if cvar >= var_ {
			t.Errorf("Case %d: CVaR should be less than VaR", i)
		}
	}
}

func TestVarMonteCarloSimple_Basic(t *testing.T) {
	// 2 assets, 50 periods
	dim := 2
	nPeriods := 50
	returns := make([]float64, dim*nPeriods)

	// Generate synthetic returns
	for i := 0; i < dim; i++ {
		for j := 0; j < nPeriods; j++ {
			returns[i*nPeriods+j] = 0.01 * float64((i+j)%10-5)
		}
	}

	weights := []float64{0.6, 0.4}
	nPaths := 5000
	confidence := 0.95
	seed := uint64(42)

	var_, cvar, err := VarMonteCarloSimple(returns, weights, dim, nPeriods, nPaths, confidence, seed)
	if err != nil {
		t.Fatalf("VarMonteCarloSimple failed: %v", err)
	}

	if cvar >= var_ {
		t.Errorf("CVaR should be less than VaR: cvar=%f, var=%f", cvar, var_)
	}

	if math.IsNaN(var_) || math.IsInf(var_, 0) {
		t.Errorf("VaR is not finite: %f", var_)
	}
}

func TestVarMonteCarloSimple_InsufficientPeriods(t *testing.T) {
	dim := 3
	nPeriods := 3 // Need at least dim + 1 = 4
	returns := make([]float64, dim*nPeriods)
	weights := []float64{0.5, 0.3, 0.2}

	_, _, err := VarMonteCarloSimple(returns, weights, dim, nPeriods, 1000, 0.95, 42)
	if err == nil {
		t.Error("Expected error for insufficient periods")
	}
}

func TestVarMonteCarloSimple_InvalidDimensions(t *testing.T) {
	dim := 2
	nPeriods := 10
	returns := make([]float64, dim*nPeriods-1) // Wrong size
	weights := []float64{0.5, 0.5}

	_, _, err := VarMonteCarloSimple(returns, weights, dim, nPeriods, 1000, 0.95, 42)
	if err == nil {
		t.Error("Expected error for wrong returns length")
	}
}

func TestVarMonteCarloSimple_Reproducibility(t *testing.T) {
	dim := 2
	nPeriods := 30
	returns := make([]float64, dim*nPeriods)

	for i := 0; i < dim*nPeriods; i++ {
		returns[i] = 0.01 * float64(i%10-5)
	}

	weights := []float64{0.5, 0.5}
	seed := uint64(123)

	var1, cvar1, err := VarMonteCarloSimple(returns, weights, dim, nPeriods, 5000, 0.95, seed)
	if err != nil {
		t.Fatalf("First call failed: %v", err)
	}

	var2, cvar2, err := VarMonteCarloSimple(returns, weights, dim, nPeriods, 5000, 0.95, seed)
	if err != nil {
		t.Fatalf("Second call failed: %v", err)
	}

	// With same seed, results should be identical
	if math.Abs(var1-var2) > 1e-10 {
		t.Errorf("VaR not reproducible: %f vs %f", var1, var2)
	}

	if math.Abs(cvar1-cvar2) > 1e-10 {
		t.Errorf("CVaR not reproducible: %f vs %f", cvar1, cvar2)
	}
}

func TestVarMonteCarloBatch_Basic(t *testing.T) {
	state, err := NewVarMonteCarloState(2, 5000, 42)
	if err != nil {
		t.Fatalf("Failed to create state: %v", err)
	}
	defer state.Close()

	// 2 assets, 30 periods
	dim := 2
	nPeriods := 30
	returns := make([][]float64, dim)
	for i := 0; i < dim; i++ {
		returns[i] = make([]float64, nPeriods)
		for j := 0; j < nPeriods; j++ {
			returns[i][j] = 0.01 * float64((i+j)%10-5)
		}
	}

	// 3 portfolios
	weightsMatrix := [][]float64{
		{0.6, 0.4},
		{0.7, 0.3},
		{0.5, 0.5},
	}

	confidence := 0.95

	var_, cvar, err := state.VarMonteCarloBatch(returns, weightsMatrix, confidence)
	if err != nil {
		t.Fatalf("VarMonteCarloBatch failed: %v", err)
	}

	if len(var_) != 3 {
		t.Errorf("Expected 3 VaR values, got %d", len(var_))
	}

	if len(cvar) != 3 {
		t.Errorf("Expected 3 CVaR values, got %d", len(cvar))
	}

	for i := 0; i < 3; i++ {
		if cvar[i] >= var_[i] {
			t.Errorf("Portfolio %d: CVaR should be less than VaR", i)
		}
	}
}

func TestVarMonteCarloBatch_InvalidDimensions(t *testing.T) {
	state, err := NewVarMonteCarloState(2, 1000, 42)
	if err != nil {
		t.Fatalf("Failed to create state: %v", err)
	}
	defer state.Close()

	returns := [][]float64{
		{0.01, 0.02, 0.03},
		{-0.01, 0.02, -0.03},
	}

	// Wrong weights dimension
	weightsMatrix := [][]float64{
		{0.6, 0.4, 0.0}, // 3 elements instead of 2
	}

	_, _, err = state.VarMonteCarloBatch(returns, weightsMatrix, 0.95)
	if err == nil {
		t.Error("Expected error for wrong weights dimension")
	}
}

func BenchmarkSetCovariance_10Assets(b *testing.B) {
	state, err := NewVarMonteCarloState(10, 5000, 42)
	if err != nil {
		b.Fatalf("Failed to create state: %v", err)
	}
	defer state.Close()

	covMatrix := make([][]float64, 10)
	for i := 0; i < 10; i++ {
		covMatrix[i] = make([]float64, 10)
		for j := 0; j < 10; j++ {
			if i == j {
				covMatrix[i][j] = 0.0004
			} else {
				covMatrix[i][j] = 0.0001
			}
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		state.SetCovariance(covMatrix)
	}
}

func BenchmarkVarMonteCarloCached_vs_Regular(b *testing.B) {
	state, err := NewVarMonteCarloState(5, 10000, 42)
	if err != nil {
		b.Fatalf("Failed to create state: %v", err)
	}
	defer state.Close()

	means := []float64{0.001, 0.002, -0.001, 0.0015, -0.0005}
	covMatrix := make([][]float64, 5)
	for i := 0; i < 5; i++ {
		covMatrix[i] = make([]float64, 5)
		for j := 0; j < 5; j++ {
			if i == j {
				covMatrix[i][j] = 0.0004
			} else {
				covMatrix[i][j] = 0.0001
			}
		}
	}
	weights := []float64{0.2, 0.2, 0.2, 0.2, 0.2}

	b.Run("Regular", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			state.VarMonteCarlo(means, covMatrix, weights, 0.95)
		}
	})

	state.SetCovariance(covMatrix)

	b.Run("Cached", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			state.VarMonteCarloCached(means, weights, 0.95)
		}
	})
}

func BenchmarkVarMonteCarloSimple_10Assets_100Periods(b *testing.B) {
	dim := 10
	nPeriods := 100
	returns := make([]float64, dim*nPeriods)
	weights := make([]float64, dim)

	for i := 0; i < dim; i++ {
		weights[i] = 1.0 / float64(dim)
		for j := 0; j < nPeriods; j++ {
			returns[i*nPeriods+j] = 0.01 * float64((i+j)%10-5)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VarMonteCarloSimple(returns, weights, dim, nPeriods, 5000, 0.95, 42)
	}
}

func BenchmarkVarMonteCarloBatch_5Portfolios(b *testing.B) {
	state, err := NewVarMonteCarloState(5, 5000, 42)
	if err != nil {
		b.Fatalf("Failed to create state: %v", err)
	}
	defer state.Close()

	dim := 5
	nPeriods := 50
	returns := make([][]float64, dim)
	for i := 0; i < dim; i++ {
		returns[i] = make([]float64, nPeriods)
		for j := 0; j < nPeriods; j++ {
			returns[i][j] = 0.01 * float64((i+j)%10-5)
		}
	}

	weightsMatrix := make([][]float64, 5)
	for i := 0; i < 5; i++ {
		weightsMatrix[i] = make([]float64, dim)
		for j := 0; j < dim; j++ {
			weightsMatrix[i][j] = 1.0 / float64(dim)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		state.VarMonteCarloBatch(returns, weightsMatrix, 0.95)
	}
}
