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
