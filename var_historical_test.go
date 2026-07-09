package optim

import (
	"testing"
)

func TestVarHistorical_Basic(t *testing.T) {
	returns := []float64{-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06}
	confidence := 0.90

	var_, cvar, err := VarHistorical(returns, confidence)
	if err != nil {
		t.Fatalf("VarHistorical failed: %v", err)
	}

	if var_ >= 0.0 {
		t.Errorf("VaR should be negative (indicates loss): var=%f", var_)
	}

	if cvar >= var_ {
		t.Errorf("CVaR should be less than or equal to VaR: cvar=%f, var=%f", cvar, var_)
	}
}

func TestVarHistorical_95Confidence(t *testing.T) {
	returns := []float64{-0.10, -0.08, -0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05,
		0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15}
	confidence := 0.95

	var_, cvar, err := VarHistorical(returns, confidence)
	if err != nil {
		t.Fatalf("VarHistorical failed: %v", err)
	}

	if var_ >= 0.0 {
		t.Errorf("VaR should be negative")
	}

	if cvar > var_ {
		t.Errorf("CVaR should be less than or equal to VaR")
	}
}

func TestVarHistorical_EmptyInput(t *testing.T) {
	_, _, err := VarHistorical([]float64{}, 0.95)
	if err == nil {
		t.Error("Expected error for empty input")
	}
}

func TestVarHistorical_InvalidConfidence(t *testing.T) {
	returns := []float64{0.01, 0.02, 0.03}

	_, _, err := VarHistorical(returns, 0.0)
	if err == nil {
		t.Error("Expected error for confidence = 0")
	}

	_, _, err = VarHistorical(returns, 1.0)
	if err == nil {
		t.Error("Expected error for confidence = 1")
	}
}

func TestVarHistoricalPortfolio_Basic(t *testing.T) {
	returns := [][]float64{
		{-0.02, 0.01, 0.03, -0.01, 0.02},
		{0.01, -0.01, 0.02, 0.00, 0.01},
		{0.00, 0.02, -0.02, 0.01, -0.01},
	}
	weights := []float64{0.5, 0.3, 0.2}
	confidence := 0.90

	var_, cvar, err := VarHistoricalPortfolio(returns, weights, confidence)
	if err != nil {
		t.Fatalf("VarHistoricalPortfolio failed: %v", err)
	}

	if cvar > var_ {
		t.Errorf("CVaR should be less than or equal to VaR: cvar=%f, var=%f", cvar, var_)
	}
}

func TestVarHistoricalPortfolio_EqualWeights(t *testing.T) {
	returns := [][]float64{
		{0.01, 0.02, -0.01, 0.00},
		{-0.01, 0.01, 0.02, -0.02},
	}
	weights := []float64{0.5, 0.5}
	confidence := 0.75

	var_, cvar, err := VarHistoricalPortfolio(returns, weights, confidence)
	if err != nil {
		t.Fatalf("VarHistoricalPortfolio failed: %v", err)
	}

	if cvar > var_ {
		t.Errorf("CVaR should be less than or equal to VaR")
	}
}

func TestVarHistoricalPortfolio_EmptyInput(t *testing.T) {
	_, _, err := VarHistoricalPortfolio([][]float64{}, []float64{}, 0.95)
	if err == nil {
		t.Error("Expected error for empty input")
	}
}

func TestVarHistoricalPortfolio_WeightMismatch(t *testing.T) {
	returns := [][]float64{
		{0.01, 0.02},
		{-0.01, 0.01},
	}
	weights := []float64{0.5}

	_, _, err := VarHistoricalPortfolio(returns, weights, 0.95)
	if err == nil {
		t.Error("Expected error for weight length mismatch")
	}
}

func TestVarHistoricalPortfolio_UnevenPeriods(t *testing.T) {
	returns := [][]float64{
		{0.01, 0.02, 0.03},
		{-0.01, 0.01},
	}
	weights := []float64{0.5, 0.5}

	_, _, err := VarHistoricalPortfolio(returns, weights, 0.95)
	if err == nil {
		t.Error("Expected error for uneven periods")
	}
}

func TestVarHistoricalBatch_Basic(t *testing.T) {
	returns := [][]float64{
		{-0.02, 0.01, 0.03, -0.01},
		{0.01, -0.01, 0.02, 0.00},
	}
	weightsMatrix := [][]float64{
		{0.6, 0.4},
		{0.5, 0.5},
		{0.7, 0.3},
	}
	confidence := 0.75

	var_, cvar, err := VarHistoricalBatch(returns, weightsMatrix, confidence)
	if err != nil {
		t.Fatalf("VarHistoricalBatch failed: %v", err)
	}

	if len(var_) != 3 {
		t.Errorf("Expected 3 VaR values, got %d", len(var_))
	}

	if len(cvar) != 3 {
		t.Errorf("Expected 3 CVaR values, got %d", len(cvar))
	}

	for i := 0; i < 3; i++ {
		if cvar[i] > var_[i] {
			t.Errorf("CVaR[%d] should be less than or equal to VaR[%d]", i, i)
		}
	}
}

func TestVarHistoricalBatch_EmptyInput(t *testing.T) {
	_, _, err := VarHistoricalBatch([][]float64{}, [][]float64{}, 0.95)
	if err == nil {
		t.Error("Expected error for empty input")
	}
}

func TestVarHistoricalBatch_WeightMismatch(t *testing.T) {
	returns := [][]float64{
		{0.01, 0.02},
		{-0.01, 0.01},
	}
	weightsMatrix := [][]float64{
		{0.5},
	}

	_, _, err := VarHistoricalBatch(returns, weightsMatrix, 0.95)
	if err == nil {
		t.Error("Expected error for weight length mismatch")
	}
}

func BenchmarkVarHistorical_100(b *testing.B) {
	returns := make([]float64, 100)
	for i := 0; i < 100; i++ {
		returns[i] = 0.02 * (float64(i%100) - 50.0) / 50.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VarHistorical(returns, 0.95)
	}
}

func BenchmarkVarHistorical_1000(b *testing.B) {
	returns := make([]float64, 1000)
	for i := 0; i < 1000; i++ {
		returns[i] = 0.02 * (float64(i%100) - 50.0) / 50.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VarHistorical(returns, 0.95)
	}
}

func BenchmarkVarHistoricalPortfolio_500x1000(b *testing.B) {
	m := 500
	n := 1000
	returns := make([][]float64, m)
	for i := 0; i < m; i++ {
		returns[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			returns[i][j] = 0.02 * (float64((i*n+j)%100) - 50.0) / 50.0
		}
	}

	weights := make([]float64, m)
	for i := 0; i < m; i++ {
		weights[i] = 1.0 / float64(m)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VarHistoricalPortfolio(returns, weights, 0.95)
	}
}

func BenchmarkVarHistoricalBatch_10Portfolios(b *testing.B) {
	m := 50
	n := 100
	returns := make([][]float64, m)
	for i := 0; i < m; i++ {
		returns[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			returns[i][j] = 0.02 * (float64((i*n+j)%100) - 50.0) / 50.0
		}
	}

	numPortfolios := 10
	weightsMatrix := make([][]float64, numPortfolios)
	for p := 0; p < numPortfolios; p++ {
		weightsMatrix[p] = make([]float64, m)
		for i := 0; i < m; i++ {
			weightsMatrix[p][i] = 1.0 / float64(m)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VarHistoricalBatch(returns, weightsMatrix, 0.95)
	}
}
