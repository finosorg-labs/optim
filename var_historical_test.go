package optim

import (
	"math"
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

func TestVarHistoricalBatchFromReturns_Basic(t *testing.T) {
	// 3 assets, each with 30 returns
	returnsMatrix := [][]float64{
		{0.02, -0.01, 0.03, 0.01, -0.02, 0.015, -0.005, 0.025, 0.005, -0.015,
			0.01, 0.02, -0.01, 0.03, -0.02, 0.015, -0.005, 0.025, 0.005, -0.015,
			0.02, -0.01, 0.03, 0.01, -0.02, 0.015, -0.005, 0.025, 0.005, -0.015},
		{-0.01, 0.03, 0.01, -0.02, 0.02, -0.015, 0.025, -0.005, 0.015, 0.005,
			-0.01, 0.03, 0.01, -0.02, 0.02, -0.015, 0.025, -0.005, 0.015, 0.005,
			-0.01, 0.03, 0.01, -0.02, 0.02, -0.015, 0.025, -0.005, 0.015, 0.005},
		{0.015, -0.01, 0.02, 0.005, -0.015, 0.01, -0.005, 0.02, 0.0, -0.01,
			0.015, -0.01, 0.02, 0.005, -0.015, 0.01, -0.005, 0.02, 0.0, -0.01,
			0.015, -0.01, 0.02, 0.005, -0.015, 0.01, -0.005, 0.02, 0.0, -0.01},
	}
	confidence := 0.95

	var_, cvar, err := VarHistoricalBatchFromReturns(returnsMatrix, confidence)
	if err != nil {
		t.Fatalf("VarHistoricalBatchFromReturns failed: %v", err)
	}

	if len(var_) != 3 {
		t.Errorf("Expected 3 VaR values, got %d", len(var_))
	}

	if len(cvar) != 3 {
		t.Errorf("Expected 3 CVaR values, got %d", len(cvar))
	}

	for i := 0; i < 3; i++ {
		if cvar[i] > var_[i] {
			t.Errorf("Asset %d: CVaR should be <= VaR: cvar=%f, var=%f", i, cvar[i], var_[i])
		}

		if math.IsNaN(var_[i]) || math.IsInf(var_[i], 0) {
			t.Errorf("Asset %d: VaR is not finite: %f", i, var_[i])
		}

		if math.IsNaN(cvar[i]) || math.IsInf(cvar[i], 0) {
			t.Errorf("Asset %d: CVaR is not finite: %f", i, cvar[i])
		}
	}
}

func TestVarHistoricalBatchFromReturns_SingleAsset(t *testing.T) {
	returnsMatrix := [][]float64{
		{0.02, -0.01, 0.03, 0.01, -0.02, 0.015, -0.005, 0.025, 0.005, -0.015},
	}
	confidence := 0.95

	var_, cvar, err := VarHistoricalBatchFromReturns(returnsMatrix, confidence)
	if err != nil {
		t.Fatalf("VarHistoricalBatchFromReturns failed: %v", err)
	}

	if len(var_) != 1 {
		t.Errorf("Expected 1 VaR value, got %d", len(var_))
	}

	if cvar[0] > var_[0] {
		t.Errorf("CVaR should be <= VaR: cvar=%f, var=%f", cvar[0], var_[0])
	}
}

func TestVarHistoricalBatchFromReturns_EmptyInput(t *testing.T) {
	_, _, err := VarHistoricalBatchFromReturns([][]float64{}, 0.95)
	if err == nil {
		t.Error("Expected error for empty input")
	}
}

func TestVarHistoricalBatchFromReturns_UnequalPeriods(t *testing.T) {
	returnsMatrix := [][]float64{
		{0.01, 0.02, 0.03},
		{-0.01, 0.02}, // Different length
	}

	_, _, err := VarHistoricalBatchFromReturns(returnsMatrix, 0.95)
	if err == nil {
		t.Error("Expected error for unequal periods")
	}
}

func TestVarHistoricalBatchFromReturns_InvalidConfidence(t *testing.T) {
	returnsMatrix := [][]float64{
		{0.01, 0.02, 0.03},
	}

	_, _, err := VarHistoricalBatchFromReturns(returnsMatrix, 0.0)
	if err == nil {
		t.Error("Expected error for confidence = 0")
	}

	_, _, err = VarHistoricalBatchFromReturns(returnsMatrix, 1.0)
	if err == nil {
		t.Error("Expected error for confidence = 1")
	}

	_, _, err = VarHistoricalBatchFromReturns(returnsMatrix, 1.5)
	if err == nil {
		t.Error("Expected error for confidence > 1")
	}
}

func TestVarHistoricalBatchFromReturns_ConsistencyWithSingle(t *testing.T) {
	returnsMatrix := [][]float64{
		{0.02, -0.01, 0.03, 0.01, -0.02, 0.015, -0.005, 0.025, 0.005, -0.015,
			0.01, 0.02, -0.01, 0.03, -0.02},
		{-0.01, 0.03, 0.01, -0.02, 0.02, -0.015, 0.025, -0.005, 0.015, 0.005,
			-0.01, 0.03, 0.01, -0.02, 0.02},
	}
	confidence := 0.95

	varBatch, cvarBatch, err := VarHistoricalBatchFromReturns(returnsMatrix, confidence)
	if err != nil {
		t.Fatalf("VarHistoricalBatchFromReturns failed: %v", err)
	}

	// Compare with individual calls
	for i := 0; i < len(returnsMatrix); i++ {
		varSingle, cvarSingle, err := VarHistorical(returnsMatrix[i], confidence)
		if err != nil {
			t.Fatalf("VarHistorical failed for asset %d: %v", i, err)
		}

		if math.Abs(varBatch[i]-varSingle) > 1e-10 {
			t.Errorf("Asset %d VaR mismatch: batch=%f, single=%f", i, varBatch[i], varSingle)
		}

		if math.Abs(cvarBatch[i]-cvarSingle) > 1e-10 {
			t.Errorf("Asset %d CVaR mismatch: batch=%f, single=%f", i, cvarBatch[i], cvarSingle)
		}
	}
}

func TestVarHistoricalBatchFromReturns_99Confidence(t *testing.T) {
	// Test with higher confidence level
	returnsMatrix := [][]float64{
		{0.02, -0.01, 0.03, 0.01, -0.02, 0.015, -0.005, 0.025, 0.005, -0.015,
			0.01, 0.02, -0.01, 0.03, -0.02, 0.015, -0.005, 0.025, 0.005, -0.015},
		{-0.01, 0.03, 0.01, -0.02, 0.02, -0.015, 0.025, -0.005, 0.015, 0.005,
			-0.01, 0.03, 0.01, -0.02, 0.02, -0.015, 0.025, -0.005, 0.015, 0.005},
	}
	confidence := 0.99

	var_, cvar, err := VarHistoricalBatchFromReturns(returnsMatrix, confidence)
	if err != nil {
		t.Fatalf("VarHistoricalBatchFromReturns failed: %v", err)
	}

	for i := 0; i < len(returnsMatrix); i++ {
		if cvar[i] > var_[i] {
			t.Errorf("Asset %d: CVaR should be <= VaR at 99%% confidence", i)
		}
	}
}

func TestVarHistoricalBatchFromReturns_ManyAssets(t *testing.T) {
	// Test with many assets
	numAssets := 20
	nPeriods := 50
	returnsMatrix := make([][]float64, numAssets)

	for i := 0; i < numAssets; i++ {
		returnsMatrix[i] = make([]float64, nPeriods)
		for j := 0; j < nPeriods; j++ {
			returnsMatrix[i][j] = 0.01 * float64((i+j)%10-5)
		}
	}

	confidence := 0.95

	var_, cvar, err := VarHistoricalBatchFromReturns(returnsMatrix, confidence)
	if err != nil {
		t.Fatalf("VarHistoricalBatchFromReturns failed: %v", err)
	}

	if len(var_) != numAssets {
		t.Errorf("Expected %d VaR values, got %d", numAssets, len(var_))
	}

	if len(cvar) != numAssets {
		t.Errorf("Expected %d CVaR values, got %d", numAssets, len(cvar))
	}

	for i := 0; i < numAssets; i++ {
		if cvar[i] > var_[i] {
			t.Errorf("Asset %d: CVaR should be <= VaR", i)
		}
	}
}

func BenchmarkVarHistoricalBatchFromReturns_10Assets_100Periods(b *testing.B) {
	numAssets := 10
	nPeriods := 100
	returnsMatrix := make([][]float64, numAssets)

	for i := 0; i < numAssets; i++ {
		returnsMatrix[i] = make([]float64, nPeriods)
		for j := 0; j < nPeriods; j++ {
			returnsMatrix[i][j] = 0.01 * float64((i+j)%10-5)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VarHistoricalBatchFromReturns(returnsMatrix, 0.95)
	}
}

func BenchmarkVarHistoricalBatchFromReturns_50Assets_250Periods(b *testing.B) {
	numAssets := 50
	nPeriods := 250
	returnsMatrix := make([][]float64, numAssets)

	for i := 0; i < numAssets; i++ {
		returnsMatrix[i] = make([]float64, nPeriods)
		for j := 0; j < nPeriods; j++ {
			returnsMatrix[i][j] = 0.01 * float64((i+j)%10-5)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VarHistoricalBatchFromReturns(returnsMatrix, 0.95)
	}
}

func BenchmarkVarHistoricalBatchFromReturns_100Assets_500Periods(b *testing.B) {
	numAssets := 100
	nPeriods := 500
	returnsMatrix := make([][]float64, numAssets)

	for i := 0; i < numAssets; i++ {
		returnsMatrix[i] = make([]float64, nPeriods)
		for j := 0; j < nPeriods; j++ {
			returnsMatrix[i][j] = 0.01 * float64((i+j)%10-5)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VarHistoricalBatchFromReturns(returnsMatrix, 0.95)
	}
}
