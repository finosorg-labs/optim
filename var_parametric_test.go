package optim

import (
	"math"
	"testing"
)

func TestVarParametric_Basic(t *testing.T) {
	mean := 0.001
	stddev := 0.02
	confidence := 0.95

	var_, cvar, err := VarParametric(mean, stddev, confidence)
	if err != nil {
		t.Fatalf("VarParametric failed: %v", err)
	}

	if var_ >= mean {
		t.Errorf("VaR should be less than mean: var=%f, mean=%f", var_, mean)
	}

	if cvar >= var_ {
		t.Errorf("CVaR should be less than VaR: cvar=%f, var=%f", cvar, var_)
	}

	expectedVar := mean - 1.645*stddev
	if math.Abs(var_-expectedVar) > 0.001 {
		t.Errorf("VaR mismatch: got %f, expected ~%f", var_, expectedVar)
	}
}

func TestVarParametric_99Confidence(t *testing.T) {
	mean := 0.0
	stddev := 1.0
	confidence := 0.99

	var_, cvar, err := VarParametric(mean, stddev, confidence)
	if err != nil {
		t.Fatalf("VarParametric failed: %v", err)
	}

	if math.Abs(var_-(-2.326)) > 0.01 {
		t.Errorf("VaR mismatch for 99%% confidence: got %f, expected ~-2.326", var_)
	}

	if cvar >= var_ {
		t.Errorf("CVaR should be less than VaR")
	}
}

func TestVarParametric_InvalidStddev(t *testing.T) {
	_, _, err := VarParametric(0.0, 0.0, 0.95)
	if err == nil {
		t.Error("Expected error for zero stddev")
	}

	_, _, err = VarParametric(0.0, -0.1, 0.95)
	if err == nil {
		t.Error("Expected error for negative stddev")
	}
}

func TestVarParametric_InvalidConfidence(t *testing.T) {
	_, _, err := VarParametric(0.0, 1.0, 0.0)
	if err == nil {
		t.Error("Expected error for confidence = 0")
	}

	_, _, err = VarParametric(0.0, 1.0, 1.0)
	if err == nil {
		t.Error("Expected error for confidence = 1")
	}

	_, _, err = VarParametric(0.0, 1.0, -0.5)
	if err == nil {
		t.Error("Expected error for negative confidence")
	}

	_, _, err = VarParametric(0.0, 1.0, 1.5)
	if err == nil {
		t.Error("Expected error for confidence > 1")
	}
}

func TestVarParametricBatch_Basic(t *testing.T) {
	means := []float64{0.001, 0.002, -0.001, 0.0}
	stddevs := []float64{0.02, 0.03, 0.025, 0.01}
	confidence := 0.95

	var_, cvar, err := VarParametricBatch(means, stddevs, confidence)
	if err != nil {
		t.Fatalf("VarParametricBatch failed: %v", err)
	}

	if len(var_) != len(means) {
		t.Errorf("Output length mismatch: got %d, expected %d", len(var_), len(means))
	}

	if len(cvar) != len(means) {
		t.Errorf("Output length mismatch: got %d, expected %d", len(cvar), len(means))
	}

	for i := 0; i < len(means); i++ {
		if var_[i] >= means[i] {
			t.Errorf("VaR[%d] should be less than mean: var=%f, mean=%f", i, var_[i], means[i])
		}
		if cvar[i] >= var_[i] {
			t.Errorf("CVaR[%d] should be less than VaR: cvar=%f, var=%f", i, cvar[i], var_[i])
		}
	}
}

func TestVarParametricBatch_Large(t *testing.T) {
	n := 500
	means := make([]float64, n)
	stddevs := make([]float64, n)

	for i := 0; i < n; i++ {
		means[i] = 0.001 * float64(i%10)
		stddevs[i] = 0.01 + 0.001*float64(i%20)
	}

	var_, cvar, err := VarParametricBatch(means, stddevs, 0.95)
	if err != nil {
		t.Fatalf("VarParametricBatch failed: %v", err)
	}

	if len(var_) != n {
		t.Errorf("Output length mismatch: got %d, expected %d", len(var_), n)
	}

	for i := 0; i < n; i++ {
		if var_[i] >= means[i] {
			t.Errorf("VaR[%d] should be less than mean", i)
		}
		if cvar[i] >= var_[i] {
			t.Errorf("CVaR[%d] should be less than VaR", i)
		}
	}
}

func TestVarParametricBatch_Consistency(t *testing.T) {
	means := []float64{0.001, 0.002, -0.001}
	stddevs := []float64{0.02, 0.03, 0.025}
	confidence := 0.95

	varBatch, cvarBatch, err := VarParametricBatch(means, stddevs, confidence)
	if err != nil {
		t.Fatalf("VarParametricBatch failed: %v", err)
	}

	for i := 0; i < len(means); i++ {
		varSingle, cvarSingle, err := VarParametric(means[i], stddevs[i], confidence)
		if err != nil {
			t.Fatalf("VarParametric failed for index %d: %v", i, err)
		}

		if math.Abs(varBatch[i]-varSingle) > 1e-10 {
			t.Errorf("VaR[%d] mismatch: batch=%f, single=%f", i, varBatch[i], varSingle)
		}

		if math.Abs(cvarBatch[i]-cvarSingle) > 1e-10 {
			t.Errorf("CVaR[%d] mismatch: batch=%f, single=%f", i, cvarBatch[i], cvarSingle)
		}
	}
}

func TestVarParametricBatch_EmptyInput(t *testing.T) {
	_, _, err := VarParametricBatch([]float64{}, []float64{}, 0.95)
	if err == nil {
		t.Error("Expected error for empty input")
	}
}

func TestVarParametricBatch_LengthMismatch(t *testing.T) {
	means := []float64{0.001, 0.002}
	stddevs := []float64{0.02}

	_, _, err := VarParametricBatch(means, stddevs, 0.95)
	if err == nil {
		t.Error("Expected error for length mismatch")
	}
}

func TestVarParametricBatch_InvalidStddev(t *testing.T) {
	means := []float64{0.001, 0.002}
	stddevs := []float64{0.02, -0.01}

	_, _, err := VarParametricBatch(means, stddevs, 0.95)
	if err == nil {
		t.Error("Expected error for negative stddev")
	}

	stddevs[1] = 0.0
	_, _, err = VarParametricBatch(means, stddevs, 0.95)
	if err == nil {
		t.Error("Expected error for zero stddev")
	}
}

func TestVarParametricBatch_InvalidConfidence(t *testing.T) {
	means := []float64{0.001}
	stddevs := []float64{0.02}

	_, _, err := VarParametricBatch(means, stddevs, 0.0)
	if err == nil {
		t.Error("Expected error for confidence = 0")
	}

	_, _, err = VarParametricBatch(means, stddevs, 1.0)
	if err == nil {
		t.Error("Expected error for confidence = 1")
	}
}

func BenchmarkVarParametric(b *testing.B) {
	mean := 0.001
	stddev := 0.02
	confidence := 0.95

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VarParametric(mean, stddev, confidence)
	}
}

func BenchmarkVarParametricBatch_100(b *testing.B) {
	n := 100
	means := make([]float64, n)
	stddevs := make([]float64, n)

	for i := 0; i < n; i++ {
		means[i] = 0.001 * float64(i%10)
		stddevs[i] = 0.01 + 0.001*float64(i%20)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VarParametricBatch(means, stddevs, 0.95)
	}
}

func BenchmarkVarParametricBatch_500(b *testing.B) {
	n := 500
	means := make([]float64, n)
	stddevs := make([]float64, n)

	for i := 0; i < n; i++ {
		means[i] = 0.001 * float64(i%10)
		stddevs[i] = 0.01 + 0.001*float64(i%20)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VarParametricBatch(means, stddevs, 0.95)
	}
}

func BenchmarkVarParametricBatch_1000(b *testing.B) {
	n := 1000
	means := make([]float64, n)
	stddevs := make([]float64, n)

	for i := 0; i < n; i++ {
		means[i] = 0.001 * float64(i%10)
		stddevs[i] = 0.01 + 0.001*float64(i%20)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VarParametricBatch(means, stddevs, 0.95)
	}
}
func TestVarParametricFromReturns_Basic(t *testing.T) {
	// Generate synthetic returns with known mean and stddev
	returns := []float64{
		0.02, -0.01, 0.03, 0.01, -0.02,
		0.015, -0.005, 0.025, 0.005, -0.015,
		0.01, 0.02, -0.01, 0.03, -0.02,
	}
	confidence := 0.95

	var_, cvar, err := VarParametricFromReturns(returns, confidence)
	if err != nil {
		t.Fatalf("VarParametricFromReturns failed: %v", err)
	}

	// VaR should be negative (loss) for typical returns
	if cvar >= var_ {
		t.Errorf("CVaR should be less than VaR: cvar=%f, var=%f", cvar, var_)
	}

	// Verify it's finite
	if math.IsNaN(var_) || math.IsInf(var_, 0) {
		t.Errorf("VaR is not finite: %f", var_)
	}
	if math.IsNaN(cvar) || math.IsInf(cvar, 0) {
		t.Errorf("CVaR is not finite: %f", cvar)
	}
}

func TestVarParametricFromReturns_InsufficientData(t *testing.T) {
	// Only 1 return - need at least 2
	returns := []float64{0.01}
	_, _, err := VarParametricFromReturns(returns, 0.95)
	if err == nil {
		t.Error("Expected error for insufficient data")
	}
}

func TestVarParametricFromReturns_InvalidConfidence(t *testing.T) {
	returns := []float64{0.01, 0.02, -0.01}

	_, _, err := VarParametricFromReturns(returns, 0.0)
	if err == nil {
		t.Error("Expected error for confidence = 0")
	}

	_, _, err = VarParametricFromReturns(returns, 1.0)
	if err == nil {
		t.Error("Expected error for confidence = 1")
	}

	_, _, err = VarParametricFromReturns(returns, 1.5)
	if err == nil {
		t.Error("Expected error for confidence > 1")
	}
}

func TestVarParametricFromPortfolioReturns_Basic(t *testing.T) {
	// 2 assets, 10 periods
	dim := 2
	nPeriods := 10
	returns := []float64{
		// Asset 0 returns
		0.01, 0.02, -0.01, 0.03, -0.02, 0.015, -0.005, 0.025, 0.005, -0.015,
		// Asset 1 returns
		-0.01, 0.03, 0.01, -0.02, 0.02, -0.015, 0.025, -0.005, 0.015, 0.005,
	}
	weights := []float64{0.6, 0.4}
	confidence := 0.95

	var_, cvar, err := VarParametricFromPortfolioReturns(returns, weights, dim, nPeriods, confidence)
	if err != nil {
		t.Fatalf("VarParametricFromPortfolioReturns failed: %v", err)
	}

	if cvar >= var_ {
		t.Errorf("CVaR should be less than VaR: cvar=%f, var=%f", cvar, var_)
	}

	// Verify finite values
	if math.IsNaN(var_) || math.IsInf(var_, 0) {
		t.Errorf("VaR is not finite: %f", var_)
	}
	if math.IsNaN(cvar) || math.IsInf(cvar, 0) {
		t.Errorf("CVaR is not finite: %f", cvar)
	}
}

func TestVarParametricFromPortfolioReturns_InsufficientPeriods(t *testing.T) {
	dim := 2
	nPeriods := 2 // Need at least dim + 1 = 3
	returns := make([]float64, dim*nPeriods)
	weights := []float64{0.5, 0.5}

	_, _, err := VarParametricFromPortfolioReturns(returns, weights, dim, nPeriods, 0.95)
	if err == nil {
		t.Error("Expected error for insufficient periods")
	}
}

func TestVarParametricFromPortfolioReturns_InvalidDimensions(t *testing.T) {
	dim := 2
	nPeriods := 10
	returns := make([]float64, dim*nPeriods-1) // Wrong size
	weights := []float64{0.5, 0.5}

	_, _, err := VarParametricFromPortfolioReturns(returns, weights, dim, nPeriods, 0.95)
	if err == nil {
		t.Error("Expected error for wrong returns length")
	}

	returns = make([]float64, dim*nPeriods)
	weights = []float64{0.5} // Wrong size

	_, _, err = VarParametricFromPortfolioReturns(returns, weights, dim, nPeriods, 0.95)
	if err == nil {
		t.Error("Expected error for wrong weights length")
	}
}

func TestVarParametricFromPortfolioReturns_ThreeAssets(t *testing.T) {
	// 3 assets, 20 periods
	dim := 3
	nPeriods := 20
	returns := make([]float64, dim*nPeriods)

	// Generate some synthetic returns
	for i := 0; i < dim; i++ {
		for j := 0; j < nPeriods; j++ {
			returns[i*nPeriods+j] = 0.01 * float64((i+j)%5-2)
		}
	}

	weights := []float64{0.5, 0.3, 0.2}
	confidence := 0.99

	var_, cvar, err := VarParametricFromPortfolioReturns(returns, weights, dim, nPeriods, confidence)
	if err != nil {
		t.Fatalf("VarParametricFromPortfolioReturns failed: %v", err)
	}

	if cvar >= var_ {
		t.Errorf("CVaR should be less than VaR: cvar=%f, var=%f", cvar, var_)
	}
}

func BenchmarkVarParametricFromReturns_50Periods(b *testing.B) {
	returns := make([]float64, 50)
	for i := 0; i < 50; i++ {
		returns[i] = 0.01 * float64(i%10-5)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VarParametricFromReturns(returns, 0.95)
	}
}

func BenchmarkVarParametricFromReturns_250Periods(b *testing.B) {
	returns := make([]float64, 250)
	for i := 0; i < 250; i++ {
		returns[i] = 0.01 * float64(i%10-5)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VarParametricFromReturns(returns, 0.95)
	}
}

func BenchmarkVarParametricFromPortfolioReturns_10Assets_100Periods(b *testing.B) {
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
		VarParametricFromPortfolioReturns(returns, weights, dim, nPeriods, 0.95)
	}
}

func BenchmarkVarParametricFromPortfolioReturns_50Assets_250Periods(b *testing.B) {
	dim := 50
	nPeriods := 250
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
		VarParametricFromPortfolioReturns(returns, weights, dim, nPeriods, 0.95)
	}
}
