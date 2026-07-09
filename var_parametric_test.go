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
