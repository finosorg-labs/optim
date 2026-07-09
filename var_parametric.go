package optim

/*
#include "var_parametric.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// VarParametric computes VaR and CVaR using the parametric method.
//
// Assumes portfolio returns follow a normal distribution and uses analytical
// formulas based on mean and standard deviation.
//
// Parameters:
//   - mean: Portfolio mean return
//   - stddev: Portfolio standard deviation (must be > 0)
//   - confidence: Confidence level (must be in (0, 1), typically 0.95 or 0.99)
//
// Returns:
//   - var: Value at Risk (negative indicates potential loss)
//   - cvar: Conditional Value at Risk (negative indicates potential loss)
//   - error: nil on success, error otherwise
//
// Performance: Completes in < 100ns for single portfolio
func VarParametric(mean, stddev, confidence float64) (var_, cvar float64, err error) {
	var cVar, cCvar C.double

	ret := C.fc_optim_var_parametric(
		C.double(mean),
		C.double(stddev),
		C.double(confidence),
		&cVar,
		&cCvar,
	)

	if ret != 0 {
		switch ret {
		case -1:
			return 0, 0, fmt.Errorf("internal error: NULL output pointer")
		case -2:
			return 0, 0, fmt.Errorf("stddev must be > 0")
		case -3:
			return 0, 0, fmt.Errorf("confidence must be in (0, 1)")
		default:
			return 0, 0, fmt.Errorf("unknown error: %d", ret)
		}
	}

	return float64(cVar), float64(cCvar), nil
}

// VarParametricBatch computes VaR and CVaR for multiple portfolios using the parametric method.
//
// Batch processing with SIMD optimizations for improved performance.
//
// Parameters:
//   - means: Portfolio mean returns (must have same length as stddevs)
//   - stddevs: Portfolio standard deviations (all must be > 0)
//   - confidence: Confidence level (must be in (0, 1))
//
// Returns:
//   - var: VaR values for each portfolio
//   - cvar: CVaR values for each portfolio
//   - error: nil on success, error otherwise
//
// Performance: 500 portfolios in < 5ms on modern x86_64 with AVX2
func VarParametricBatch(means, stddevs []float64, confidence float64) (var_, cvar []float64, err error) {
	n := len(means)
	if n == 0 {
		return nil, nil, fmt.Errorf("input slices cannot be empty")
	}

	if len(stddevs) != n {
		return nil, nil, fmt.Errorf("means and stddevs must have same length")
	}

	var_ = make([]float64, n)
	cvar = make([]float64, n)

	ret := C.fc_optim_var_parametric_batch(
		C.size_t(n),
		(*C.double)(unsafe.Pointer(&means[0])),
		(*C.double)(unsafe.Pointer(&stddevs[0])),
		C.double(confidence),
		(*C.double)(unsafe.Pointer(&var_[0])),
		(*C.double)(unsafe.Pointer(&cvar[0])),
	)

	if ret != 0 {
		switch ret {
		case -1:
			return nil, nil, fmt.Errorf("required input pointer is NULL")
		case -2:
			return nil, nil, fmt.Errorf("size is zero")
		case -3:
			return nil, nil, fmt.Errorf("invalid input: stddev <= 0 or confidence not in (0, 1)")
		case -4:
			return nil, nil, fmt.Errorf("memory allocation failed")
		default:
			return nil, nil, fmt.Errorf("unknown error: %d", ret)
		}
	}

	return var_, cvar, nil
}
