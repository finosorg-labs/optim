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

// VarParametricFromReturns computes VaR and CVaR from historical returns using the parametric method.
//
// Convenience function that computes mean and standard deviation from historical
// returns, then applies the parametric method. Assumes returns are normally distributed.
//
// Parameters:
//   - returns: Historical returns data
//   - confidence: Confidence level (must be in (0, 1))
//
// Returns:
//   - var: Value at Risk
//   - cvar: Conditional Value at Risk
//   - error: nil on success, error otherwise
//
// Performance: O(n_periods) time complexity
func VarParametricFromReturns(returns []float64, confidence float64) (var_, cvar float64, err error) {
	if len(returns) < 2 {
		return 0, 0, fmt.Errorf("returns must have at least 2 elements")
	}

	var cVar, cCvar C.double

	ret := C.fc_optim_var_parametric_from_returns(
		(*C.double)(unsafe.Pointer(&returns[0])),
		C.size_t(len(returns)),
		C.double(confidence),
		&cVar,
		&cCvar,
	)

	if ret != 0 {
		switch ret {
		case -1:
			return 0, 0, fmt.Errorf("required input pointer is NULL")
		case -2:
			return 0, 0, fmt.Errorf("insufficient data: need at least 2 periods")
		case -3:
			return 0, 0, fmt.Errorf("confidence must be in (0, 1)")
		case -4:
			return 0, 0, fmt.Errorf("computation failed")
		default:
			return 0, 0, fmt.Errorf("unknown error: %d", ret)
		}
	}

	return float64(cVar), float64(cCvar), nil
}

// VarParametricFromPortfolioReturns computes VaR and CVaR for a portfolio from multi-asset returns.
//
// This is the most complete convenience function that handles the full workflow
// from raw multi-asset data to VaR/CVaR:
// 1. Compute mean return for each asset
// 2. Compute covariance matrix
// 3. Calculate portfolio mean and variance
// 4. Apply parametric formulas
//
// Parameters:
//   - returns: Historical returns matrix (dim × n_periods, row-major)
//   - weights: Portfolio weights (dim elements, should sum to 1.0)
//   - dim: Number of assets
//   - nPeriods: Number of historical periods (must be >= dim + 1)
//   - confidence: Confidence level (must be in (0, 1))
//
// Returns:
//   - var: Value at Risk
//   - cvar: Conditional Value at Risk
//   - error: nil on success, error otherwise
//
// Performance: O(dim² × n_periods + dim²) time complexity
func VarParametricFromPortfolioReturns(returns []float64, weights []float64, dim, nPeriods int, confidence float64) (var_, cvar float64, err error) {
	if dim <= 0 {
		return 0, 0, fmt.Errorf("dim must be > 0")
	}

	if nPeriods <= dim {
		return 0, 0, fmt.Errorf("nPeriods must be > dim (need at least %d periods)", dim+1)
	}

	if len(returns) != dim*nPeriods {
		return 0, 0, fmt.Errorf("returns length must be dim × nPeriods (%d × %d = %d)", dim, nPeriods, dim*nPeriods)
	}

	if len(weights) != dim {
		return 0, 0, fmt.Errorf("weights length must equal dim (%d)", dim)
	}

	var cVar, cCvar C.double

	ret := C.fc_optim_var_parametric_from_portfolio_returns(
		(*C.double)(unsafe.Pointer(&returns[0])),
		(*C.double)(unsafe.Pointer(&weights[0])),
		C.size_t(dim),
		C.size_t(nPeriods),
		C.double(confidence),
		&cVar,
		&cCvar,
	)

	if ret != 0 {
		switch ret {
		case -1:
			return 0, 0, fmt.Errorf("required input pointer is NULL")
		case -2:
			return 0, 0, fmt.Errorf("invalid dimensions or insufficient data")
		case -3:
			return 0, 0, fmt.Errorf("confidence must be in (0, 1)")
		case -4:
			return 0, 0, fmt.Errorf("computation failed (memory allocation or covariance error)")
		default:
			return 0, 0, fmt.Errorf("unknown error: %d", ret)
		}
	}

	return float64(cVar), float64(cCvar), nil
}
