package optim

/*
#include "var_monte_carlo.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// VarMonteCarloState holds the state for Monte Carlo VaR/CVaR simulations.
//
// Contains random number generator state and workspace buffers.
// Must be created with NewVarMonteCarloState and freed with Close.
type VarMonteCarloState struct {
	state *C.fc_var_mc_state_t
	dim   int
}

// NewVarMonteCarloState creates a new Monte Carlo VaR calculation state.
//
// Parameters:
//   - dim: Number of assets (dimension of correlated random vectors)
//   - nPaths: Number of Monte Carlo simulation paths
//   - seed: Random seed (use 0 for time-based seed)
//
// Returns state object or error. Call Close() when done to free resources.
func NewVarMonteCarloState(dim, nPaths int, seed uint64) (*VarMonteCarloState, error) {
	if dim <= 0 || nPaths <= 0 {
		return nil, fmt.Errorf("dim and nPaths must be positive")
	}

	state := C.fc_optim_var_monte_carlo_state_create(
		C.size_t(dim),
		C.size_t(nPaths),
		C.uint64_t(seed),
	)

	if state == nil {
		return nil, fmt.Errorf("failed to create Monte Carlo state")
	}

	return &VarMonteCarloState{state: state, dim: dim}, nil
}

// Close frees resources associated with the Monte Carlo state.
func (s *VarMonteCarloState) Close() {
	if s.state != nil {
		C.fc_optim_var_monte_carlo_state_destroy(s.state)
		s.state = nil
	}
}

// VarMonteCarlo computes VaR and CVaR using Monte Carlo simulation.
//
// Simulates correlated asset returns using multivariate normal distribution
// and computes VaR/CVaR from simulated portfolio returns.
//
// Parameters:
//   - means: Asset mean returns (length must equal dim)
//   - covMatrix: Covariance matrix (dim × dim, row-major, must be positive definite)
//   - weights: Portfolio weights (length must equal dim)
//   - confidence: Confidence level (must be in (0, 1))
//
// Returns:
//   - var: Value at Risk
//   - cvar: Conditional Value at Risk
//   - error: nil on success, error otherwise
//
// Performance: 500 assets × 10000 paths in < 100ms on modern x86_64 with AVX2
// Note: Results are stochastic and vary between runs unless seed is fixed
func (s *VarMonteCarloState) VarMonteCarlo(means []float64, covMatrix [][]float64, weights []float64, confidence float64) (var_, cvar float64, err error) {
	if s.state == nil {
		return 0, 0, fmt.Errorf("state is closed or invalid")
	}

	if len(means) != s.dim || len(weights) != s.dim {
		return 0, 0, fmt.Errorf("means and weights length must equal dim (%d)", s.dim)
	}

	if len(covMatrix) != s.dim {
		return 0, 0, fmt.Errorf("covMatrix must be %d×%d", s.dim, s.dim)
	}

	covFlat := make([]float64, s.dim*s.dim)
	for i := 0; i < s.dim; i++ {
		if len(covMatrix[i]) != s.dim {
			return 0, 0, fmt.Errorf("covMatrix row %d has wrong length", i)
		}
		for j := 0; j < s.dim; j++ {
			covFlat[i*s.dim+j] = covMatrix[i][j]
		}
	}

	var cVar, cCvar C.double

	ret := C.fc_optim_var_monte_carlo(
		s.state,
		(*C.double)(unsafe.Pointer(&means[0])),
		(*C.double)(unsafe.Pointer(&covFlat[0])),
		(*C.double)(unsafe.Pointer(&weights[0])),
		C.size_t(s.dim),
		C.double(confidence),
		&cVar,
		&cCvar,
	)

	if ret != 0 {
		switch ret {
		case -1:
			return 0, 0, fmt.Errorf("internal error: NULL pointer")
		case -2:
			return 0, 0, fmt.Errorf("dimension mismatch")
		case -3:
			return 0, 0, fmt.Errorf("confidence must be in (0, 1)")
		case -4:
			return 0, 0, fmt.Errorf("covariance matrix is not positive definite")
		case -5:
			return 0, 0, fmt.Errorf("failed to generate correlated samples")
		default:
			return 0, 0, fmt.Errorf("unknown error: %d", ret)
		}
	}

	return float64(cVar), float64(cCvar), nil
}

// VarMonteCarloFromHistory computes VaR and CVaR from historical returns.
//
// Convenience function that estimates covariance matrix from historical data
// and then performs Monte Carlo simulation.
//
// Parameters:
//   - returns: Historical returns matrix (dim × nPeriods, row-major)
//   - weights: Portfolio weights (length must equal dim)
//   - confidence: Confidence level
//
// Returns VaR, CVaR, and error. Requires nPeriods >= dim + 1.
func (s *VarMonteCarloState) VarMonteCarloFromHistory(returns [][]float64, weights []float64, confidence float64) (var_, cvar float64, err error) {
	if s.state == nil {
		return 0, 0, fmt.Errorf("state is closed or invalid")
	}

	if len(returns) != s.dim {
		return 0, 0, fmt.Errorf("returns must have %d assets", s.dim)
	}

	if len(weights) != s.dim {
		return 0, 0, fmt.Errorf("weights length must equal dim (%d)", s.dim)
	}

	nPeriods := len(returns[0])
	for i := 1; i < s.dim; i++ {
		if len(returns[i]) != nPeriods {
			return 0, 0, fmt.Errorf("all assets must have same number of periods")
		}
	}

	returnsFlat := make([]float64, s.dim*nPeriods)
	for i := 0; i < s.dim; i++ {
		for j := 0; j < nPeriods; j++ {
			returnsFlat[i*nPeriods+j] = returns[i][j]
		}
	}

	var cVar, cCvar C.double

	ret := C.fc_optim_var_monte_carlo_from_history(
		s.state,
		(*C.double)(unsafe.Pointer(&returnsFlat[0])),
		(*C.double)(unsafe.Pointer(&weights[0])),
		C.size_t(s.dim),
		C.size_t(nPeriods),
		C.double(confidence),
		&cVar,
		&cCvar,
	)

	if ret != 0 {
		switch ret {
		case -1:
			return 0, 0, fmt.Errorf("internal error: NULL pointer")
		case -2:
			return 0, 0, fmt.Errorf("dimension mismatch")
		case -3:
			return 0, 0, fmt.Errorf("confidence must be in (0, 1)")
		case -6:
			return 0, 0, fmt.Errorf("insufficient historical data: need nPeriods >= dim + 1")
		case -7:
			return 0, 0, fmt.Errorf("memory allocation failed")
		default:
			return 0, 0, fmt.Errorf("unknown error: %d", ret)
		}
	}

	return float64(cVar), float64(cCvar), nil
}
