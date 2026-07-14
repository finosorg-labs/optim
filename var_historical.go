package optim

/*
#include "var_historical.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// VarHistorical computes VaR and CVaR using historical simulation.
//
// Uses the empirical distribution of historical returns without making
// distributional assumptions. VaR is the (1-α) quantile, CVaR is the
// average of returns at or below VaR.
//
// Parameters:
//   - returns: Historical return observations
//   - confidence: Confidence level (must be in (0, 1), typically 0.95 or 0.99)
//
// Returns:
//   - var: Value at Risk (negative indicates loss)
//   - cvar: Conditional Value at Risk (negative indicates loss)
//   - error: nil on success, error otherwise
//
// Note: Requires sufficient historical data (typically n >= 250)
func VarHistorical(returns []float64, confidence float64) (var_, cvar float64, err error) {
	n := len(returns)
	if n == 0 {
		return 0, 0, fmt.Errorf("returns cannot be empty")
	}

	var cVar, cCvar C.double

	ret := C.fc_optim_var_historical(
		(*C.double)(unsafe.Pointer(&returns[0])),
		C.size_t(n),
		C.double(confidence),
		&cVar,
		&cCvar,
	)

	if ret != 0 {
		switch ret {
		case -1:
			return 0, 0, fmt.Errorf("internal error: NULL pointer")
		case -2:
			return 0, 0, fmt.Errorf("size is zero")
		case -3:
			return 0, 0, fmt.Errorf("confidence must be in (0, 1)")
		case -4:
			return 0, 0, fmt.Errorf("memory allocation failed")
		default:
			return 0, 0, fmt.Errorf("unknown error: %d", ret)
		}
	}

	return float64(cVar), float64(cCvar), nil
}

// VarHistoricalPortfolio computes VaR and CVaR for a weighted portfolio.
//
// Computes portfolio returns as weighted sum of individual asset returns,
// then applies historical simulation method.
//
// Parameters:
//   - returns: Asset return matrix (m assets × n time periods, row-major)
//   - weights: Portfolio weights (m elements, should sum to ~1.0)
//   - confidence: Confidence level (must be in (0, 1))
//
// Returns:
//   - var: Portfolio VaR
//   - cvar: Portfolio CVaR
//   - error: nil on success, error otherwise
//
// Performance: 500 assets × 1000 periods in < 10ms
func VarHistoricalPortfolio(returns [][]float64, weights []float64, confidence float64) (var_, cvar float64, err error) {
	if len(returns) == 0 {
		return 0, 0, fmt.Errorf("returns cannot be empty")
	}

	m := len(returns)
	n := len(returns[0])

	if len(weights) != m {
		return 0, 0, fmt.Errorf("weights length must match number of assets")
	}

	for i := 1; i < m; i++ {
		if len(returns[i]) != n {
			return 0, 0, fmt.Errorf("all assets must have same number of periods")
		}
	}

	returnsFlat := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			returnsFlat[i*n+j] = returns[i][j]
		}
	}

	var cVar, cCvar C.double

	ret := C.fc_optim_var_historical_portfolio(
		(*C.double)(unsafe.Pointer(&returnsFlat[0])),
		(*C.double)(unsafe.Pointer(&weights[0])),
		C.size_t(m),
		C.size_t(n),
		C.double(confidence),
		&cVar,
		&cCvar,
	)

	if ret != 0 {
		switch ret {
		case -1:
			return 0, 0, fmt.Errorf("internal error: NULL pointer")
		case -2:
			return 0, 0, fmt.Errorf("size is zero")
		case -3:
			return 0, 0, fmt.Errorf("confidence must be in (0, 1)")
		case -4:
			return 0, 0, fmt.Errorf("memory allocation failed")
		default:
			return 0, 0, fmt.Errorf("unknown error: %d", ret)
		}
	}

	return float64(cVar), float64(cCvar), nil
}

// VarHistoricalBatch computes VaR and CVaR for multiple portfolios.
//
// Efficient batch processing for multiple portfolios sharing the same
// asset return history but with different weights.
//
// Parameters:
//   - returns: Asset return matrix (m assets × n time periods)
//   - weightsMatrix: Portfolio weights matrix (num_portfolios × m assets)
//   - confidence: Confidence level
//
// Returns:
//   - var: VaR values for each portfolio
//   - cvar: CVaR values for each portfolio
//   - error: nil on success, error otherwise
func VarHistoricalBatch(returns [][]float64, weightsMatrix [][]float64, confidence float64) (var_, cvar []float64, err error) {
	if len(returns) == 0 || len(weightsMatrix) == 0 {
		return nil, nil, fmt.Errorf("inputs cannot be empty")
	}

	m := len(returns)
	n := len(returns[0])
	numPortfolios := len(weightsMatrix)

	for i := 0; i < m; i++ {
		if len(returns[i]) != n {
			return nil, nil, fmt.Errorf("all assets must have same number of periods")
		}
	}

	for i := 0; i < numPortfolios; i++ {
		if len(weightsMatrix[i]) != m {
			return nil, nil, fmt.Errorf("all weight vectors must have length m")
		}
	}

	returnsFlat := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			returnsFlat[i*n+j] = returns[i][j]
		}
	}

	weightsFlat := make([]float64, numPortfolios*m)
	for i := 0; i < numPortfolios; i++ {
		for j := 0; j < m; j++ {
			weightsFlat[i*m+j] = weightsMatrix[i][j]
		}
	}

	var_ = make([]float64, numPortfolios)
	cvar = make([]float64, numPortfolios)

	ret := C.fc_optim_var_historical_batch(
		(*C.double)(unsafe.Pointer(&returnsFlat[0])),
		(*C.double)(unsafe.Pointer(&weightsFlat[0])),
		C.size_t(numPortfolios),
		C.size_t(m),
		C.size_t(n),
		C.double(confidence),
		(*C.double)(unsafe.Pointer(&var_[0])),
		(*C.double)(unsafe.Pointer(&cvar[0])),
	)

	if ret != 0 {
		switch ret {
		case -1:
			return nil, nil, fmt.Errorf("internal error: NULL pointer")
		case -2:
			return nil, nil, fmt.Errorf("size is zero")
		case -3:
			return nil, nil, fmt.Errorf("confidence must be in (0, 1)")
		case -4:
			return nil, nil, fmt.Errorf("memory allocation failed")
		default:
			return nil, nil, fmt.Errorf("unknown error: %d", ret)
		}
	}

	return var_, cvar, nil
}

// VarHistoricalBatchFromReturns computes VaR and CVaR for multiple single-asset return series in batch.
//
// Batch processing for multiple single assets, each with its own historical return series.
// Efficient for risk measurement across many individual assets.
//
// Parameters:
//   - returnsMatrix: Matrix of return series (numAssets × n, row-major)
//     Each row is a separate asset's historical returns
//   - confidence: Confidence level (must be in (0, 1))
//
// Returns:
//   - var: VaR values for each asset
//   - cvar: CVaR values for each asset
//   - error: nil on success, error otherwise
func VarHistoricalBatchFromReturns(returnsMatrix [][]float64, confidence float64) (var_, cvar []float64, err error) {
	if len(returnsMatrix) == 0 {
		return nil, nil, fmt.Errorf("returnsMatrix cannot be empty")
	}

	numAssets := len(returnsMatrix)
	n := len(returnsMatrix[0])

	for i := 1; i < numAssets; i++ {
		if len(returnsMatrix[i]) != n {
			return nil, nil, fmt.Errorf("all assets must have same number of periods")
		}
	}

	returnsFlat := make([]float64, numAssets*n)
	for i := 0; i < numAssets; i++ {
		for j := 0; j < n; j++ {
			returnsFlat[i*n+j] = returnsMatrix[i][j]
		}
	}

	var_ = make([]float64, numAssets)
	cvar = make([]float64, numAssets)

	ret := C.fc_optim_var_historical_batch_from_returns(
		(*C.double)(unsafe.Pointer(&returnsFlat[0])),
		C.size_t(numAssets),
		C.size_t(n),
		C.double(confidence),
		(*C.double)(unsafe.Pointer(&var_[0])),
		(*C.double)(unsafe.Pointer(&cvar[0])),
	)

	if ret != 0 {
		switch ret {
		case -1:
			return nil, nil, fmt.Errorf("internal error: NULL pointer")
		case -2:
			return nil, nil, fmt.Errorf("size is zero")
		case -3:
			return nil, nil, fmt.Errorf("confidence must be in (0, 1)")
		case -4:
			return nil, nil, fmt.Errorf("memory allocation failed")
		default:
			return nil, nil, fmt.Errorf("unknown error: %d", ret)
		}
	}

	return var_, cvar, nil
}
