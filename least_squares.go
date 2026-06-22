package optim

/*
#include "least_squares.h"
*/
import "C"
import (
	"errors"
	"unsafe"
)

var (
	ErrNullPointer       = errors.New("null pointer provided")
	ErrInvalidDimensions = errors.New("invalid dimensions")
	ErrRankDeficient     = errors.New("matrix is rank deficient")
)

// LeastSquares solves ordinary least squares regression using QR decomposition.
//
// Solves: min ||y - X*beta||^2
//
// Parameters:
//   - X: Design matrix in column-major order (n x p)
//   - y: Response vector (n x 1)
//   - n: Number of observations
//   - p: Number of predictors
//
// Returns:
//   - beta: Regression coefficients (p x 1)
//   - error: nil on success, error otherwise
//
// Note: The input matrix X is modified during computation.
func LeastSquares(X, y []float64, n, p int) ([]float64, error) {
	if len(X) < n*p {
		return nil, ErrInvalidDimensions
	}
	if len(y) < n {
		return nil, ErrInvalidDimensions
	}
	if n <= 0 || p <= 0 || n < p {
		return nil, ErrInvalidDimensions
	}

	beta := make([]float64, p)

	ret := C.fc_optim_least_squares(
		(*C.double)(unsafe.Pointer(&X[0])),
		(*C.double)(unsafe.Pointer(&y[0])),
		C.size_t(n),
		C.size_t(p),
		(*C.double)(unsafe.Pointer(&beta[0])),
	)

	if ret == -1 {
		return nil, ErrNullPointer
	} else if ret == -2 {
		return nil, ErrInvalidDimensions
	} else if ret == -3 {
		return nil, ErrRankDeficient
	}

	return beta, nil
}

// LeastSquaresWork solves ordinary least squares regression with caller-provided workspace.
//
// This version avoids heap allocation for better performance in hot paths.
//
// Parameters:
//   - X: Design matrix in column-major order (n x p)
//   - y: Response vector (n x 1)
//   - n: Number of observations
//   - p: Number of predictors
//   - work: Workspace buffer with size >= (n + p)
//
// Returns:
//   - beta: Regression coefficients (p x 1)
//   - error: nil on success, error otherwise
//
// Note: The input matrix X is modified during computation.
func LeastSquaresWork(X, y []float64, n, p int, work []float64) ([]float64, error) {
	if len(X) < n*p {
		return nil, ErrInvalidDimensions
	}
	if len(y) < n {
		return nil, ErrInvalidDimensions
	}
	if len(work) < n+p {
		return nil, ErrInvalidDimensions
	}
	if n <= 0 || p <= 0 || n < p {
		return nil, ErrInvalidDimensions
	}

	beta := make([]float64, p)

	ret := C.fc_optim_least_squares_work(
		(*C.double)(unsafe.Pointer(&X[0])),
		(*C.double)(unsafe.Pointer(&y[0])),
		C.size_t(n),
		C.size_t(p),
		(*C.double)(unsafe.Pointer(&beta[0])),
		(*C.double)(unsafe.Pointer(&work[0])),
	)

	if ret == -1 {
		return nil, ErrNullPointer
	} else if ret == -2 {
		return nil, ErrInvalidDimensions
	} else if ret == -3 {
		return nil, ErrRankDeficient
	}

	return beta, nil
}

// LeastSquaresBatch solves multiple independent least squares regressions.
//
// Parameters:
//   - X: Design matrices (batch_size x n x p) in column-major order
//   - y: Response vectors (batch_size x n)
//   - n: Number of observations per regression
//   - p: Number of predictors per regression
//   - batchSize: Number of independent regressions
//
// Returns:
//   - beta: Regression coefficients (batch_size x p)
//   - successCount: Number of successfully solved regressions
//   - error: nil on success, error otherwise
//
// Note: If a regression fails, the corresponding beta coefficients are set to 0.
func LeastSquaresBatch(X, y []float64, n, p, batchSize int) ([]float64, int, error) {
	if len(X) < batchSize*n*p {
		return nil, 0, ErrInvalidDimensions
	}
	if len(y) < batchSize*n {
		return nil, 0, ErrInvalidDimensions
	}
	if n <= 0 || p <= 0 || batchSize <= 0 || n < p {
		return nil, 0, ErrInvalidDimensions
	}

	beta := make([]float64, batchSize*p)

	count := C.fc_optim_least_squares_batch(
		(*C.double)(unsafe.Pointer(&X[0])),
		(*C.double)(unsafe.Pointer(&y[0])),
		C.size_t(n),
		C.size_t(p),
		C.size_t(batchSize),
		(*C.double)(unsafe.Pointer(&beta[0])),
	)

	return beta, int(count), nil
}

// LeastSquaresResult contains the results of extended least squares regression.
type LeastSquaresResult struct {
	Beta      []float64 // Regression coefficients
	Residuals []float64 // Residuals (optional)
	RSquared  float64   // Coefficient of determination
	StdError  float64   // Residual standard error
}

// LeastSquaresExt solves ordinary least squares regression with additional statistics.
//
// Parameters:
//   - X: Design matrix in column-major order (n x p)
//   - y: Response vector (n x 1)
//   - n: Number of observations
//   - p: Number of predictors
//   - computeResiduals: Whether to compute residuals
//
// Returns:
//   - result: LeastSquaresResult containing coefficients and statistics
//   - error: nil on success, error otherwise
func LeastSquaresExt(X, y []float64, n, p int, computeResiduals bool) (*LeastSquaresResult, error) {
	if len(X) < n*p {
		return nil, ErrInvalidDimensions
	}
	if len(y) < n {
		return nil, ErrInvalidDimensions
	}
	if n <= 0 || p <= 0 || n < p {
		return nil, ErrInvalidDimensions
	}

	result := &LeastSquaresResult{
		Beta: make([]float64, p),
	}

	var residualsPtr *C.double
	if computeResiduals {
		result.Residuals = make([]float64, n)
		residualsPtr = (*C.double)(unsafe.Pointer(&result.Residuals[0]))
	}

	var rSquared C.double
	var stdError C.double

	ret := C.fc_optim_least_squares_ext(
		(*C.double)(unsafe.Pointer(&X[0])),
		(*C.double)(unsafe.Pointer(&y[0])),
		C.size_t(n),
		C.size_t(p),
		(*C.double)(unsafe.Pointer(&result.Beta[0])),
		residualsPtr,
		&rSquared,
		&stdError,
	)

	if ret == -1 {
		return nil, ErrNullPointer
	} else if ret == -2 {
		return nil, ErrInvalidDimensions
	} else if ret == -3 {
		return nil, ErrRankDeficient
	}

	result.RSquared = float64(rSquared)
	result.StdError = float64(stdError)

	return result, nil
}
