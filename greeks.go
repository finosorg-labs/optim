package optim

/*
#include "greeks.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// GreeksBatch computes Black-Scholes Greeks for a batch of options.
//
// Calculates Delta, Gamma, Vega, Theta, and Rho for European options.
// Any output slice can be nil if that Greek is not needed.
//
// Parameters:
//   - S: Spot prices (must be > 0)
//   - K: Strike prices (must be > 0)
//   - T: Times to expiration in years (must be > 0)
//   - r: Risk-free rates
//   - sigma: Volatilities (must be > 0)
//   - isCall: Option types (true for call, false for put)
//   - delta: Output Delta values (can be nil)
//   - gamma: Output Gamma values (can be nil)
//   - vega: Output Vega values (can be nil)
//   - theta: Output Theta values (can be nil)
//   - rho: Output Rho values (can be nil)
//
// Returns an error if:
//   - Input slices have different lengths
//   - Any input slice is empty
//   - All output slices are nil
//   - Any input value is invalid (S/K/sigma <= 0, T <= 0)
//
// Performance: 5000 options in < 1ms on modern x86_64 CPU with AVX2
func GreeksBatch(
	S, K, T, r, sigma []float64,
	isCall []bool,
	delta, gamma, vega, theta, rho []float64,
) error {
	n := len(S)
	if n == 0 {
		return fmt.Errorf("input slices cannot be empty")
	}

	if len(K) != n || len(T) != n || len(r) != n || len(sigma) != n || len(isCall) != n {
		return fmt.Errorf("all input slices must have the same length")
	}

	if delta == nil && gamma == nil && vega == nil && theta == nil && rho == nil {
		return fmt.Errorf("at least one output slice must be non-nil")
	}

	if delta != nil && len(delta) != n {
		return fmt.Errorf("delta slice length mismatch")
	}
	if gamma != nil && len(gamma) != n {
		return fmt.Errorf("gamma slice length mismatch")
	}
	if vega != nil && len(vega) != n {
		return fmt.Errorf("vega slice length mismatch")
	}
	if theta != nil && len(theta) != n {
		return fmt.Errorf("theta slice length mismatch")
	}
	if rho != nil && len(rho) != n {
		return fmt.Errorf("rho slice length mismatch")
	}

	isCallInt := make([]C.int, n)
	for i, v := range isCall {
		if v {
			isCallInt[i] = 1
		} else {
			isCallInt[i] = 0
		}
	}

	var deltaPtr, gammaPtr, vegaPtr, thetaPtr, rhoPtr *C.double
	if delta != nil {
		deltaPtr = (*C.double)(unsafe.Pointer(&delta[0]))
	}
	if gamma != nil {
		gammaPtr = (*C.double)(unsafe.Pointer(&gamma[0]))
	}
	if vega != nil {
		vegaPtr = (*C.double)(unsafe.Pointer(&vega[0]))
	}
	if theta != nil {
		thetaPtr = (*C.double)(unsafe.Pointer(&theta[0]))
	}
	if rho != nil {
		rhoPtr = (*C.double)(unsafe.Pointer(&rho[0]))
	}

	ret := C.fc_optim_greeks_batch(
		C.size_t(n),
		(*C.double)(unsafe.Pointer(&S[0])),
		(*C.double)(unsafe.Pointer(&K[0])),
		(*C.double)(unsafe.Pointer(&T[0])),
		(*C.double)(unsafe.Pointer(&r[0])),
		(*C.double)(unsafe.Pointer(&sigma[0])),
		(*C.int)(unsafe.Pointer(&isCallInt[0])),
		deltaPtr,
		gammaPtr,
		vegaPtr,
		thetaPtr,
		rhoPtr,
	)

	if ret != 0 {
		switch ret {
		case -1:
			return fmt.Errorf("required input pointer is NULL")
		case -2:
			return fmt.Errorf("size is zero")
		case -3:
			return fmt.Errorf("invalid input value (S/K/sigma <= 0 or T <= 0)")
		default:
			return fmt.Errorf("unknown error: %d", ret)
		}
	}

	return nil
}
