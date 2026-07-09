/**
 * @file var_parametric.h
 * @brief Parametric VaR/CVaR calculation using normal distribution assumption
 *
 * Implements Value at Risk (VaR) and Conditional Value at Risk (CVaR) using
 * the parametric method, which assumes returns follow a normal distribution.
 * This method uses analytical formulas based on mean and standard deviation.
 */

#ifndef FC_OPTIM_VAR_PARAMETRIC_H
#define FC_OPTIM_VAR_PARAMETRIC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate parametric VaR/CVaR for a single portfolio
 *
 * Computes VaR and CVaR assuming portfolio returns follow a normal distribution.
 * Uses analytical formulas:
 * - VaR_α = μ - σ·Φ^(-1)(1-α)
 * - CVaR_α = μ - σ·φ(Φ^(-1)(1-α))/(1-α)
 *
 * Where:
 * - μ is the portfolio mean return
 * - σ is the portfolio standard deviation
 * - Φ^(-1) is the inverse normal CDF
 * - φ is the normal PDF
 * - α is the confidence level (e.g., 0.95 for 95% confidence)
 *
 * @param mean Portfolio mean return
 * @param stddev Portfolio standard deviation (must be > 0)
 * @param confidence Confidence level (must be in (0, 1), typically 0.95 or 0.99)
 * @param var Output: Value at Risk (negative value indicates loss)
 * @param cvar Output: Conditional Value at Risk (negative value indicates loss)
 * @return 0 on success
 *         -1 if var or cvar is NULL
 *         -2 if stddev <= 0
 *         -3 if confidence is not in (0, 1)
 *
 * Time complexity: O(1)
 * Space complexity: O(1)
 * Thread safety: Thread-safe
 *
 * @note VaR and CVaR are reported as returns (not losses), so negative values
 *       indicate potential losses
 * @note Precision: Relative error < 1e-10 compared to analytical solution
 */
int fc_optim_var_parametric(
    double mean,
    double stddev,
    double confidence,
    double* var,
    double* cvar
);

/**
 * @brief Calculate parametric VaR/CVaR for multiple portfolios
 *
 * Batch version that computes VaR/CVaR for n portfolios in parallel,
 * leveraging SIMD optimizations in the normal inverse CDF calculations.
 *
 * @param n Number of portfolios (must be > 0)
 * @param means Portfolio mean returns (n elements, can contain any finite values)
 * @param stddevs Portfolio standard deviations (n elements, all must be > 0)
 * @param confidence Confidence level (must be in (0, 1))
 * @param var Output: VaR values (n elements, pre-allocated by caller)
 * @param cvar Output: CVaR values (n elements, pre-allocated by caller)
 * @return 0 on success
 *         -1 if any required pointer is NULL
 *         -2 if n == 0
 *         -3 if any stddev <= 0 or confidence not in (0, 1)
 *
 * Time complexity: O(n)
 * Space complexity: O(n) temporary space
 * Thread safety: Thread-safe if different output buffers are used
 *
 * @note Performance target: 500 portfolios in < 5ms on modern x86_64 with AVX2
 * @note Uses SIMD-optimized normal inverse CDF from math module
 */
int fc_optim_var_parametric_batch(
    size_t n,
    const double* means,
    const double* stddevs,
    double confidence,
    double* var,
    double* cvar
);

#ifdef __cplusplus
}
#endif

#endif /* FC_OPTIM_VAR_PARAMETRIC_H */
