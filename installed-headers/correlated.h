/**
 * @file correlated.h
 * @brief Correlated normal random number generation
 *
 * Generates multivariate normal random variables with specified correlation structure
 * using Cholesky decomposition. For correlation matrix Σ, computes L where LL^T = Σ,
 * then transforms independent standard normals Z to correlated normals X = μ + LZ.
 */

#ifndef FC_RANDOM_CORRELATED_H
#define FC_RANDOM_CORRELATED_H

#include "error.h"
#include "platform.h"
#include <stddef.h>

FC_BEGIN_DECLS

/**
 * @brief Compute Cholesky decomposition of a correlation/covariance matrix
 *
 * Computes lower triangular matrix L such that LL^T = Σ using the
 * Cholesky-Banachiewicz algorithm. The input matrix must be symmetric
 * positive definite.
 *
 * @param n Dimension of the matrix
 * @param sigma Input correlation/covariance matrix (row-major, n×n)
 * @param L Output lower triangular Cholesky factor (row-major, n×n)
 * @return FC_OK on success, error code otherwise
 *
 * Error codes:
 * - FC_ERR_INVALID_ARG: n is 0, sigma or L is NULL, or matrix is not symmetric (debug mode)
 * - FC_ERR_NOT_POSITIVE_DEF: Matrix is not positive definite
 *
 * Time complexity: O(n³)
 * Space complexity: O(1) auxiliary
 *
 * Thread safety: Thread-safe if different output buffers are used
 *
 * Numerical stability: Includes checks for near-zero diagonal elements to avoid
 * division instability. In debug builds, verifies input matrix symmetry.
 *
 * @example
 * // Compute Cholesky factor of 2×2 correlation matrix with ρ=0.7
 * double sigma[4] = {1.0, 0.7, 0.7, 1.0};
 * double L[4];
 * fc_status_t status = fc_random_cholesky_decompose(2, sigma, L);
 * if (status != FC_OK) {
 *     // Handle error
 * }
 */
FC_API fc_status_t fc_random_cholesky_decompose(size_t n, const double* sigma, double* L);

/**
 * @brief Generate batch of correlated normal random vectors
 *
 * Generates count samples from multivariate normal distribution N(μ, Σ)
 * where Σ = LL^T. Uses the transformation X = μ + LZ where Z ~ N(0,I).
 *
 * The caller must provide:
 * - Pre-computed Cholesky factor L (use fc_random_cholesky_decompose)
 * - Buffer of independent standard normal random numbers (count × dim)
 * - Mean vector μ (or NULL for zero mean)
 *
 * @param dim Dimension of each vector
 * @param count Number of vectors to generate
 * @param L Lower triangular Cholesky factor (row-major, dim×dim)
 * @param mean Mean vector (length dim), or NULL for zero mean
 * @param z Independent standard normal samples (length count×dim)
 * @param output Output correlated normal vectors (length count×dim)
 * @return FC_OK on success, error code otherwise
 *
 * Error codes:
 * - FC_ERR_INVALID_ARG: Invalid parameters
 *
 * Time complexity: O(count × dim²)
 * Space complexity: O(1) auxiliary
 *
 * Thread safety: Thread-safe if different output buffers are used
 *
 * Performance: Uses cache-optimized blocked computation for dim > 4 to improve
 * memory locality. Typical throughput: 3-7 GB/s depending on dimension.
 *
 * Note: The caller is responsible for generating the independent normal
 * samples z using fc_random_normal_batch or equivalent.
 *
 * @example
 * // Generate 1000 samples from 2D normal with correlation 0.7
 * double sigma[4] = {1.0, 0.7, 0.7, 1.0};
 * double L[4];
 * fc_random_cholesky_decompose(2, sigma, L);
 *
 * double z[2000];  // 1000 * 2, filled with independent N(0,1) samples
 * double output[2000];
 * fc_random_correlated_normal_batch(2, 1000, L, NULL, z, output);
 */
FC_API fc_status_t fc_random_correlated_normal_batch(
    size_t dim,
    size_t count,
    const double* L,
    const double* mean,
    const double* z,
    double* output
);

/**
 * @brief One-shot generation of correlated normal samples from covariance matrix
 *
 * Convenience function that combines Cholesky decomposition and sample generation
 * in a single call. Suitable for scenarios where the Cholesky factor is not reused.
 *
 * Internally computes L = cholesky(sigma) and then generates samples X = μ + LZ.
 *
 * @param dim Dimension of each vector
 * @param count Number of vectors to generate
 * @param sigma Input correlation/covariance matrix (row-major, dim×dim)
 * @param mean Mean vector (length dim), or NULL for zero mean
 * @param z Independent standard normal samples (length count×dim)
 * @param output Output correlated normal vectors (length count×dim)
 * @param L_workspace Optional workspace for Cholesky factor (dim×dim).
 *                    If NULL, memory is allocated internally (slower).
 *                    If provided, must have size dim×dim doubles.
 * @return FC_OK on success, error code otherwise
 *
 * Error codes:
 * - FC_ERR_INVALID_ARG: Invalid parameters or matrix not symmetric (debug mode)
 * - FC_ERR_NOT_POSITIVE_DEF: Matrix is not positive definite
 * - FC_ERR_OUT_OF_MEMORY: Memory allocation failed (when L_workspace is NULL)
 *
 * Time complexity: O(dim³ + count × dim²)
 * Space complexity: O(dim²) for Cholesky factor (stack or heap depending on L_workspace)
 *
 * Performance note: If generating multiple batches with the same covariance matrix,
 * prefer calling fc_random_cholesky_decompose once and reusing L with
 * fc_random_correlated_normal_batch for better performance.
 *
 * @example
 * // One-shot generation with workspace
 * double sigma[4] = {1.0, 0.7, 0.7, 1.0};
 * double L_work[4];  // Workspace to avoid allocation
 * double z[2000];    // 1000 * 2 independent normals
 * double output[2000];
 * fc_random_correlated_normal_from_cov(2, 1000, sigma, NULL, z, output, L_work);
 */
FC_API fc_status_t fc_random_correlated_normal_from_cov(
    size_t dim,
    size_t count,
    const double* sigma,
    const double* mean,
    const double* z,
    double* output,
    double* L_workspace
);

FC_END_DECLS

#endif /* FC_RANDOM_CORRELATED_H */
