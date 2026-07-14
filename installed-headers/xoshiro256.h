/**
 * @file xoshiro256.h
 * @brief xoshiro256** pseudo-random number generator
 *
 * Implementation of the xoshiro256** generator by Blackman and Vigna (2018).
 * This is a high-quality, fast PRNG suitable for scientific simulations and
 * Monte Carlo methods. It has a period of 2^256 - 1 and passes all statistical
 * tests including BigCrush.
 *
 * Features:
 * - 256-bit state space
 * - Period: 2^256 - 1
 * - Passes TestU01 BigCrush
 * - SIMD-optimized batch generation (AVX2/SSE4.2)
 * - Jump functions for parallel streams
 */

#ifndef FC_RANDOM_XOSHIRO256_H
#define FC_RANDOM_XOSHIRO256_H

#include "error.h"
#include "platform.h"
#include <stddef.h>
#include <stdint.h>

FC_BEGIN_DECLS

/**
 * @brief xoshiro256** generator state
 *
 * State consists of four 64-bit integers. Must be initialized with
 * non-zero values (at least one element must be non-zero).
 */
typedef struct {
    uint64_t s[4];
} fc_xoshiro256_state_t;

/**
 * @brief Initialize xoshiro256** state from a seed
 *
 * Seeds the generator using SplitMix64 to ensure good initial state distribution.
 * Any 64-bit seed value is acceptable, including zero.
 *
 * @param state Generator state to initialize
 * @param seed 64-bit seed value
 * @return FC_OK on success, FC_ERR_INVALID_ARG if state is NULL
 *
 * Thread safety: Not thread-safe, state must not be shared between threads
 *                without external synchronization
 *
 * @example
 * fc_xoshiro256_state_t rng;
 * fc_xoshiro256_init(&rng, 12345);
 */
FC_API fc_status_t fc_xoshiro256_init(fc_xoshiro256_state_t* state, uint64_t seed);

/**
 * @brief Generate a single 64-bit random number
 *
 * Generates one pseudo-random uint64_t using the xoshiro256** algorithm.
 *
 * @param state Generator state
 * @param output Pointer to output variable
 * @return FC_OK on success, FC_ERR_INVALID_ARG if parameters are NULL
 *
 * Time complexity: O(1)
 * Performance: ~1-2 ns per call on modern CPUs
 *
 * Thread safety: Not thread-safe, each thread should have its own state
 *
 * @example
 * uint64_t value;
 * fc_xoshiro256_next(&rng, &value);
 */
FC_API fc_status_t fc_xoshiro256_next(fc_xoshiro256_state_t* state, uint64_t* output);

/**
 * @brief Generate batch of random uint64_t values
 *
 * Generates n pseudo-random uint64_t values. Uses SIMD optimization when
 * available (AVX2/SSE4.2) for processing multiple streams in parallel.
 *
 * @param state Generator state
 * @param n Number of values to generate
 * @param output Output buffer (length n)
 * @return FC_OK on success, error code otherwise
 *
 * Error codes:
 * - FC_ERR_INVALID_ARG: state or output is NULL, or n is 0
 *
 * Time complexity: O(n)
 * Performance: ~0.3-0.5 ns per value (AVX2), ~10 billion/sec
 *
 * Thread safety: Not thread-safe, each thread should have its own state
 *
 * @example
 * uint64_t values[10000];
 * fc_xoshiro256_batch(&rng, 10000, values);
 */
FC_API fc_status_t fc_xoshiro256_batch(fc_xoshiro256_state_t* state, size_t n, uint64_t* output);

/**
 * @brief Generate batch of uniform random doubles in [0, 1)
 *
 * Generates n uniform random doubles in the range [0, 1) using the
 * upper 53 bits of each uint64_t for full IEEE 754 double precision.
 *
 * @param state Generator state
 * @param n Number of values to generate
 * @param output Output buffer (length n)
 * @return FC_OK on success, error code otherwise
 *
 * Error codes:
 * - FC_ERR_INVALID_ARG: state or output is NULL, or n is 0
 *
 * Time complexity: O(n)
 * Performance: ~0.4-0.6 ns per value (AVX2)
 *
 * Thread safety: Not thread-safe, each thread should have its own state
 *
 * Range: [0, 1) with 53 bits of precision (all representable doubles)
 *
 * @example
 * double uniforms[10000];
 * fc_xoshiro256_uniform(&rng, 10000, uniforms);
 */
FC_API fc_status_t fc_xoshiro256_uniform(fc_xoshiro256_state_t* state, size_t n, double* output);

/**
 * @brief Jump function for parallel stream generation
 *
 * Advances the generator state by 2^128 calls, equivalent to generating
 * 2^128 random numbers. This is useful for generating independent parallel
 * streams: initialize one generator, jump to create streams for other threads.
 *
 * @param state Generator state to advance
 * @return FC_OK on success, FC_ERR_INVALID_ARG if state is NULL
 *
 * Time complexity: O(1) - constant time regardless of jump distance
 * Performance: ~100-200 ns per jump
 *
 * Thread safety: Not thread-safe on the same state
 *
 * Use case: Create 2^128 non-overlapping streams for parallel computation
 *
 * @example
 * // Create independent streams for 4 threads
 * fc_xoshiro256_state_t rng_main, rng_thread[4];
 * fc_xoshiro256_init(&rng_main, 42);
 * for (int i = 0; i < 4; i++) {
 *     rng_thread[i] = rng_main;
 *     fc_xoshiro256_jump(&rng_main);  // Jump to next stream
 * }
 */
FC_API fc_status_t fc_xoshiro256_jump(fc_xoshiro256_state_t* state);

/**
 * @brief Long jump function for massive parallel stream generation
 *
 * Advances the generator state by 2^192 calls. This allows creating
 * 2^64 independent streams, each with 2^128 period before overlap.
 *
 * @param state Generator state to advance
 * @return FC_OK on success, FC_ERR_INVALID_ARG if state is NULL
 *
 * Time complexity: O(1)
 * Performance: ~100-200 ns per jump
 *
 * Use case: Create massive number of independent streams (e.g., distributed computing)
 *
 * @example
 * // Create streams for distributed cluster (64 nodes)
 * fc_xoshiro256_state_t rng_master, rng_node[64];
 * fc_xoshiro256_init(&rng_master, 12345);
 * for (int i = 0; i < 64; i++) {
 *     rng_node[i] = rng_master;
 *     fc_xoshiro256_long_jump(&rng_master);
 * }
 */
FC_API fc_status_t fc_xoshiro256_long_jump(fc_xoshiro256_state_t* state);

FC_END_DECLS

#endif /* FC_RANDOM_XOSHIRO256_H */
