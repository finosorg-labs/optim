/**
 * @file bench_optim.c
 * @brief optim module benchmark entry point
 *
 * This file serves as the main benchmark registration point for the optim module.
 * Individual benchmark modules are in separate files:
 */

#include "bench_framework.h"
#include <simd_detect.h>
#include <stdio.h>

/* External benchmark functions from sub-modules */
extern void bench_least_squares_run(void);
extern void bench_greeks_run(void);
extern void bench_var_parametric_run(void);
extern void bench_var_historical_run(void);
extern void bench_var_monte_carlo_run(void);

/* Entry point for optim benchmarks */
void bench_optim_run(void) {
    printf("\n");
    printf("============================================================\n");
    printf("  optim Module Performance Benchmarks\n");
    printf("  SIMD level: %s\n", fc_simd_level_string(fc_get_simd_level()));
    printf("============================================================\n");

    /* Run all sub-module benchmarks */
    bench_least_squares_run();
    bench_greeks_run();
    bench_var_parametric_run();
    bench_var_historical_run();
    bench_var_monte_carlo_run();

    printf("\n");
    printf("============================================================\n");
    printf("  optim benchmarks complete\n");
    printf("============================================================\n");
}
