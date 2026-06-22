/**
 * @file bench_main.c
 * @brief Benchmark runner entry point for optim module
 */

#include "bench_framework.h"
#include <platform.h>
#include <simd_detect.h>
#include <stdio.h>

/* External benchmark suites */
extern void bench_optim_run(void);

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    fc_init();

    fc_bench_init();

    printf("optim performance benchmarks v%s\n", FC_BENCH_VERSION);

    bench_optim_run();

    fc_bench_cleanup();
    return 0;
}
