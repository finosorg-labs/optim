#include "least_squares.h"
#include "bench_framework.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    double* X;
    double* X_backup;
    double* y;
    double* beta;
    double* residuals;
    double* work;
    size_t n;
    size_t p;
    size_t batch_size;
} bench_ls_ctx_t;

static void bench_ls_setup(bench_ls_ctx_t* ctx, size_t n, size_t p) {
    ctx->n = n;
    ctx->p = p;
    ctx->X = (double*)malloc(n * p * sizeof(double));
    ctx->X_backup = (double*)malloc(n * p * sizeof(double));
    ctx->y = (double*)malloc(n * sizeof(double));
    ctx->beta = (double*)malloc(p * sizeof(double));
    ctx->work = (double*)malloc((n + p) * sizeof(double));

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < p; j++) {
            ctx->X[j * n + i] = (double)(i * p + j + 1);
        }
        ctx->y[i] = (double)(i + 1);
    }
    memcpy(ctx->X_backup, ctx->X, n * p * sizeof(double));
}

static void bench_ls_batch_setup(bench_ls_ctx_t* ctx, size_t n, size_t p, size_t batch_size) {
    ctx->n = n;
    ctx->p = p;
    ctx->batch_size = batch_size;
    ctx->X = (double*)malloc(batch_size * n * p * sizeof(double));
    ctx->X_backup = (double*)malloc(batch_size * n * p * sizeof(double));
    ctx->y = (double*)malloc(batch_size * n * sizeof(double));
    ctx->beta = (double*)malloc(batch_size * p * sizeof(double));
    ctx->work = NULL;  // Batch function allocates its own workspace

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < p; j++) {
                ctx->X[b * n * p + j * n + i] = (double)(b * n * p + i * p + j + 1);
            }
            ctx->y[b * n + i] = (double)(b * n + i + 1);
        }
    }
    memcpy(ctx->X_backup, ctx->X, batch_size * n * p * sizeof(double));
}

static void bench_ls_ext_setup(bench_ls_ctx_t* ctx, size_t n, size_t p) {
    bench_ls_setup(ctx, n, p);
    ctx->residuals = (double*)malloc(n * sizeof(double));
}

static void bench_ls_teardown(bench_ls_ctx_t* ctx) {
    free(ctx->X);
    free(ctx->X_backup);
    free(ctx->y);
    free(ctx->beta);
    if (ctx->work) {
        free(ctx->work);
    }
}

static void bench_ls_ext_teardown(bench_ls_ctx_t* ctx) {
    bench_ls_teardown(ctx);
    free(ctx->residuals);
}

static void bench_ls_reset(bench_ls_ctx_t* ctx) {
    memcpy(ctx->X, ctx->X_backup, ctx->n * ctx->p * sizeof(double));
}

static void bench_ls_batch_reset(bench_ls_ctx_t* ctx) {
    memcpy(ctx->X, ctx->X_backup, ctx->batch_size * ctx->n * ctx->p * sizeof(double));
}

static void bench_least_squares(void* user_data) {
    bench_ls_ctx_t* ctx = (bench_ls_ctx_t*)user_data;
    bench_ls_reset(ctx);
    fc_optim_least_squares_work(ctx->X, ctx->y, ctx->n, ctx->p, ctx->beta, ctx->work);
}

static void bench_least_squares_batch(void* user_data) {
    bench_ls_ctx_t* ctx = (bench_ls_ctx_t*)user_data;
    bench_ls_batch_reset(ctx);
    fc_optim_least_squares_batch(ctx->X, ctx->y, ctx->n, ctx->p, ctx->batch_size, ctx->beta);
}

static void bench_least_squares_ext(void* user_data) {
    bench_ls_ctx_t* ctx = (bench_ls_ctx_t*)user_data;
    bench_ls_reset(ctx);
    double r_squared, std_error;
    fc_optim_least_squares_ext(ctx->X, ctx->y, ctx->n, ctx->p, ctx->beta, ctx->residuals, &r_squared,
                                &std_error);
}

static void run_bench_suite_single(const char* name, size_t n, size_t p) {
    bench_ls_ctx_t ctx;
    bench_ls_setup(&ctx, n, p);

    fc_bench_config_t config = FC_BENCH_CONFIG_DEFAULT;
    config.name = name;
    config.data_size = n * p * sizeof(double);
    config.min_time_ms = 200.0;
    config.warmup_ms = 50.0;

    fc_bench_result_t result;

    fc_bench_run(&config, bench_least_squares, &ctx, &result);

    bench_ls_teardown(&ctx);
}

static void run_bench_suite_batch(const char* name, size_t n, size_t p, size_t batch_size) {
    bench_ls_ctx_t ctx;
    bench_ls_batch_setup(&ctx, n, p, batch_size);

    fc_bench_config_t config = FC_BENCH_CONFIG_DEFAULT;
    config.name = name;
    config.data_size = batch_size * n * p * sizeof(double);
    config.min_time_ms = 200.0;
    config.warmup_ms = 50.0;

    fc_bench_result_t result;

    fc_bench_run(&config, bench_least_squares_batch, &ctx, &result);

    bench_ls_teardown(&ctx);
}

static void run_bench_suite_ext(const char* name, size_t n, size_t p) {
    bench_ls_ctx_t ctx;
    bench_ls_ext_setup(&ctx, n, p);

    fc_bench_config_t config = FC_BENCH_CONFIG_DEFAULT;
    config.name = name;
    config.data_size = n * p * sizeof(double);
    config.min_time_ms = 200.0;
    config.warmup_ms = 50.0;

    fc_bench_result_t result;

    fc_bench_run(&config, bench_least_squares_ext, &ctx, &result);

    bench_ls_ext_teardown(&ctx);
}

void bench_least_squares_run(void) {
    printf("\nLeast Squares Regression Benchmarks\n");
    printf("===================================\n\n");

    printf("Single regression:\n");
    run_bench_suite_single("ls_100x5", 100, 5);
    run_bench_suite_single("ls_1Kx10", 1000, 10);
    run_bench_suite_single("ls_5Kx20", 5000, 20);
    run_bench_suite_single("ls_10Kx50", 10000, 50);

    printf("\nBatch regression:\n");
    run_bench_suite_batch("ls_batch_50x3x100", 50, 3, 100);
    run_bench_suite_batch("ls_batch_100x5x50", 100, 5, 50);
    run_bench_suite_batch("ls_batch_500x10x10", 500, 10, 10);

    printf("\nExtended regression (with statistics):\n");
    run_bench_suite_ext("ls_ext_1Kx10", 1000, 10);
    run_bench_suite_ext("ls_ext_5Kx20", 5000, 20);
}
