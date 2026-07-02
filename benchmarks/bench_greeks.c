#include "greeks.h"
#include "bench_framework.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    double* S;
    double* K;
    double* T;
    double* r;
    double* sigma;
    int* is_call;
    double* delta;
    double* gamma;
    double* vega;
    double* theta;
    double* rho;
    size_t n;
} bench_greeks_ctx_t;

static void bench_greeks_setup(bench_greeks_ctx_t* ctx, size_t n) {
    ctx->n = n;
    ctx->S = (double*)malloc(n * sizeof(double));
    ctx->K = (double*)malloc(n * sizeof(double));
    ctx->T = (double*)malloc(n * sizeof(double));
    ctx->r = (double*)malloc(n * sizeof(double));
    ctx->sigma = (double*)malloc(n * sizeof(double));
    ctx->is_call = (int*)malloc(n * sizeof(int));
    ctx->delta = (double*)malloc(n * sizeof(double));
    ctx->gamma = (double*)malloc(n * sizeof(double));
    ctx->vega = (double*)malloc(n * sizeof(double));
    ctx->theta = (double*)malloc(n * sizeof(double));
    ctx->rho = (double*)malloc(n * sizeof(double));

    for (size_t i = 0; i < n; i++) {
        ctx->S[i] = 100.0 + (i % 50) - 25.0;
        ctx->K[i] = 100.0;
        ctx->T[i] = 0.25 + (i % 8) * 0.125;
        ctx->r[i] = 0.05;
        ctx->sigma[i] = 0.2 + (i % 10) * 0.01;
        ctx->is_call[i] = (i % 2);
    }
}

static void bench_greeks_teardown(bench_greeks_ctx_t* ctx) {
    free(ctx->S);
    free(ctx->K);
    free(ctx->T);
    free(ctx->r);
    free(ctx->sigma);
    free(ctx->is_call);
    free(ctx->delta);
    free(ctx->gamma);
    free(ctx->vega);
    free(ctx->theta);
    free(ctx->rho);
}

static void bench_greeks_all(void* user_data) {
    bench_greeks_ctx_t* ctx = (bench_greeks_ctx_t*)user_data;
    fc_optim_greeks_batch(ctx->n, ctx->S, ctx->K, ctx->T, ctx->r, ctx->sigma, ctx->is_call,
                         ctx->delta, ctx->gamma, ctx->vega, ctx->theta, ctx->rho);
}

static void bench_greeks_delta_only(void* user_data) {
    bench_greeks_ctx_t* ctx = (bench_greeks_ctx_t*)user_data;
    fc_optim_greeks_batch(ctx->n, ctx->S, ctx->K, ctx->T, ctx->r, ctx->sigma, ctx->is_call,
                         ctx->delta, NULL, NULL, NULL, NULL);
}

static void bench_greeks_gamma_only(void* user_data) {
    bench_greeks_ctx_t* ctx = (bench_greeks_ctx_t*)user_data;
    fc_optim_greeks_batch(ctx->n, ctx->S, ctx->K, ctx->T, ctx->r, ctx->sigma, ctx->is_call,
                         NULL, ctx->gamma, NULL, NULL, NULL);
}

static void bench_greeks_vega_only(void* user_data) {
    bench_greeks_ctx_t* ctx = (bench_greeks_ctx_t*)user_data;
    fc_optim_greeks_batch(ctx->n, ctx->S, ctx->K, ctx->T, ctx->r, ctx->sigma, ctx->is_call,
                         NULL, NULL, ctx->vega, NULL, NULL);
}

static void bench_greeks_theta_only(void* user_data) {
    bench_greeks_ctx_t* ctx = (bench_greeks_ctx_t*)user_data;
    fc_optim_greeks_batch(ctx->n, ctx->S, ctx->K, ctx->T, ctx->r, ctx->sigma, ctx->is_call,
                         NULL, NULL, NULL, ctx->theta, NULL);
}

static void bench_greeks_rho_only(void* user_data) {
    bench_greeks_ctx_t* ctx = (bench_greeks_ctx_t*)user_data;
    fc_optim_greeks_batch(ctx->n, ctx->S, ctx->K, ctx->T, ctx->r, ctx->sigma, ctx->is_call,
                         NULL, NULL, NULL, NULL, ctx->rho);
}

static void run_bench_suite(const char* name, fc_bench_fn fn, size_t n) {
    bench_greeks_ctx_t ctx;
    bench_greeks_setup(&ctx, n);

    fc_bench_config_t config = FC_BENCH_CONFIG_DEFAULT;
    config.name = name;
    config.data_size = n * sizeof(double) * 5;
    config.min_time_ms = 200.0;
    config.warmup_ms = 50.0;

    fc_bench_result_t result;

    fc_bench_run(&config, fn, &ctx, &result);

    bench_greeks_teardown(&ctx);
}

void bench_greeks_run(void) {
    printf("\n--- Greeks Benchmarks ---\n");
    fc_bench_print_header();

    const size_t sizes[] = {100, 1000, 5000, 10000};
    const size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];
        char name[128];

        snprintf(name, sizeof(name), "greeks_all_%zu", n);
        run_bench_suite(name, bench_greeks_all, n);

        snprintf(name, sizeof(name), "greeks_delta_only_%zu", n);
        run_bench_suite(name, bench_greeks_delta_only, n);

        snprintf(name, sizeof(name), "greeks_gamma_only_%zu", n);
        run_bench_suite(name, bench_greeks_gamma_only, n);

        snprintf(name, sizeof(name), "greeks_vega_only_%zu", n);
        run_bench_suite(name, bench_greeks_vega_only, n);

        snprintf(name, sizeof(name), "greeks_theta_only_%zu", n);
        run_bench_suite(name, bench_greeks_theta_only, n);

        snprintf(name, sizeof(name), "greeks_rho_only_%zu", n);
        run_bench_suite(name, bench_greeks_rho_only, n);
    }
}
