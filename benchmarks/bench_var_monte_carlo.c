/**
 * @file bench_var_monte_carlo.c
 * @brief Benchmarks for Monte Carlo VaR/CVaR calculation
 */

#include "var_monte_carlo.h"
#include "bench_framework.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct {
    fc_var_mc_state_t* state;
    double* means;
    double* cov_matrix;
    double* weights;
    size_t dim;
    double confidence;
    double var;
    double cvar;
} bench_mc_data_t;

/* Global volatile sink to prevent compiler optimization */
static volatile double g_sink = 0.0;

static void bench_var_mc_10x1000(void* user_data) {
    bench_mc_data_t* data = (bench_mc_data_t*)user_data;
    fc_optim_var_monte_carlo(data->state, data->means, data->cov_matrix,
                             data->weights, data->dim, data->confidence,
                             &data->var, &data->cvar);
    g_sink = data->var + data->cvar;
}

static void bench_var_mc_50x5000(void* user_data) {
    bench_mc_data_t* data = (bench_mc_data_t*)user_data;
    fc_optim_var_monte_carlo(data->state, data->means, data->cov_matrix,
                             data->weights, data->dim, data->confidence,
                             &data->var, &data->cvar);
    g_sink = data->var + data->cvar;
}

static void bench_var_mc_100x10000(void* user_data) {
    bench_mc_data_t* data = (bench_mc_data_t*)user_data;
    fc_optim_var_monte_carlo(data->state, data->means, data->cov_matrix,
                             data->weights, data->dim, data->confidence,
                             &data->var, &data->cvar);
    g_sink = data->var + data->cvar;
}

static void bench_var_mc_500x10000(void* user_data) {
    bench_mc_data_t* data = (bench_mc_data_t*)user_data;
    fc_optim_var_monte_carlo(data->state, data->means, data->cov_matrix,
                             data->weights, data->dim, data->confidence,
                             &data->var, &data->cvar);
    g_sink = data->var + data->cvar;
}

static bench_mc_data_t* create_mc_data(size_t dim, size_t n_paths) {
    bench_mc_data_t* data = malloc(sizeof(bench_mc_data_t));
    data->state = fc_optim_var_monte_carlo_state_create(dim, n_paths, 42);
    data->means = malloc(dim * sizeof(double));
    data->cov_matrix = malloc(dim * dim * sizeof(double));
    data->weights = malloc(dim * sizeof(double));
    data->dim = dim;
    data->confidence = 0.95;

    for (size_t i = 0; i < dim; i++) {
        data->means[i] = 0.001 * ((double)(i % 10) - 5.0);
        data->weights[i] = 1.0 / dim;
    }

    // Create a valid positive definite covariance matrix
    // Use a simple diagonal dominant matrix with small off-diagonal correlations
    double base_vol = 0.02;  // 2% volatility
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            if (i == j) {
                data->cov_matrix[i * dim + j] = base_vol * base_vol;  // variance
            } else {
                // Small correlation that ensures positive definiteness
                double corr = 0.1 * (1.0 / (1.0 + abs((int)i - (int)j)));
                data->cov_matrix[i * dim + j] = corr * base_vol * base_vol;
            }
        }
    }

    return data;
}

static void destroy_mc_data(bench_mc_data_t* data) {
    if (data) {
        fc_optim_var_monte_carlo_state_destroy(data->state);
        free(data->means);
        free(data->cov_matrix);
        free(data->weights);
        free(data);
    }
}

void bench_var_monte_carlo_run(void) {
    printf("\n--- Monte Carlo VaR/CVaR Benchmarks ---\n");
    fc_bench_print_header();

    fc_bench_config_t config = FC_BENCH_CONFIG_DEFAULT;
    fc_bench_result_t result;

    config.name = "var_mc_10x1000";
    config.data_size = sizeof(double) * 1000 * 10;
    bench_mc_data_t* data_10_1k = create_mc_data(10, 1000);
    fc_bench_run(&config, bench_var_mc_10x1000, data_10_1k, &result);
    fc_bench_result_print(&result);
    destroy_mc_data(data_10_1k);

    config.name = "var_mc_50x5000";
    config.data_size = sizeof(double) * 5000 * 50;
    config.min_time_ms = 200.0;
    bench_mc_data_t* data_50_5k = create_mc_data(50, 5000);
    fc_bench_run(&config, bench_var_mc_50x5000, data_50_5k, &result);
    fc_bench_result_print(&result);
    destroy_mc_data(data_50_5k);

    config.name = "var_mc_100x10000";
    config.data_size = sizeof(double) * 10000 * 100;
    config.min_time_ms = 300.0;
    bench_mc_data_t* data_100_10k = create_mc_data(100, 10000);
    fc_bench_run(&config, bench_var_mc_100x10000, data_100_10k, &result);
    fc_bench_result_print(&result);
    destroy_mc_data(data_100_10k);

    config.name = "var_mc_500x10000";
    config.data_size = sizeof(double) * 10000 * 500;
    config.min_time_ms = 500.0;
    config.min_iterations = 3;
    bench_mc_data_t* data_500_10k = create_mc_data(500, 10000);
    fc_bench_run(&config, bench_var_mc_500x10000, data_500_10k, &result);
    fc_bench_result_print(&result);
    destroy_mc_data(data_500_10k);
}
