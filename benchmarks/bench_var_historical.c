/**
 * @file bench_var_historical.c
 * @brief Benchmarks for historical simulation VaR/CVaR calculation
 */

#include "var_historical.h"
#include "bench_framework.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    double* returns;
    size_t n;
    double confidence;
    double var;
    double cvar;
} bench_hist_data_t;

typedef struct {
    double* returns;
    double* weights;
    size_t m;
    size_t n;
    double confidence;
    double var;
    double cvar;
} bench_hist_portfolio_data_t;

static void bench_var_historical_100(void* user_data) {
    bench_hist_data_t* data = (bench_hist_data_t*)user_data;
    fc_optim_var_historical(data->returns, 100, data->confidence, &data->var, &data->cvar);
}

static void bench_var_historical_1000(void* user_data) {
    bench_hist_data_t* data = (bench_hist_data_t*)user_data;
    fc_optim_var_historical(data->returns, 1000, data->confidence, &data->var, &data->cvar);
}

static void bench_var_historical_5000(void* user_data) {
    bench_hist_data_t* data = (bench_hist_data_t*)user_data;
    fc_optim_var_historical(data->returns, 5000, data->confidence, &data->var, &data->cvar);
}

static void bench_var_historical_portfolio_500x1000(void* user_data) {
    bench_hist_portfolio_data_t* data = (bench_hist_portfolio_data_t*)user_data;
    fc_optim_var_historical_portfolio(data->returns, data->weights, data->m, data->n,
                                      data->confidence, &data->var, &data->cvar);
}

static bench_hist_data_t* create_hist_data(size_t n) {
    bench_hist_data_t* data = malloc(sizeof(bench_hist_data_t));
    data->returns = malloc(n * sizeof(double));
    data->n = n;
    data->confidence = 0.95;

    for (size_t i = 0; i < n; i++) {
        data->returns[i] = 0.02 * ((double)(i % 100) - 50.0) / 50.0;
    }

    return data;
}

static void destroy_hist_data(bench_hist_data_t* data) {
    if (data) {
        free(data->returns);
        free(data);
    }
}

static bench_hist_portfolio_data_t* create_portfolio_data(size_t m, size_t n) {
    bench_hist_portfolio_data_t* data = malloc(sizeof(bench_hist_portfolio_data_t));
    data->returns = malloc(m * n * sizeof(double));
    data->weights = malloc(m * sizeof(double));
    data->m = m;
    data->n = n;
    data->confidence = 0.95;

    for (size_t i = 0; i < m * n; i++) {
        data->returns[i] = 0.02 * ((double)(i % 100) - 50.0) / 50.0;
    }

    for (size_t i = 0; i < m; i++) {
        data->weights[i] = 1.0 / m;
    }

    return data;
}

static void destroy_portfolio_data(bench_hist_portfolio_data_t* data) {
    if (data) {
        free(data->returns);
        free(data->weights);
        free(data);
    }
}

void bench_var_historical_run(void) {
    printf("\n--- Historical Simulation VaR/CVaR Benchmarks ---\n");
    fc_bench_print_header();

    fc_bench_config_t config = FC_BENCH_CONFIG_DEFAULT;
    fc_bench_result_t result;

    config.name = "var_historical_100";
    config.data_size = sizeof(double) * 100;
    bench_hist_data_t* data100 = create_hist_data(5000);
    fc_bench_run(&config, bench_var_historical_100, data100, &result);
    fc_bench_result_print(&result);

    config.name = "var_historical_1000";
    config.data_size = sizeof(double) * 1000;
    fc_bench_run(&config, bench_var_historical_1000, data100, &result);
    fc_bench_result_print(&result);

    config.name = "var_historical_5000";
    config.data_size = sizeof(double) * 5000;
    config.min_time_ms = 200.0;
    fc_bench_run(&config, bench_var_historical_5000, data100, &result);
    fc_bench_result_print(&result);

    destroy_hist_data(data100);

    config.name = "var_historical_portfolio_500x1000";
    config.data_size = sizeof(double) * 500 * 1000;
    config.min_time_ms = 200.0;
    bench_hist_portfolio_data_t* portfolio_data = create_portfolio_data(500, 1000);
    fc_bench_run(&config, bench_var_historical_portfolio_500x1000, portfolio_data, &result);
    fc_bench_result_print(&result);

    destroy_portfolio_data(portfolio_data);
}
