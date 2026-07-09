/**
 * @file bench_var_parametric.c
 * @brief Benchmarks for parametric VaR/CVaR calculation
 */

#include "var_parametric.h"
#include "bench_framework.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    size_t n;
    double* means;
    double* stddevs;
    double confidence;
    double* var;
    double* cvar;
} bench_var_data_t;

static void bench_var_parametric_single(void* user_data) {
    bench_var_data_t* data = (bench_var_data_t*)user_data;
    double var, cvar;
    fc_optim_var_parametric(data->means[0], data->stddevs[0], data->confidence, &var, &cvar);
}

static void bench_var_parametric_batch_100(void* user_data) {
    bench_var_data_t* data = (bench_var_data_t*)user_data;
    fc_optim_var_parametric_batch(100, data->means, data->stddevs, data->confidence, data->var, data->cvar);
}

static void bench_var_parametric_batch_500(void* user_data) {
    bench_var_data_t* data = (bench_var_data_t*)user_data;
    fc_optim_var_parametric_batch(500, data->means, data->stddevs, data->confidence, data->var, data->cvar);
}

static void bench_var_parametric_batch_1000(void* user_data) {
    bench_var_data_t* data = (bench_var_data_t*)user_data;
    fc_optim_var_parametric_batch(1000, data->means, data->stddevs, data->confidence, data->var, data->cvar);
}

static void bench_var_parametric_batch_5000(void* user_data) {
    bench_var_data_t* data = (bench_var_data_t*)user_data;
    fc_optim_var_parametric_batch(5000, data->means, data->stddevs, data->confidence, data->var, data->cvar);
}

static bench_var_data_t* create_bench_data(size_t n) {
    bench_var_data_t* data = malloc(sizeof(bench_var_data_t));
    data->n = n;
    data->means = malloc(n * sizeof(double));
    data->stddevs = malloc(n * sizeof(double));
    data->var = malloc(n * sizeof(double));
    data->cvar = malloc(n * sizeof(double));
    data->confidence = 0.95;

    for (size_t i = 0; i < n; i++) {
        data->means[i] = 0.001 * ((double)(i % 10) - 5.0);
        data->stddevs[i] = 0.01 + 0.001 * (i % 20);
    }

    return data;
}

static void destroy_bench_data(bench_var_data_t* data) {
    if (data) {
        free(data->means);
        free(data->stddevs);
        free(data->var);
        free(data->cvar);
        free(data);
    }
}

void bench_var_parametric_run(void) {
    printf("\n--- Parametric VaR/CVaR Benchmarks ---\n");
    fc_bench_print_header();

    fc_bench_config_t config = FC_BENCH_CONFIG_DEFAULT;
    fc_bench_result_t result;

    config.name = "var_parametric_single";
    config.data_size = sizeof(double) * 5;
    bench_var_data_t* data1 = create_bench_data(1);
    fc_bench_run(&config, bench_var_parametric_single, data1, &result);
    fc_bench_result_print(&result);
    destroy_bench_data(data1);

    config.name = "var_parametric_batch_100";
    config.data_size = sizeof(double) * 100 * 4;
    bench_var_data_t* data100 = create_bench_data(5000);
    fc_bench_run(&config, bench_var_parametric_batch_100, data100, &result);
    fc_bench_result_print(&result);

    config.name = "var_parametric_batch_500";
    config.data_size = sizeof(double) * 500 * 4;
    fc_bench_run(&config, bench_var_parametric_batch_500, data100, &result);
    fc_bench_result_print(&result);

    config.name = "var_parametric_batch_1000";
    config.data_size = sizeof(double) * 1000 * 4;
    fc_bench_run(&config, bench_var_parametric_batch_1000, data100, &result);
    fc_bench_result_print(&result);

    config.name = "var_parametric_batch_5000";
    config.data_size = sizeof(double) * 5000 * 4;
    config.min_time_ms = 200.0;
    fc_bench_run(&config, bench_var_parametric_batch_5000, data100, &result);
    fc_bench_result_print(&result);

    destroy_bench_data(data100);
}
