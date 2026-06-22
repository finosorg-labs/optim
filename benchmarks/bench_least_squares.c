#include "least_squares.h"
#include "bench_framework.h"
#include <stdlib.h>
#include <string.h>

static void bench_least_squares_small(void) {
    size_t n = 100;
    size_t p = 5;

    double* X = (double*)malloc(n * p * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    double beta[5];

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < p; j++) {
            X[j * n + i] = (double)(i * p + j + 1);
        }
        y[i] = (double)(i + 1);
    }

    fc_bench_time_t start = fc_bench_time_now();

    for (int iter = 0; iter < 1000; iter++) {
        double* X_copy = (double*)malloc(n * p * sizeof(double));
        memcpy(X_copy, X, n * p * sizeof(double));
        fc_optim_least_squares(X_copy, y, n, p, beta);
        free(X_copy);
    }

    fc_bench_time_t end = fc_bench_time_now();
    double elapsed_ms = fc_bench_time_elapsed_ms(&start, &end);

    printf("  least_squares_small (n=100, p=5): %.2f ms (1000 iterations, %.2f us/iter)\n",
           elapsed_ms, elapsed_ms * 1000.0 / 1000.0);

    free(X);
    free(y);
}

static void bench_least_squares_medium(void) {
    size_t n = 1000;
    size_t p = 10;

    double* X = (double*)malloc(n * p * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    double* beta = (double*)malloc(p * sizeof(double));

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < p; j++) {
            X[j * n + i] = (double)(i * p + j + 1);
        }
        y[i] = (double)(i + 1);
    }

    fc_bench_time_t start = fc_bench_time_now();

    for (int iter = 0; iter < 100; iter++) {
        double* X_copy = (double*)malloc(n * p * sizeof(double));
        memcpy(X_copy, X, n * p * sizeof(double));
        fc_optim_least_squares(X_copy, y, n, p, beta);
        free(X_copy);
    }

    fc_bench_time_t end = fc_bench_time_now();
    double elapsed_ms = fc_bench_time_elapsed_ms(&start, &end);

    printf("  least_squares_medium (n=1000, p=10): %.2f ms (100 iterations, %.2f ms/iter)\n",
           elapsed_ms, elapsed_ms / 100.0);

    free(X);
    free(y);
    free(beta);
}

static void bench_least_squares_large(void) {
    size_t n = 5000;
    size_t p = 20;

    double* X = (double*)malloc(n * p * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    double* beta = (double*)malloc(p * sizeof(double));

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < p; j++) {
            X[j * n + i] = (double)(i * p + j + 1);
        }
        y[i] = (double)(i + 1);
    }

    fc_bench_time_t start = fc_bench_time_now();

    for (int iter = 0; iter < 10; iter++) {
        double* X_copy = (double*)malloc(n * p * sizeof(double));
        memcpy(X_copy, X, n * p * sizeof(double));
        fc_optim_least_squares(X_copy, y, n, p, beta);
        free(X_copy);
    }

    fc_bench_time_t end = fc_bench_time_now();
    double elapsed_ms = fc_bench_time_elapsed_ms(&start, &end);

    printf("  least_squares_large (n=5000, p=20): %.2f ms (10 iterations, %.2f ms/iter)\n",
           elapsed_ms, elapsed_ms / 10.0);

    free(X);
    free(y);
    free(beta);
}

static void bench_least_squares_batch_small(void) {
    size_t n = 50;
    size_t p = 3;
    size_t batch_size = 100;

    double* X = (double*)malloc(batch_size * n * p * sizeof(double));
    double* y = (double*)malloc(batch_size * n * sizeof(double));
    double* beta = (double*)malloc(batch_size * p * sizeof(double));

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < p; j++) {
                X[b * n * p + j * n + i] = (double)(b * n * p + i * p + j + 1);
            }
            y[b * n + i] = (double)(b * n + i + 1);
        }
    }

    fc_bench_time_t start = fc_bench_time_now();

    for (int iter = 0; iter < 100; iter++) {
        double* X_copy = (double*)malloc(batch_size * n * p * sizeof(double));
        memcpy(X_copy, X, batch_size * n * p * sizeof(double));
        fc_optim_least_squares_batch(X_copy, y, n, p, batch_size, beta);
        free(X_copy);
    }

    fc_bench_time_t end = fc_bench_time_now();
    double elapsed_ms = fc_bench_time_elapsed_ms(&start, &end);

    printf("  least_squares_batch (n=50, p=3, batch=100): %.2f ms (100 iterations, %.2f ms/iter)\n",
           elapsed_ms, elapsed_ms / 100.0);

    free(X);
    free(y);
    free(beta);
}

static void bench_least_squares_batch_medium(void) {
    size_t n = 100;
    size_t p = 5;
    size_t batch_size = 50;

    double* X = (double*)malloc(batch_size * n * p * sizeof(double));
    double* y = (double*)malloc(batch_size * n * sizeof(double));
    double* beta = (double*)malloc(batch_size * p * sizeof(double));

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < p; j++) {
                X[b * n * p + j * n + i] = (double)(b * n * p + i * p + j + 1);
            }
            y[b * n + i] = (double)(b * n + i + 1);
        }
    }

    fc_bench_time_t start = fc_bench_time_now();

    for (int iter = 0; iter < 50; iter++) {
        double* X_copy = (double*)malloc(batch_size * n * p * sizeof(double));
        memcpy(X_copy, X, batch_size * n * p * sizeof(double));
        fc_optim_least_squares_batch(X_copy, y, n, p, batch_size, beta);
        free(X_copy);
    }

    fc_bench_time_t end = fc_bench_time_now();
    double elapsed_ms = fc_bench_time_elapsed_ms(&start, &end);

    printf("  least_squares_batch (n=100, p=5, batch=50): %.2f ms (50 iterations, %.2f ms/iter)\n",
           elapsed_ms, elapsed_ms / 50.0);

    free(X);
    free(y);
    free(beta);
}

static void bench_least_squares_ext(void) {
    size_t n = 1000;
    size_t p = 10;

    double* X = (double*)malloc(n * p * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    double* beta = (double*)malloc(p * sizeof(double));
    double* residuals = (double*)malloc(n * sizeof(double));
    double r_squared;
    double std_error;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < p; j++) {
            X[j * n + i] = (double)(i * p + j + 1);
        }
        y[i] = (double)(i + 1);
    }

    fc_bench_time_t start = fc_bench_time_now();

    for (int iter = 0; iter < 100; iter++) {
        double* X_copy = (double*)malloc(n * p * sizeof(double));
        memcpy(X_copy, X, n * p * sizeof(double));
        fc_optim_least_squares_ext(X_copy, y, n, p, beta, residuals, &r_squared, &std_error);
        free(X_copy);
    }

    fc_bench_time_t end = fc_bench_time_now();
    double elapsed_ms = fc_bench_time_elapsed_ms(&start, &end);

    printf("  least_squares_ext (n=1000, p=10): %.2f ms (100 iterations, %.2f ms/iter)\n",
           elapsed_ms, elapsed_ms / 100.0);

    free(X);
    free(y);
    free(beta);
    free(residuals);
}

void bench_least_squares_run(void) {
    printf("\nLeast Squares Regression Benchmarks:\n");
    printf("------------------------------------------------------------\n");

    bench_least_squares_small();
    bench_least_squares_medium();
    bench_least_squares_large();
    bench_least_squares_batch_small();
    bench_least_squares_batch_medium();
    bench_least_squares_ext();

    printf("\n");
}
