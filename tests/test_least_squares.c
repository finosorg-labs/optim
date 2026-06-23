/**
 * @file test_least_squares.c
 * @brief Unit tests for least squares regression
 */

#include "least_squares.h"
#include "test_framework.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

TEST(least_squares_basic) {
    double X[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    double y[] = {2.0, 4.0, 5.0, 4.0, 5.0};
    double beta[2];

    int ret = fc_optim_least_squares(X, y, 5, 2, 0, beta);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(fabs(beta[0] - 2.2) < 0.1);
    ASSERT_TRUE(fabs(beta[1] - 0.6) < 0.1);
}

TEST(least_squares_perfect_fit) {
    double X[] = {1.0, 1.0, 1.0, 1.0, 2.0, 3.0};
    double y[] = {3.0, 5.0, 7.0};
    double beta[2];

    int ret = fc_optim_least_squares(X, y, 3, 2, 0, beta);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(fabs(beta[0] - 1.0) < 1e-10);
    ASSERT_TRUE(fabs(beta[1] - 2.0) < 1e-10);
}

TEST(least_squares_null_inputs) {
    double X[] = {1.0, 2.0, 3.0, 4.0};
    double y[] = {1.0, 2.0};
    double beta[2];

    ASSERT_EQ(fc_optim_least_squares(NULL, y, 2, 2, 0, beta), -1);
    ASSERT_EQ(fc_optim_least_squares(X, NULL, 2, 2, 0, beta), -1);
    ASSERT_EQ(fc_optim_least_squares(X, y, 2, 2, 0, NULL), -1);
}

TEST(least_squares_invalid_dimensions) {
    double X[] = {1.0, 2.0, 3.0, 4.0};
    double y[] = {1.0, 2.0};
    double beta[2];

    ASSERT_EQ(fc_optim_least_squares(X, y, 0, 2, 0, beta), -2);
    ASSERT_EQ(fc_optim_least_squares(X, y, 2, 0, 0, beta), -2);
    ASSERT_EQ(fc_optim_least_squares(X, y, 2, 3, 0, beta), -2);
}

TEST(least_squares_multicollinearity) {
    double X[] = {1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 2.0, 4.0, 6.0};
    double y[] = {1.0, 2.0, 3.0};
    double beta[3];

    int ret = fc_optim_least_squares(X, y, 3, 3, 0, beta);

    ASSERT_EQ(ret, -3);
}

TEST(least_squares_batch_basic) {
    double X[] = {1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0};
    double y[] = {3.0, 5.0, 7.0, 4.0, 7.0, 10.0};
    double beta[4];

    size_t count = fc_optim_least_squares_batch(X, y, 3, 2, 2, 0, beta);

    ASSERT_EQ(count, 2);
    ASSERT_TRUE(fabs(beta[0] - 1.0) < 1e-10);
    ASSERT_TRUE(fabs(beta[1] - 2.0) < 1e-10);
    ASSERT_TRUE(fabs(beta[2] - (-2.0)) < 1e-10);
    ASSERT_TRUE(fabs(beta[3] - 3.0) < 1e-10);
}

TEST(least_squares_batch_null_inputs) {
    double X[] = {1.0, 2.0, 3.0, 4.0};
    double y[] = {1.0, 2.0};
    double beta[2];

    ASSERT_EQ(fc_optim_least_squares_batch(NULL, y, 2, 2, 1, 0, beta), 0);
    ASSERT_EQ(fc_optim_least_squares_batch(X, NULL, 2, 2, 1, 0, beta), 0);
    ASSERT_EQ(fc_optim_least_squares_batch(X, y, 2, 2, 1, 0, NULL), 0);
}

TEST(least_squares_batch_invalid_dimensions) {
    double X[] = {1.0, 2.0, 3.0, 4.0};
    double y[] = {1.0, 2.0};
    double beta[2];

    ASSERT_EQ(fc_optim_least_squares_batch(X, y, 0, 2, 1, 0, beta), 0);
    ASSERT_EQ(fc_optim_least_squares_batch(X, y, 2, 0, 1, 0, beta), 0);
    ASSERT_EQ(fc_optim_least_squares_batch(X, y, 2, 2, 0, 0, beta), 0);
}

TEST(least_squares_ext_basic) {
    double X[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    double y[] = {2.0, 4.0, 5.0, 4.0, 5.0};
    double beta[2];
    double residuals[5];
    double r_squared;
    double std_error;

    int ret = fc_optim_least_squares_ext(X, y, 5, 2, 0, beta, residuals, &r_squared, &std_error);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(fabs(beta[0] - 2.2) < 0.1);
    ASSERT_TRUE(fabs(beta[1] - 0.6) < 0.1);
    ASSERT_TRUE(r_squared >= 0.0 && r_squared <= 1.0);
    ASSERT_TRUE(std_error >= 0.0);

    double residual_sum = 0.0;
    for (size_t i = 0; i < 5; i++) {
        residual_sum += fabs(residuals[i]);
    }
    ASSERT_TRUE(residual_sum > 0.0);
}

TEST(least_squares_ext_perfect_fit) {
    double X[] = {1.0, 1.0, 1.0, 1.0, 2.0, 3.0};
    double y[] = {3.0, 5.0, 7.0};
    double beta[2];
    double residuals[3];
    double r_squared;
    double std_error;

    int ret = fc_optim_least_squares_ext(X, y, 3, 2, 0, beta, residuals, &r_squared, &std_error);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(fabs(r_squared - 1.0) < 1e-10);
    ASSERT_TRUE(std_error < 1e-10);

    for (size_t i = 0; i < 3; i++) {
        ASSERT_TRUE(fabs(residuals[i]) < 1e-10);
    }
}

TEST(least_squares_ext_null_outputs) {
    double X[] = {1.0, 1.0, 1.0, 1.0, 2.0, 3.0};
    double y[] = {3.0, 5.0, 7.0};
    double beta[2];

    int ret = fc_optim_least_squares_ext(X, y, 3, 2, 0, beta, NULL, NULL, NULL);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(fabs(beta[0] - 1.0) < 1e-10);
    ASSERT_TRUE(fabs(beta[1] - 2.0) < 1e-10);
}

TEST(least_squares_larger_dataset) {
    size_t n = 100;
    size_t p = 3;

    double* X = (double*)malloc(n * p * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    double beta[3];

    for (size_t i = 0; i < n; i++) {
        X[0 * n + i] = 1.0;
        X[1 * n + i] = (double)(i + 1);
        X[2 * n + i] = (double)((i + 1) * (i + 1));

        y[i] = 2.0 + 3.0 * (i + 1) + 0.5 * (i + 1) * (i + 1) + ((i % 2) ? 0.1 : -0.1);
    }

    int ret = fc_optim_least_squares(X, y, n, p, 0, beta);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(fabs(beta[0] - 2.0) < 0.5);
    ASSERT_TRUE(fabs(beta[1] - 3.0) < 0.5);
    ASSERT_TRUE(fabs(beta[2] - 0.5) < 0.1);

    free(X);
    free(y);
}

void register_optim_tests(void) {
    RUN_TEST(least_squares_basic);
    RUN_TEST(least_squares_perfect_fit);
    RUN_TEST(least_squares_null_inputs);
    RUN_TEST(least_squares_invalid_dimensions);
    RUN_TEST(least_squares_multicollinearity);
    RUN_TEST(least_squares_batch_basic);
    RUN_TEST(least_squares_batch_null_inputs);
    RUN_TEST(least_squares_batch_invalid_dimensions);
    RUN_TEST(least_squares_ext_basic);
    RUN_TEST(least_squares_ext_perfect_fit);
    RUN_TEST(least_squares_ext_null_outputs);
    RUN_TEST(least_squares_larger_dataset);
}
