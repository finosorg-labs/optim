/**
 * @file test_var_monte_carlo.c
 * @brief Unit tests for Monte Carlo VaR/CVaR calculation
 */

#include "var_monte_carlo.h"
#include "test_framework.h"
#include <math.h>
#include <stdlib.h>

TEST(var_monte_carlo_state_create_destroy) {
    fc_var_mc_state_t* state = fc_optim_var_monte_carlo_state_create(10, 1000, 42);
    ASSERT_NOT_NULL(state);
    fc_optim_var_monte_carlo_state_destroy(state);
}

TEST(var_monte_carlo_state_create_invalid) {
    fc_var_mc_state_t* state = fc_optim_var_monte_carlo_state_create(0, 1000, 42);
    ASSERT_NULL(state);

    state = fc_optim_var_monte_carlo_state_create(10, 0, 42);
    ASSERT_NULL(state);
}

TEST(var_monte_carlo_basic) {
    size_t dim = 3;
    size_t n_paths = 10000;

    fc_var_mc_state_t* state = fc_optim_var_monte_carlo_state_create(dim, n_paths, 42);
    ASSERT_NOT_NULL(state);

    double means[] = {0.001, 0.002, -0.001};
    double cov_matrix[] = {
        0.0004, 0.0001, 0.0000,
        0.0001, 0.0009, 0.0002,
        0.0000, 0.0002, 0.0006
    };
    double weights[] = {0.5, 0.3, 0.2};
    double confidence = 0.95;
    double var, cvar;

    int ret = fc_optim_var_monte_carlo(state, means, cov_matrix, weights, dim, confidence, &var, &cvar);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(cvar <= var);

    fc_optim_var_monte_carlo_state_destroy(state);
}

TEST(var_monte_carlo_single_asset) {
    size_t dim = 1;
    size_t n_paths = 5000;

    fc_var_mc_state_t* state = fc_optim_var_monte_carlo_state_create(dim, n_paths, 123);
    ASSERT_NOT_NULL(state);

    double means[] = {0.001};
    double cov_matrix[] = {0.0004};
    double weights[] = {1.0};
    double confidence = 0.95;
    double var, cvar;

    int ret = fc_optim_var_monte_carlo(state, means, cov_matrix, weights, dim, confidence, &var, &cvar);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(var < means[0]);
    ASSERT_TRUE(cvar < var);

    fc_optim_var_monte_carlo_state_destroy(state);
}

TEST(var_monte_carlo_uncorrelated) {
    size_t dim = 2;
    size_t n_paths = 10000;

    fc_var_mc_state_t* state = fc_optim_var_monte_carlo_state_create(dim, n_paths, 456);
    ASSERT_NOT_NULL(state);

    double means[] = {0.0, 0.0};
    double cov_matrix[] = {
        1.0, 0.0,
        0.0, 1.0
    };
    double weights[] = {0.5, 0.5};
    double confidence = 0.95;
    double var, cvar;

    int ret = fc_optim_var_monte_carlo(state, means, cov_matrix, weights, dim, confidence, &var, &cvar);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(cvar <= var);

    fc_optim_var_monte_carlo_state_destroy(state);
}

TEST(var_monte_carlo_null_inputs) {
    size_t dim = 2;
    fc_var_mc_state_t* state = fc_optim_var_monte_carlo_state_create(dim, 1000, 42);
    double means[] = {0.0, 0.0};
    double cov_matrix[] = {1.0, 0.0, 0.0, 1.0};
    double weights[] = {0.5, 0.5};
    double var, cvar;

    int ret = fc_optim_var_monte_carlo(NULL, means, cov_matrix, weights, dim, 0.95, &var, &cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_monte_carlo(state, NULL, cov_matrix, weights, dim, 0.95, &var, &cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_monte_carlo(state, means, NULL, weights, dim, 0.95, &var, &cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_monte_carlo(state, means, cov_matrix, NULL, dim, 0.95, &var, &cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_monte_carlo(state, means, cov_matrix, weights, dim, 0.95, NULL, &cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_monte_carlo(state, means, cov_matrix, weights, dim, 0.95, &var, NULL);
    ASSERT_EQ(ret, -1);

    fc_optim_var_monte_carlo_state_destroy(state);
}

TEST(var_monte_carlo_dim_mismatch) {
    fc_var_mc_state_t* state = fc_optim_var_monte_carlo_state_create(2, 1000, 42);
    double means[] = {0.0, 0.0, 0.0};
    double cov_matrix[] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double weights[] = {0.33, 0.33, 0.34};
    double var, cvar;

    int ret = fc_optim_var_monte_carlo(state, means, cov_matrix, weights, 3, 0.95, &var, &cvar);
    ASSERT_EQ(ret, -2);

    fc_optim_var_monte_carlo_state_destroy(state);
}

TEST(var_monte_carlo_invalid_confidence) {
    fc_var_mc_state_t* state = fc_optim_var_monte_carlo_state_create(2, 1000, 42);
    double means[] = {0.0, 0.0};
    double cov_matrix[] = {1.0, 0.0, 0.0, 1.0};
    double weights[] = {0.5, 0.5};
    double var, cvar;

    int ret = fc_optim_var_monte_carlo(state, means, cov_matrix, weights, 2, 0.0, &var, &cvar);
    ASSERT_EQ(ret, -3);

    ret = fc_optim_var_monte_carlo(state, means, cov_matrix, weights, 2, 1.0, &var, &cvar);
    ASSERT_EQ(ret, -3);

    fc_optim_var_monte_carlo_state_destroy(state);
}

TEST(var_monte_carlo_not_positive_definite) {
    fc_var_mc_state_t* state = fc_optim_var_monte_carlo_state_create(2, 1000, 42);
    double means[] = {0.0, 0.0};
    double cov_matrix[] = {
        1.0, 2.0,
        2.0, 1.0
    };
    double weights[] = {0.5, 0.5};
    double var, cvar;

    int ret = fc_optim_var_monte_carlo(state, means, cov_matrix, weights, 2, 0.95, &var, &cvar);
    ASSERT_NE(ret, 0);

    fc_optim_var_monte_carlo_state_destroy(state);
}

TEST(var_monte_carlo_from_history_basic) {
    size_t dim = 2;
    size_t n_periods = 100;
    size_t n_paths = 5000;

    fc_var_mc_state_t* state = fc_optim_var_monte_carlo_state_create(dim, n_paths, 789);
    ASSERT_NOT_NULL(state);

    double* returns = malloc(dim * n_periods * sizeof(double));
    for (size_t t = 0; t < n_periods; t++) {
        returns[0 * n_periods + t] = 0.02 * ((double)(t % 20) - 10.0) / 10.0;
        returns[1 * n_periods + t] = 0.015 * ((double)((t + 5) % 20) - 10.0) / 10.0;
    }

    double weights[] = {0.6, 0.4};
    double confidence = 0.95;
    double var, cvar;

    int ret = fc_optim_var_monte_carlo_from_history(state, returns, weights, dim, n_periods, confidence, &var, &cvar);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(cvar <= var);

    free(returns);
    fc_optim_var_monte_carlo_state_destroy(state);
}

TEST(var_monte_carlo_from_history_insufficient_data) {
    size_t dim = 10;
    size_t n_periods = 5;
    fc_var_mc_state_t* state = fc_optim_var_monte_carlo_state_create(dim, 1000, 42);

    double* returns = malloc(dim * n_periods * sizeof(double));
    double* weights = malloc(dim * sizeof(double));
    for (size_t i = 0; i < dim; i++) {
        weights[i] = 1.0 / dim;
    }

    double var, cvar;
    int ret = fc_optim_var_monte_carlo_from_history(state, returns, weights, dim, n_periods, 0.95, &var, &cvar);

    ASSERT_EQ(ret, -6);

    free(returns);
    free(weights);
    fc_optim_var_monte_carlo_state_destroy(state);
}

TEST(var_monte_carlo_from_history_null_inputs) {
    fc_var_mc_state_t* state = fc_optim_var_monte_carlo_state_create(2, 1000, 42);
    double returns[] = {0.01, 0.02, -0.01, 0.00};
    double weights[] = {0.5, 0.5};
    double var, cvar;

    int ret = fc_optim_var_monte_carlo_from_history(NULL, returns, weights, 2, 2, 0.95, &var, &cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_monte_carlo_from_history(state, NULL, weights, 2, 2, 0.95, &var, &cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_monte_carlo_from_history(state, returns, NULL, 2, 2, 0.95, &var, &cvar);
    ASSERT_EQ(ret, -1);

    fc_optim_var_monte_carlo_state_destroy(state);
}

TEST(var_monte_carlo_reproducibility) {
    size_t dim = 3;
    size_t n_paths = 1000;

    fc_var_mc_state_t* state1 = fc_optim_var_monte_carlo_state_create(dim, n_paths, 12345);
    fc_var_mc_state_t* state2 = fc_optim_var_monte_carlo_state_create(dim, n_paths, 12345);

    double means[] = {0.001, 0.002, -0.001};
    double cov_matrix[] = {
        0.0004, 0.0001, 0.0000,
        0.0001, 0.0009, 0.0002,
        0.0000, 0.0002, 0.0006
    };
    double weights[] = {0.5, 0.3, 0.2};

    double var1, cvar1, var2, cvar2;

    fc_optim_var_monte_carlo(state1, means, cov_matrix, weights, dim, 0.95, &var1, &cvar1);
    fc_optim_var_monte_carlo(state2, means, cov_matrix, weights, dim, 0.95, &var2, &cvar2);

    ASSERT_TRUE(fabs(var1 - var2) < 1e-10);
    ASSERT_TRUE(fabs(cvar1 - cvar2) < 1e-10);

    fc_optim_var_monte_carlo_state_destroy(state1);
    fc_optim_var_monte_carlo_state_destroy(state2);
}

void register_var_monte_carlo_tests(void) {
    RUN_TEST(var_monte_carlo_state_create_destroy);
    RUN_TEST(var_monte_carlo_state_create_invalid);
    RUN_TEST(var_monte_carlo_basic);
    RUN_TEST(var_monte_carlo_single_asset);
    RUN_TEST(var_monte_carlo_uncorrelated);
    RUN_TEST(var_monte_carlo_null_inputs);
    RUN_TEST(var_monte_carlo_dim_mismatch);
    RUN_TEST(var_monte_carlo_invalid_confidence);
    RUN_TEST(var_monte_carlo_not_positive_definite);
    RUN_TEST(var_monte_carlo_from_history_basic);
    RUN_TEST(var_monte_carlo_from_history_insufficient_data);
    RUN_TEST(var_monte_carlo_from_history_null_inputs);
    RUN_TEST(var_monte_carlo_reproducibility);
}
