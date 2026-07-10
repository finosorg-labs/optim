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

TEST(var_monte_carlo_batch_basic) {
    size_t dim = 2;
    size_t n_periods = 50;
    size_t n_paths = 10000;
    double returns[] = {
        0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.00, -0.01, 0.02,
        0.015, -0.015, 0.025, -0.02, 0.01, -0.01, 0.03, -0.02, 0.01, 0.02,
        0.02, -0.01, 0.01, -0.02, 0.03, -0.01, 0.00, 0.01, -0.03, 0.02,
        -0.01, 0.02, -0.015, 0.025, -0.01, 0.02, -0.02, 0.01, 0.00, -0.01,
        0.01, -0.01, 0.02, -0.02, 0.015, -0.025, 0.01, 0.00, -0.01, 0.03,
        -0.02, 0.01, 0.015, -0.015, 0.02, -0.01, 0.00, 0.01, -0.02, 0.015,
        0.01, -0.015, 0.02, -0.01, 0.01, -0.02, 0.00, 0.015, -0.01, 0.02,
        -0.01, 0.02, -0.02, 0.01, 0.00, -0.015, 0.025, -0.01, 0.01, 0.00,
        0.02, -0.01, 0.015, -0.02, 0.01, 0.00, -0.01, 0.02, -0.015, 0.01,
        -0.015, 0.025, -0.01, 0.01, -0.02, 0.015, 0.00, -0.01, 0.02, -0.01
    };

    // 3 different portfolios
    double weights_matrix[] = {
        0.6, 0.4,   // Portfolio 1
        0.5, 0.5,   // Portfolio 2
        0.7, 0.3    // Portfolio 3
    };
    size_t num_portfolios = 3;
    double confidence = 0.95;
    double var[3], cvar[3];

    fc_var_mc_state_t* state = fc_optim_var_monte_carlo_state_create(dim, n_paths, 42);
    ASSERT_TRUE(state != NULL);

    int ret = fc_optim_var_monte_carlo_batch(
        state, returns, weights_matrix, num_portfolios, dim, n_periods, confidence, var, cvar
    );

    ASSERT_EQ(ret, 0);
    // All portfolios should have negative VaR and CVaR < VaR
    for (size_t i = 0; i < num_portfolios; i++) {
        ASSERT_TRUE(var[i] < 0.0);
        ASSERT_TRUE(cvar[i] < var[i]);
    }

    fc_optim_var_monte_carlo_state_destroy(state);
}

TEST(var_monte_carlo_batch_consistency) {
    // Verify batch results match individual calls
    size_t dim = 2;
    size_t n_periods = 30;
    size_t n_paths = 5000;
    double returns[] = {
        0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.00, -0.01, 0.02,
        0.015, -0.015, 0.025, -0.02, 0.01, -0.01, 0.03, -0.02, 0.01, 0.02,
        0.02, -0.01, 0.01, -0.02, 0.03, -0.01, 0.00, 0.01, -0.03, 0.02,
        -0.02, 0.01, 0.015, -0.015, 0.02, -0.01, 0.00, 0.01, -0.02, 0.015,
        0.01, -0.015, 0.02, -0.01, 0.01, -0.02, 0.00, 0.015, -0.01, 0.02,
        -0.01, 0.02, -0.02, 0.01, 0.00, -0.015, 0.025, -0.01, 0.01, 0.00
    };

    double weights_matrix[] = {
        0.6, 0.4,
        0.5, 0.5
    };
    size_t num_portfolios = 2;
    double confidence = 0.95;
    double var_batch[2], cvar_batch[2];
    double var_single[2], cvar_single[2];

    fc_var_mc_state_t* state = fc_optim_var_monte_carlo_state_create(dim, n_paths, 123);
    ASSERT_TRUE(state != NULL);

    // Batch call
    int ret_batch = fc_optim_var_monte_carlo_batch(
        state, returns, weights_matrix, num_portfolios, dim, n_periods, confidence, var_batch, cvar_batch
    );
    ASSERT_EQ(ret_batch, 0);

    // Individual calls (need fresh state with same seed for each)
    for (size_t i = 0; i < num_portfolios; i++) {
        fc_var_mc_state_t* state_single = fc_optim_var_monte_carlo_state_create(dim, n_paths, 123);
        int ret = fc_optim_var_monte_carlo_from_history(
            state_single, returns, &weights_matrix[i * dim], dim, n_periods, confidence,
            &var_single[i], &cvar_single[i]
        );
        ASSERT_EQ(ret, 0);
        fc_optim_var_monte_carlo_state_destroy(state_single);
    }

    // Results should be close (Monte Carlo has randomness)
    for (size_t i = 0; i < num_portfolios; i++) {
        double var_rel_err = fabs((var_batch[i] - var_single[i]) / var_single[i]);
        double cvar_rel_err = fabs((cvar_batch[i] - cvar_single[i]) / cvar_single[i]);
        ASSERT_TRUE(var_rel_err < 0.05);  // Within 5%
        ASSERT_TRUE(cvar_rel_err < 0.05);
    }

    fc_optim_var_monte_carlo_state_destroy(state);
}

TEST(var_monte_carlo_batch_null_inputs) {
    size_t dim = 2;
    size_t n_periods = 20;
    double returns[40];
    double weights[4];
    double var[2], cvar[2];
    fc_var_mc_state_t* state = fc_optim_var_monte_carlo_state_create(dim, 1000, 42);

    int ret = fc_optim_var_monte_carlo_batch(NULL, returns, weights, 2, dim, n_periods, 0.95, var, cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_monte_carlo_batch(state, NULL, weights, 2, dim, n_periods, 0.95, var, cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_monte_carlo_batch(state, returns, NULL, 2, dim, n_periods, 0.95, var, cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_monte_carlo_batch(state, returns, weights, 2, dim, n_periods, 0.95, NULL, cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_monte_carlo_batch(state, returns, weights, 2, dim, n_periods, 0.95, var, NULL);
    ASSERT_EQ(ret, -1);

    fc_optim_var_monte_carlo_state_destroy(state);
}

TEST(var_monte_carlo_batch_zero_size) {
    size_t dim = 2;
    size_t n_periods = 20;
    double returns[40];
    double weights[4];
    double var[2], cvar[2];
    fc_var_mc_state_t* state = fc_optim_var_monte_carlo_state_create(dim, 1000, 42);

    int ret = fc_optim_var_monte_carlo_batch(state, returns, weights, 0, dim, n_periods, 0.95, var, cvar);
    ASSERT_EQ(ret, -2);

    ret = fc_optim_var_monte_carlo_batch(state, returns, weights, 2, 0, n_periods, 0.95, var, cvar);
    ASSERT_EQ(ret, -2);

    ret = fc_optim_var_monte_carlo_batch(state, returns, weights, 2, dim, 0, 0.95, var, cvar);
    ASSERT_EQ(ret, -2);

    fc_optim_var_monte_carlo_state_destroy(state);
}

TEST(var_monte_carlo_batch_insufficient_data) {
    size_t dim = 3;
    size_t n_periods = 3;  // n_periods < dim + 1
    double returns[9];
    double weights[3];
    double var[1], cvar[1];
    fc_var_mc_state_t* state = fc_optim_var_monte_carlo_state_create(dim, 1000, 42);

    int ret = fc_optim_var_monte_carlo_batch(state, returns, weights, 1, dim, n_periods, 0.95, var, cvar);
    ASSERT_EQ(ret, -7);

    fc_optim_var_monte_carlo_state_destroy(state);
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
    RUN_TEST(var_monte_carlo_batch_basic);
    RUN_TEST(var_monte_carlo_batch_consistency);
    RUN_TEST(var_monte_carlo_batch_null_inputs);
    RUN_TEST(var_monte_carlo_batch_zero_size);
    RUN_TEST(var_monte_carlo_batch_insufficient_data);
}
