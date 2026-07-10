/**
 * @file test_var_historical.c
 * @brief Unit tests for historical simulation VaR/CVaR calculation
 */

#include "var_historical.h"
#include "test_framework.h"
#include <math.h>
#include <stdlib.h>

TEST(var_historical_basic) {
    double returns[] = {-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06};
    size_t n = 10;
    double confidence = 0.90;
    double var, cvar;

    int ret = fc_optim_var_historical(returns, n, confidence, &var, &cvar);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(var < 0.0);
    ASSERT_TRUE(cvar < var);
}

TEST(var_historical_95_confidence) {
    double returns[] = {-0.10, -0.08, -0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05,
                        0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15};
    size_t n = 20;
    double confidence = 0.95;
    double var, cvar;

    int ret = fc_optim_var_historical(returns, n, confidence, &var, &cvar);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(var < 0.0);
    ASSERT_TRUE(cvar <= var);
}

TEST(var_historical_sorted_returns) {
    double returns[] = {-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04};
    size_t n = 10;
    double confidence = 0.90;
    double var, cvar;

    int ret = fc_optim_var_historical(returns, n, confidence, &var, &cvar);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(var >= returns[0] && var <= returns[n-1]);
    ASSERT_TRUE(cvar >= returns[0] && cvar <= var);
}

TEST(var_historical_unsorted_returns) {
    double returns[] = {0.02, -0.03, 0.05, -0.01, 0.0, -0.04, 0.03, -0.02, 0.04, 0.01};
    size_t n = 10;
    double confidence = 0.90;
    double var, cvar;

    int ret = fc_optim_var_historical(returns, n, confidence, &var, &cvar);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(var < 0.0);
    ASSERT_TRUE(cvar < var);
}

TEST(var_historical_null_inputs) {
    double returns[] = {0.01};
    double var, cvar;

    int ret = fc_optim_var_historical(NULL, 1, 0.95, &var, &cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_historical(returns, 1, 0.95, NULL, &cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_historical(returns, 1, 0.95, &var, NULL);
    ASSERT_EQ(ret, -1);
}

TEST(var_historical_zero_size) {
    double returns[] = {0.01};
    double var, cvar;

    int ret = fc_optim_var_historical(returns, 0, 0.95, &var, &cvar);
    ASSERT_EQ(ret, -2);
}

TEST(var_historical_invalid_confidence) {
    double returns[] = {0.01};
    double var, cvar;

    int ret = fc_optim_var_historical(returns, 1, 0.0, &var, &cvar);
    ASSERT_EQ(ret, -3);

    ret = fc_optim_var_historical(returns, 1, 1.0, &var, &cvar);
    ASSERT_EQ(ret, -3);

    ret = fc_optim_var_historical(returns, 1, -0.5, &var, &cvar);
    ASSERT_EQ(ret, -3);

    ret = fc_optim_var_historical(returns, 1, 1.5, &var, &cvar);
    ASSERT_EQ(ret, -3);
}

TEST(var_historical_portfolio_basic) {
    double returns[] = {
        -0.02, 0.01, 0.03, -0.01, 0.02,
        0.01, -0.01, 0.02, 0.00, 0.01,
        0.00, 0.02, -0.02, 0.01, -0.01
    };
    double weights[] = {0.5, 0.3, 0.2};
    size_t m = 3;
    size_t n = 5;
    double confidence = 0.90;
    double var, cvar;

    int ret = fc_optim_var_historical_portfolio(returns, weights, m, n, confidence, &var, &cvar);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(cvar <= var);
}

TEST(var_historical_portfolio_equal_weights) {
    double returns[] = {
        0.01, 0.02, -0.01, 0.00,
        -0.01, 0.01, 0.02, -0.02
    };
    double weights[] = {0.5, 0.5};
    size_t m = 2;
    size_t n = 4;
    double confidence = 0.75;
    double var, cvar;

    int ret = fc_optim_var_historical_portfolio(returns, weights, m, n, confidence, &var, &cvar);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(cvar <= var);
}

TEST(var_historical_portfolio_null_inputs) {
    double returns[] = {0.01};
    double weights[] = {1.0};
    double var, cvar;

    int ret = fc_optim_var_historical_portfolio(NULL, weights, 1, 1, 0.95, &var, &cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_historical_portfolio(returns, NULL, 1, 1, 0.95, &var, &cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_historical_portfolio(returns, weights, 1, 1, 0.95, NULL, &cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_historical_portfolio(returns, weights, 1, 1, 0.95, &var, NULL);
    ASSERT_EQ(ret, -1);
}

TEST(var_historical_portfolio_zero_size) {
    double returns[] = {0.01};
    double weights[] = {1.0};
    double var, cvar;

    int ret = fc_optim_var_historical_portfolio(returns, weights, 0, 1, 0.95, &var, &cvar);
    ASSERT_EQ(ret, -2);

    ret = fc_optim_var_historical_portfolio(returns, weights, 1, 0, 0.95, &var, &cvar);
    ASSERT_EQ(ret, -2);
}

TEST(var_historical_batch_basic) {
    double returns[] = {
        -0.02, 0.01, 0.03, -0.01,
        0.01, -0.01, 0.02, 0.00
    };
    double weights_matrix[] = {
        0.6, 0.4,
        0.5, 0.5,
        0.7, 0.3
    };
    size_t num_portfolios = 3;
    size_t m = 2;
    size_t n = 4;
    double confidence = 0.75;
    double var[3], cvar[3];

    int ret = fc_optim_var_historical_batch(returns, weights_matrix, num_portfolios, m, n, confidence, var, cvar);

    ASSERT_EQ(ret, 0);

    for (size_t p = 0; p < num_portfolios; p++) {
        ASSERT_TRUE(cvar[p] <= var[p]);
    }
}

TEST(var_historical_batch_large) {
    size_t num_portfolios = 10;
    size_t m = 5;
    size_t n = 100;

    double* returns = malloc(m * n * sizeof(double));
    double* weights_matrix = malloc(num_portfolios * m * sizeof(double));
    double* var = malloc(num_portfolios * sizeof(double));
    double* cvar = malloc(num_portfolios * sizeof(double));

    for (size_t i = 0; i < m * n; i++) {
        returns[i] = 0.02 * ((double)(i % 20) - 10.0) / 10.0;
    }

    for (size_t p = 0; p < num_portfolios; p++) {
        for (size_t i = 0; i < m; i++) {
            weights_matrix[p * m + i] = 1.0 / m;
        }
    }

    int ret = fc_optim_var_historical_batch(returns, weights_matrix, num_portfolios, m, n, 0.95, var, cvar);

    ASSERT_EQ(ret, 0);

    for (size_t p = 0; p < num_portfolios; p++) {
        ASSERT_TRUE(cvar[p] <= var[p]);
    }

    free(returns);
    free(weights_matrix);
    free(var);
    free(cvar);
}

TEST(var_historical_batch_null_inputs) {
    double returns[] = {0.01};
    double weights[] = {1.0};
    double var[1], cvar[1];

    int ret = fc_optim_var_historical_batch(NULL, weights, 1, 1, 1, 0.95, var, cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_historical_batch(returns, NULL, 1, 1, 1, 0.95, var, cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_historical_batch(returns, weights, 1, 1, 1, 0.95, NULL, cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_historical_batch(returns, weights, 1, 1, 1, 0.95, var, NULL);
    ASSERT_EQ(ret, -1);
}

TEST(var_historical_batch_zero_size) {
    double returns[] = {0.01};
    double weights[] = {1.0};
    double var[1], cvar[1];

    int ret = fc_optim_var_historical_batch(returns, weights, 0, 1, 1, 0.95, var, cvar);
    ASSERT_EQ(ret, -2);

    ret = fc_optim_var_historical_batch(returns, weights, 1, 0, 1, 0.95, var, cvar);
    ASSERT_EQ(ret, -2);

    ret = fc_optim_var_historical_batch(returns, weights, 1, 1, 0, 0.95, var, cvar);
    ASSERT_EQ(ret, -2);
}

TEST(var_historical_batch_from_returns_basic) {
    // 3 assets, each with 10 periods
    double returns_matrix[] = {
        -0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,  // Asset 1
        -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,  // Asset 2
        -0.07, -0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05, 0.07, 0.09  // Asset 3
    };
    size_t num_assets = 3;
    size_t n = 10;
    double confidence = 0.90;
    double var[3], cvar[3];

    int ret = fc_optim_var_historical_batch_from_returns(returns_matrix, num_assets, n, confidence, var, cvar);

    ASSERT_EQ(ret, 0);
    // All assets should have negative VaR and CVaR < VaR
    for (size_t i = 0; i < num_assets; i++) {
        ASSERT_TRUE(var[i] < 0.0);
        ASSERT_TRUE(cvar[i] < var[i]);
    }
}

TEST(var_historical_batch_from_returns_consistency) {
    // Verify batch results match individual calls
    double returns_matrix[] = {
        -0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
        -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06
    };
    size_t num_assets = 2;
    size_t n = 10;
    double confidence = 0.95;
    double var_batch[2], cvar_batch[2];
    double var_single[2], cvar_single[2];

    // Batch call
    int ret_batch = fc_optim_var_historical_batch_from_returns(
        returns_matrix, num_assets, n, confidence, var_batch, cvar_batch
    );
    ASSERT_EQ(ret_batch, 0);

    // Individual calls
    for (size_t i = 0; i < num_assets; i++) {
        int ret = fc_optim_var_historical(&returns_matrix[i * n], n, confidence, &var_single[i], &cvar_single[i]);
        ASSERT_EQ(ret, 0);
    }

    // Compare results
    for (size_t i = 0; i < num_assets; i++) {
        ASSERT_TRUE(fabs(var_batch[i] - var_single[i]) < 1e-10);
        ASSERT_TRUE(fabs(cvar_batch[i] - cvar_single[i]) < 1e-10);
    }
}

TEST(var_historical_batch_from_returns_null_inputs) {
    double returns[10];
    double var[2], cvar[2];

    int ret = fc_optim_var_historical_batch_from_returns(NULL, 2, 10, 0.95, var, cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_historical_batch_from_returns(returns, 2, 10, 0.95, NULL, cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_historical_batch_from_returns(returns, 2, 10, 0.95, var, NULL);
    ASSERT_EQ(ret, -1);
}

TEST(var_historical_batch_from_returns_zero_size) {
    double returns[10] = {0};
    double var[2], cvar[2];

    int ret = fc_optim_var_historical_batch_from_returns(returns, 0, 10, 0.95, var, cvar);
    ASSERT_EQ(ret, -2);

    ret = fc_optim_var_historical_batch_from_returns(returns, 2, 0, 0.95, var, cvar);
    ASSERT_EQ(ret, -2);
}

void register_var_historical_tests(void) {
    RUN_TEST(var_historical_basic);
    RUN_TEST(var_historical_95_confidence);
    RUN_TEST(var_historical_sorted_returns);
    RUN_TEST(var_historical_unsorted_returns);
    RUN_TEST(var_historical_null_inputs);
    RUN_TEST(var_historical_zero_size);
    RUN_TEST(var_historical_invalid_confidence);
    RUN_TEST(var_historical_portfolio_basic);
    RUN_TEST(var_historical_portfolio_equal_weights);
    RUN_TEST(var_historical_portfolio_null_inputs);
    RUN_TEST(var_historical_portfolio_zero_size);
    RUN_TEST(var_historical_batch_basic);
    RUN_TEST(var_historical_batch_large);
    RUN_TEST(var_historical_batch_null_inputs);
    RUN_TEST(var_historical_batch_zero_size);
    RUN_TEST(var_historical_batch_from_returns_basic);
    RUN_TEST(var_historical_batch_from_returns_consistency);
    RUN_TEST(var_historical_batch_from_returns_null_inputs);
    RUN_TEST(var_historical_batch_from_returns_zero_size);
}
