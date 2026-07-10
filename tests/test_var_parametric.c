/**
 * @file test_var_parametric.c
 * @brief Unit tests for parametric VaR/CVaR calculation
 */

#include "var_parametric.h"
#include "test_framework.h"
#include <math.h>
#include <stdlib.h>

TEST(var_parametric_basic) {
    double mean = 0.001;
    double stddev = 0.02;
    double confidence = 0.95;
    double var, cvar;

    int ret = fc_optim_var_parametric(mean, stddev, confidence, &var, &cvar);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(var < mean);
    ASSERT_TRUE(cvar < var);
    ASSERT_TRUE(fabs(var - (mean - 1.645 * stddev)) < 0.001);
}

TEST(var_parametric_99_confidence) {
    double mean = 0.0;
    double stddev = 1.0;
    double confidence = 0.99;
    double var, cvar;

    int ret = fc_optim_var_parametric(mean, stddev, confidence, &var, &cvar);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(fabs(var - (-2.326)) < 0.01);
    ASSERT_TRUE(cvar < var);
}

TEST(var_parametric_positive_mean) {
    double mean = 0.05;
    double stddev = 0.1;
    double confidence = 0.95;
    double var, cvar;

    int ret = fc_optim_var_parametric(mean, stddev, confidence, &var, &cvar);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(var > mean - 2.0 * stddev);
    ASSERT_TRUE(cvar < var);
}

TEST(var_parametric_negative_mean) {
    double mean = -0.01;
    double stddev = 0.05;
    double confidence = 0.95;
    double var, cvar;

    int ret = fc_optim_var_parametric(mean, stddev, confidence, &var, &cvar);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(var < mean);
    ASSERT_TRUE(cvar < var);
}

TEST(var_parametric_null_output) {
    double mean = 0.0;
    double stddev = 1.0;
    double confidence = 0.95;

    int ret = fc_optim_var_parametric(mean, stddev, confidence, NULL, NULL);
    ASSERT_EQ(ret, -1);

    double var;
    ret = fc_optim_var_parametric(mean, stddev, confidence, &var, NULL);
    ASSERT_EQ(ret, -1);

    double cvar;
    ret = fc_optim_var_parametric(mean, stddev, confidence, NULL, &cvar);
    ASSERT_EQ(ret, -1);
}

TEST(var_parametric_invalid_stddev) {
    double mean = 0.0;
    double confidence = 0.95;
    double var, cvar;

    int ret = fc_optim_var_parametric(mean, 0.0, confidence, &var, &cvar);
    ASSERT_EQ(ret, -2);

    ret = fc_optim_var_parametric(mean, -0.1, confidence, &var, &cvar);
    ASSERT_EQ(ret, -2);
}

TEST(var_parametric_invalid_confidence) {
    double mean = 0.0;
    double stddev = 1.0;
    double var, cvar;

    int ret = fc_optim_var_parametric(mean, stddev, 0.0, &var, &cvar);
    ASSERT_EQ(ret, -3);

    ret = fc_optim_var_parametric(mean, stddev, 1.0, &var, &cvar);
    ASSERT_EQ(ret, -3);

    ret = fc_optim_var_parametric(mean, stddev, -0.5, &var, &cvar);
    ASSERT_EQ(ret, -3);

    ret = fc_optim_var_parametric(mean, stddev, 1.5, &var, &cvar);
    ASSERT_EQ(ret, -3);
}

TEST(var_parametric_batch_basic) {
    double means[] = {0.001, 0.002, -0.001, 0.0};
    double stddevs[] = {0.02, 0.03, 0.025, 0.01};
    double confidence = 0.95;
    double var[4], cvar[4];

    int ret = fc_optim_var_parametric_batch(4, means, stddevs, confidence, var, cvar);

    ASSERT_EQ(ret, 0);

    for (size_t i = 0; i < 4; i++) {
        ASSERT_TRUE(var[i] < means[i]);
        ASSERT_TRUE(cvar[i] < var[i]);
    }
}

TEST(var_parametric_batch_large) {
    const size_t n = 500;
    double* means = malloc(n * sizeof(double));
    double* stddevs = malloc(n * sizeof(double));
    double* var = malloc(n * sizeof(double));
    double* cvar = malloc(n * sizeof(double));

    for (size_t i = 0; i < n; i++) {
        means[i] = 0.001 * (i % 10);
        stddevs[i] = 0.01 + 0.001 * (i % 20);
    }

    int ret = fc_optim_var_parametric_batch(n, means, stddevs, 0.95, var, cvar);

    ASSERT_EQ(ret, 0);

    for (size_t i = 0; i < n; i++) {
        ASSERT_TRUE(var[i] < means[i]);
        ASSERT_TRUE(cvar[i] < var[i]);
    }

    free(means);
    free(stddevs);
    free(var);
    free(cvar);
}

TEST(var_parametric_batch_consistency) {
    double means[] = {0.001, 0.002, -0.001};
    double stddevs[] = {0.02, 0.03, 0.025};
    double confidence = 0.95;
    double var_batch[3], cvar_batch[3];
    double var_single[3], cvar_single[3];

    int ret = fc_optim_var_parametric_batch(3, means, stddevs, confidence, var_batch, cvar_batch);
    ASSERT_EQ(ret, 0);

    for (size_t i = 0; i < 3; i++) {
        ret = fc_optim_var_parametric(means[i], stddevs[i], confidence, &var_single[i], &cvar_single[i]);
        ASSERT_EQ(ret, 0);
    }

    for (size_t i = 0; i < 3; i++) {
        ASSERT_TRUE(fabs(var_batch[i] - var_single[i]) < 1e-10);
        ASSERT_TRUE(fabs(cvar_batch[i] - cvar_single[i]) < 1e-10);
    }
}

TEST(var_parametric_batch_null_inputs) {
    double means[] = {0.001};
    double stddevs[] = {0.02};
    double var[1], cvar[1];

    int ret = fc_optim_var_parametric_batch(1, NULL, stddevs, 0.95, var, cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_parametric_batch(1, means, NULL, 0.95, var, cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_parametric_batch(1, means, stddevs, 0.95, NULL, cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_parametric_batch(1, means, stddevs, 0.95, var, NULL);
    ASSERT_EQ(ret, -1);
}

TEST(var_parametric_batch_zero_size) {
    double means[] = {0.001};
    double stddevs[] = {0.02};
    double var[1], cvar[1];

    int ret = fc_optim_var_parametric_batch(0, means, stddevs, 0.95, var, cvar);
    ASSERT_EQ(ret, -2);
}

TEST(var_parametric_batch_invalid_stddev) {
    double means[] = {0.001, 0.002};
    double stddevs[] = {0.02, -0.01};
    double var[2], cvar[2];

    int ret = fc_optim_var_parametric_batch(2, means, stddevs, 0.95, var, cvar);
    ASSERT_EQ(ret, -3);

    stddevs[1] = 0.0;
    ret = fc_optim_var_parametric_batch(2, means, stddevs, 0.95, var, cvar);
    ASSERT_EQ(ret, -3);
}

TEST(var_parametric_batch_invalid_confidence) {
    double means[] = {0.001};
    double stddevs[] = {0.02};
    double var[1], cvar[1];

    int ret = fc_optim_var_parametric_batch(1, means, stddevs, 0.0, var, cvar);
    ASSERT_EQ(ret, -3);

    ret = fc_optim_var_parametric_batch(1, means, stddevs, 1.0, var, cvar);
    ASSERT_EQ(ret, -3);
}

TEST(var_parametric_from_returns_basic) {
    double returns[] = {0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.00, -0.01, 0.02};
    size_t n_periods = 10;
    double confidence = 0.95;
    double var, cvar;

    int ret = fc_optim_var_parametric_from_returns(returns, n_periods, confidence, &var, &cvar);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(cvar < var);
}

TEST(var_parametric_from_returns_consistency) {
    double returns[] = {0.01, -0.02, 0.03, -0.01, 0.02};
    size_t n_periods = 5;
    double confidence = 0.95;

    double var1, cvar1, var2, cvar2;

    // Method 1: Direct from returns
    int ret1 = fc_optim_var_parametric_from_returns(returns, n_periods, confidence, &var1, &cvar1);

    // Method 2: Manual mean/stddev calculation
    double sum = 0.0;
    for (size_t i = 0; i < n_periods; i++) {
        sum += returns[i];
    }
    double mean = sum / n_periods;

    double var_sum = 0.0;
    for (size_t i = 0; i < n_periods; i++) {
        double dev = returns[i] - mean;
        var_sum += dev * dev;
    }
    double stddev = sqrt(var_sum / (n_periods - 1));

    int ret2 = fc_optim_var_parametric(mean, stddev, confidence, &var2, &cvar2);

    ASSERT_EQ(ret1, 0);
    ASSERT_EQ(ret2, 0);
    ASSERT_TRUE(fabs(var1 - var2) < 1e-10);
    ASSERT_TRUE(fabs(cvar1 - cvar2) < 1e-10);
}

TEST(var_parametric_from_portfolio_returns_basic) {
    // 2 assets, 10 periods
    double returns[] = {
        0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.00, -0.01, 0.02,  // Asset 1
        -0.01, 0.02, -0.02, 0.01, -0.01, 0.02, 0.00, 0.01, -0.02, 0.01   // Asset 2
    };
    double weights[] = {0.6, 0.4};
    size_t dim = 2;
    size_t n_periods = 10;
    double confidence = 0.95;
    double var, cvar;

    int ret = fc_optim_var_parametric_from_portfolio_returns(
        returns, weights, dim, n_periods, confidence, &var, &cvar
    );

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(cvar < var);
}

TEST(var_parametric_from_returns_null_inputs) {
    double returns[] = {0.01, 0.02};
    double var, cvar;

    int ret = fc_optim_var_parametric_from_returns(NULL, 10, 0.95, &var, &cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_parametric_from_returns(returns, 10, 0.95, NULL, &cvar);
    ASSERT_EQ(ret, -1);

    ret = fc_optim_var_parametric_from_returns(returns, 10, 0.95, &var, NULL);
    ASSERT_EQ(ret, -1);
}

TEST(var_parametric_from_returns_insufficient_data) {
    double returns[] = {0.01};
    double var, cvar;

    int ret = fc_optim_var_parametric_from_returns(returns, 1, 0.95, &var, &cvar);
    ASSERT_EQ(ret, -2);
}

void register_var_parametric_tests(void) {
    RUN_TEST(var_parametric_basic);
    RUN_TEST(var_parametric_99_confidence);
    RUN_TEST(var_parametric_positive_mean);
    RUN_TEST(var_parametric_negative_mean);
    RUN_TEST(var_parametric_null_output);
    RUN_TEST(var_parametric_invalid_stddev);
    RUN_TEST(var_parametric_invalid_confidence);
    RUN_TEST(var_parametric_batch_basic);
    RUN_TEST(var_parametric_batch_large);
    RUN_TEST(var_parametric_batch_consistency);
    RUN_TEST(var_parametric_batch_null_inputs);
    RUN_TEST(var_parametric_batch_zero_size);
    RUN_TEST(var_parametric_batch_invalid_stddev);
    RUN_TEST(var_parametric_batch_invalid_confidence);
    RUN_TEST(var_parametric_from_returns_basic);
    RUN_TEST(var_parametric_from_returns_consistency);
    RUN_TEST(var_parametric_from_portfolio_returns_basic);
    RUN_TEST(var_parametric_from_returns_null_inputs);
    RUN_TEST(var_parametric_from_returns_insufficient_data);
}
