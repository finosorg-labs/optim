/**
 * @file test_greeks.c
 * @brief Unit tests for Black-Scholes Greeks batch calculation
 */

#include "greeks.h"
#include "test_framework.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

TEST(greeks_call_basic) {
    double S[] = {100.0};
    double K[] = {100.0};
    double T[] = {1.0};
    double r[] = {0.05};
    double sigma[] = {0.2};
    int is_call[] = {1};
    double delta[1] = {0}, gamma[1] = {0}, vega[1] = {0}, theta[1] = {0}, rho[1] = {0};

    int ret = fc_optim_greeks_batch(1, S, K, T, r, sigma, is_call, delta, gamma, vega, theta, rho);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(delta[0] > 0.63 && delta[0] < 0.64);
    ASSERT_TRUE(gamma[0] > 0.018 && gamma[0] < 0.019);
    ASSERT_TRUE(vega[0] > 37.0 && vega[0] < 38.0);
    ASSERT_TRUE(theta[0] < 0.0);
    ASSERT_TRUE(rho[0] > 0.0);
}

TEST(greeks_put_basic) {
    double S[] = {100.0};
    double K[] = {100.0};
    double T[] = {1.0};
    double r[] = {0.05};
    double sigma[] = {0.2};
    int is_call[] = {0};
    double delta[1] = {0}, gamma[1] = {0}, vega[1] = {0}, theta[1] = {0}, rho[1] = {0};

    int ret = fc_optim_greeks_batch(1, S, K, T, r, sigma, is_call, delta, gamma, vega, theta, rho);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(delta[0] < -0.36 && delta[0] > -0.37);
    ASSERT_TRUE(gamma[0] > 0.018 && gamma[0] < 0.019);
    ASSERT_TRUE(vega[0] > 37.0 && vega[0] < 38.0);
    ASSERT_TRUE(theta[0] < 0.0);
    ASSERT_TRUE(rho[0] < 0.0);
}

TEST(greeks_itm_call) {
    double S[] = {110.0};
    double K[] = {100.0};
    double T[] = {0.5};
    double r[] = {0.05};
    double sigma[] = {0.25};
    int is_call[] = {1};
    double delta[1] = {0}, gamma[1] = {0}, vega[1] = {0}, theta[1] = {0}, rho[1] = {0};

    int ret = fc_optim_greeks_batch(1, S, K, T, r, sigma, is_call, delta, gamma, vega, theta, rho);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(delta[0] > 0.7);
    ASSERT_TRUE(gamma[0] > 0.0);
    ASSERT_TRUE(vega[0] > 0.0);
    ASSERT_TRUE(theta[0] < 0.0);
    ASSERT_TRUE(rho[0] > 0.0);
}

TEST(greeks_otm_put) {
    double S[] = {110.0};
    double K[] = {100.0};
    double T[] = {0.5};
    double r[] = {0.05};
    double sigma[] = {0.25};
    int is_call[] = {0};
    double delta[1] = {0}, gamma[1] = {0}, vega[1] = {0}, theta[1] = {0}, rho[1] = {0};

    int ret = fc_optim_greeks_batch(1, S, K, T, r, sigma, is_call, delta, gamma, vega, theta, rho);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(delta[0] < 0.0 && delta[0] > -0.3);
    ASSERT_TRUE(gamma[0] > 0.0);
    ASSERT_TRUE(vega[0] > 0.0);
    ASSERT_TRUE(rho[0] < 0.0);
}

TEST(greeks_batch_mixed) {
    double S[] = {100.0, 110.0, 90.0, 100.0};
    double K[] = {100.0, 100.0, 100.0, 100.0};
    double T[] = {1.0, 0.5, 1.0, 0.25};
    double r[] = {0.05, 0.05, 0.05, 0.05};
    double sigma[] = {0.2, 0.25, 0.3, 0.15};
    int is_call[] = {1, 1, 0, 0};
    double delta[4] = {0}, gamma[4] = {0}, vega[4] = {0}, theta[4] = {0}, rho[4] = {0};

    int ret = fc_optim_greeks_batch(4, S, K, T, r, sigma, is_call, delta, gamma, vega, theta, rho);

    ASSERT_EQ(ret, 0);

    ASSERT_TRUE(delta[0] > 0.63 && delta[0] < 0.64);
    ASSERT_TRUE(delta[1] > 0.7);
    ASSERT_TRUE(delta[2] < 0.0);
    ASSERT_TRUE(delta[3] < 0.0);

    for (size_t i = 0; i < 4; i++) {
        ASSERT_TRUE(gamma[i] > 0.0);
        ASSERT_TRUE(vega[i] > 0.0);
        ASSERT_TRUE(theta[i] < 0.0);
    }

    ASSERT_TRUE(rho[0] > 0.0);
    ASSERT_TRUE(rho[1] > 0.0);
    ASSERT_TRUE(rho[2] < 0.0);
    ASSERT_TRUE(rho[3] < 0.0);
}

TEST(greeks_null_inputs) {
    double S[] = {100.0};
    double K[] = {100.0};
    double T[] = {1.0};
    double r[] = {0.05};
    double sigma[] = {0.2};
    int is_call[] = {1};
    double delta[1];

    ASSERT_EQ(fc_optim_greeks_batch(1, NULL, K, T, r, sigma, is_call, delta, NULL, NULL, NULL, NULL), -1);
    ASSERT_EQ(fc_optim_greeks_batch(1, S, NULL, T, r, sigma, is_call, delta, NULL, NULL, NULL, NULL), -1);
    ASSERT_EQ(fc_optim_greeks_batch(1, S, K, NULL, r, sigma, is_call, delta, NULL, NULL, NULL, NULL), -1);
    ASSERT_EQ(fc_optim_greeks_batch(1, S, K, T, NULL, sigma, is_call, delta, NULL, NULL, NULL, NULL), -1);
    ASSERT_EQ(fc_optim_greeks_batch(1, S, K, T, r, NULL, is_call, delta, NULL, NULL, NULL, NULL), -1);
    ASSERT_EQ(fc_optim_greeks_batch(1, S, K, T, r, sigma, NULL, delta, NULL, NULL, NULL, NULL), -1);
}

TEST(greeks_zero_size) {
    double S[] = {100.0};
    double K[] = {100.0};
    double T[] = {1.0};
    double r[] = {0.05};
    double sigma[] = {0.2};
    int is_call[] = {1};
    double delta[1];

    ASSERT_EQ(fc_optim_greeks_batch(0, S, K, T, r, sigma, is_call, delta, NULL, NULL, NULL, NULL), -2);
}

TEST(greeks_all_null_outputs) {
    double S[] = {100.0};
    double K[] = {100.0};
    double T[] = {1.0};
    double r[] = {0.05};
    double sigma[] = {0.2};
    int is_call[] = {1};

    ASSERT_EQ(fc_optim_greeks_batch(1, S, K, T, r, sigma, is_call, NULL, NULL, NULL, NULL, NULL), 0);
}

TEST(greeks_invalid_spot) {
    double S[] = {0.0};
    double K[] = {100.0};
    double T[] = {1.0};
    double r[] = {0.05};
    double sigma[] = {0.2};
    int is_call[] = {1};
    double delta[1];

    ASSERT_EQ(fc_optim_greeks_batch(1, S, K, T, r, sigma, is_call, delta, NULL, NULL, NULL, NULL), -3);
}

TEST(greeks_invalid_strike) {
    double S[] = {100.0};
    double K[] = {-10.0};
    double T[] = {1.0};
    double r[] = {0.05};
    double sigma[] = {0.2};
    int is_call[] = {1};
    double delta[1];

    ASSERT_EQ(fc_optim_greeks_batch(1, S, K, T, r, sigma, is_call, delta, NULL, NULL, NULL, NULL), -3);
}

TEST(greeks_invalid_time) {
    double S[] = {100.0};
    double K[] = {100.0};
    double T[] = {0.0};
    double r[] = {0.05};
    double sigma[] = {0.2};
    int is_call[] = {1};
    double delta[1];

    ASSERT_EQ(fc_optim_greeks_batch(1, S, K, T, r, sigma, is_call, delta, NULL, NULL, NULL, NULL), -3);
}

TEST(greeks_invalid_volatility) {
    double S[] = {100.0};
    double K[] = {100.0};
    double T[] = {1.0};
    double r[] = {0.05};
    double sigma[] = {0.0};
    int is_call[] = {1};
    double delta[1];

    ASSERT_EQ(fc_optim_greeks_batch(1, S, K, T, r, sigma, is_call, delta, NULL, NULL, NULL, NULL), -3);
}

TEST(greeks_partial_outputs) {
    double S[] = {100.0};
    double K[] = {100.0};
    double T[] = {1.0};
    double r[] = {0.05};
    double sigma[] = {0.2};
    int is_call[] = {1};
    double delta[1], vega[1];

    int ret = fc_optim_greeks_batch(1, S, K, T, r, sigma, is_call, delta, NULL, vega, NULL, NULL);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(delta[0] > 0.63 && delta[0] < 0.64);
    ASSERT_TRUE(vega[0] > 37.0 && vega[0] < 38.0);
}

TEST(greeks_large_batch) {
    const size_t n = 1000;
    double* S = (double*)malloc(n * sizeof(double));
    double* K = (double*)malloc(n * sizeof(double));
    double* T = (double*)malloc(n * sizeof(double));
    double* r = (double*)malloc(n * sizeof(double));
    double* sigma = (double*)malloc(n * sizeof(double));
    int* is_call = (int*)malloc(n * sizeof(int));
    double* delta = (double*)malloc(n * sizeof(double));
    double* gamma = (double*)malloc(n * sizeof(double));
    double* vega = (double*)malloc(n * sizeof(double));
    double* theta = (double*)malloc(n * sizeof(double));
    double* rho = (double*)malloc(n * sizeof(double));

    for (size_t i = 0; i < n; i++) {
        S[i] = 100.0 + (i % 50) - 25.0;
        K[i] = 100.0;
        T[i] = 0.25 + (i % 8) * 0.125;
        r[i] = 0.05;
        sigma[i] = 0.2 + (i % 10) * 0.01;
        is_call[i] = (i % 2);
    }

    int ret = fc_optim_greeks_batch(n, S, K, T, r, sigma, is_call, delta, gamma, vega, theta, rho);

    ASSERT_EQ(ret, 0);

    for (size_t i = 0; i < n; i++) {
        ASSERT_TRUE(gamma[i] > 0.0);
        ASSERT_TRUE(vega[i] > 0.0);

        if (is_call[i]) {
            ASSERT_TRUE(rho[i] > 0.0 || (S[i] < K[i] && fabs(rho[i]) < 1.0));
        } else {
            ASSERT_TRUE(rho[i] < 0.0 || (S[i] > K[i] && fabs(rho[i]) < 1.0));
        }
    }

    free(S);
    free(K);
    free(T);
    free(r);
    free(sigma);
    free(is_call);
    free(delta);
    free(gamma);
    free(vega);
    free(theta);
    free(rho);
}

TEST(greeks_gamma_vega_symmetry) {
    double S[] = {100.0, 100.0};
    double K[] = {100.0, 100.0};
    double T[] = {1.0, 1.0};
    double r[] = {0.05, 0.05};
    double sigma[] = {0.2, 0.2};
    int is_call[] = {1, 0};
    double gamma[2], vega[2];

    int ret = fc_optim_greeks_batch(2, S, K, T, r, sigma, is_call, NULL, gamma, vega, NULL, NULL);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(fabs(gamma[0] - gamma[1]) < 1e-10);
    ASSERT_TRUE(fabs(vega[0] - vega[1]) < 1e-10);
}

TEST(greeks_short_expiry) {
    double S[] = {100.0};
    double K[] = {100.0};
    double T[] = {0.01};
    double r[] = {0.05};
    double sigma[] = {0.2};
    int is_call[] = {1};
    double delta[1] = {0}, gamma[1] = {0}, vega[1] = {0}, theta[1] = {0}, rho[1] = {0};

    int ret = fc_optim_greeks_batch(1, S, K, T, r, sigma, is_call, delta, gamma, vega, theta, rho);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(delta[0] > 0.4);
    ASSERT_TRUE(gamma[0] > 0.0);
    ASSERT_TRUE(vega[0] > 0.0);
}

TEST(greeks_high_volatility) {
    double S[] = {100.0};
    double K[] = {100.0};
    double T[] = {1.0};
    double r[] = {0.05};
    double sigma[] = {1.0};
    int is_call[] = {1};
    double delta[1] = {0}, gamma[1] = {0}, vega[1] = {0}, theta[1] = {0}, rho[1] = {0};

    int ret = fc_optim_greeks_batch(1, S, K, T, r, sigma, is_call, delta, gamma, vega, theta, rho);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(delta[0] > 0.5);
    ASSERT_TRUE(gamma[0] > 0.0);
    ASSERT_TRUE(vega[0] > 0.0);
}

TEST(greeks_invalid_rate) {
    double S[] = {100.0};
    double K[] = {100.0};
    double T[] = {1.0};
    double r[] = {NAN};  // Invalid rate
    double sigma[] = {0.2};
    int is_call[] = {1};
    double delta[1];

    int ret = fc_optim_greeks_batch(1, S, K, T, r, sigma, is_call, delta, NULL, NULL, NULL, NULL);

    ASSERT_EQ(ret, -3);
}

TEST(greeks_infinite_rate) {
    double S[] = {100.0};
    double K[] = {100.0};
    double T[] = {1.0};
    double r[] = {INFINITY};  // Invalid rate
    double sigma[] = {0.2};
    int is_call[] = {1};
    double delta[1];

    int ret = fc_optim_greeks_batch(1, S, K, T, r, sigma, is_call, delta, NULL, NULL, NULL, NULL);

    ASSERT_EQ(ret, -3);
}

TEST(greeks_negative_rate) {
    // Negative rates are valid (e.g., European markets)
    double S[] = {100.0};
    double K[] = {100.0};
    double T[] = {1.0};
    double r[] = {-0.01};  // -1% rate (valid)
    double sigma[] = {0.2};
    int is_call[] = {1};
    double delta[1];

    int ret = fc_optim_greeks_batch(1, S, K, T, r, sigma, is_call, delta, NULL, NULL, NULL, NULL);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(delta[0] > 0.0 && delta[0] < 1.0);
}

TEST(greeks_exact_atm_call) {
    /* ATM Call: S=100, K=100, T=1, r=0.05, sigma=0.2
     * d1 = (ln(100/100) + (0.05 + 0.2^2/2)*1) / (0.2*sqrt(1)) = 0.35
     * d2 = 0.35 - 0.2 = 0.15
     * N(d1) ≈ 0.6368, N(d2) ≈ 0.5596
     * Delta = N(d1) ≈ 0.6368
     * Gamma = N'(d1) / (S*sigma*sqrt(T)) = 0.3752 / (100*0.2*1) ≈ 0.01876
     * Vega = S * N'(d1) * sqrt(T) = 100 * 0.3752 * 1 ≈ 37.52
     */
    double S[] = {100.0};
    double K[] = {100.0};
    double T[] = {1.0};
    double r[] = {0.05};
    double sigma[] = {0.2};
    int is_call[] = {1};
    double delta[1], gamma[1], vega[1], theta[1], rho[1];

    int ret = fc_optim_greeks_batch(1, S, K, T, r, sigma, is_call, delta, gamma, vega, theta, rho);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(fabs(delta[0] - 0.6368) < 0.001);
    ASSERT_TRUE(fabs(gamma[0] - 0.01876) < 0.0001);
    ASSERT_TRUE(fabs(vega[0] - 37.52) < 0.1);
    ASSERT_TRUE(theta[0] < 0.0);
    ASSERT_TRUE(rho[0] > 0.0);
}

TEST(greeks_exact_atm_put) {
    /* ATM Put: S=100, K=100, T=1, r=0.05, sigma=0.2
     * d1 ≈ 0.35, d2 ≈ 0.15
     * Delta = N(d1) - 1 ≈ 0.6368 - 1 = -0.3632
     * Gamma (same for call and put) ≈ 0.01876
     * Vega (same for call and put) ≈ 37.52
     */
    double S[] = {100.0};
    double K[] = {100.0};
    double T[] = {1.0};
    double r[] = {0.05};
    double sigma[] = {0.2};
    int is_call[] = {0};
    double delta[1], gamma[1], vega[1], theta[1], rho[1];

    int ret = fc_optim_greeks_batch(1, S, K, T, r, sigma, is_call, delta, gamma, vega, theta, rho);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(fabs(delta[0] - (-0.3632)) < 0.001);
    ASSERT_TRUE(fabs(gamma[0] - 0.01876) < 0.0001);
    ASSERT_TRUE(fabs(vega[0] - 37.52) < 0.1);
    ASSERT_TRUE(theta[0] < 0.0);
    ASSERT_TRUE(rho[0] < 0.0);
}

TEST(greeks_exact_deep_itm_call) {
    /* Deep ITM Call: S=150, K=100, T=1, r=0.05, sigma=0.2
     * Delta should be close to 1.0
     * Gamma should be small (option behaves like stock)
     */
    double S[] = {150.0};
    double K[] = {100.0};
    double T[] = {1.0};
    double r[] = {0.05};
    double sigma[] = {0.2};
    int is_call[] = {1};
    double delta[1], gamma[1], vega[1];

    int ret = fc_optim_greeks_batch(1, S, K, T, r, sigma, is_call, delta, gamma, vega, NULL, NULL);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(delta[0] > 0.95);  // Very close to 1
    ASSERT_TRUE(gamma[0] < 0.01);  // Small gamma
    ASSERT_TRUE(vega[0] > 0.0);
}

TEST(greeks_exact_deep_otm_call) {
    /* Deep OTM Call: S=50, K=100, T=1, r=0.05, sigma=0.2
     * Delta should be close to 0.0
     * Gamma should be small
     */
    double S[] = {50.0};
    double K[] = {100.0};
    double T[] = {1.0};
    double r[] = {0.05};
    double sigma[] = {0.2};
    int is_call[] = {1};
    double delta[1], gamma[1], vega[1];

    int ret = fc_optim_greeks_batch(1, S, K, T, r, sigma, is_call, delta, gamma, vega, NULL, NULL);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(delta[0] < 0.05);  // Very close to 0
    ASSERT_TRUE(gamma[0] < 0.01);  // Small gamma
    ASSERT_TRUE(vega[0] > 0.0);
}

TEST(greeks_exact_put_call_parity_gamma_vega) {
    /* Put-Call Parity for Gamma and Vega:
     * Gamma_call = Gamma_put
     * Vega_call = Vega_put
     * Test with S=110, K=100, T=0.5, r=0.05, sigma=0.25
     */
    double S[] = {110.0, 110.0};
    double K[] = {100.0, 100.0};
    double T[] = {0.5, 0.5};
    double r[] = {0.05, 0.05};
    double sigma[] = {0.25, 0.25};
    int is_call[] = {1, 0};  // Call and Put
    double gamma[2], vega[2];

    int ret = fc_optim_greeks_batch(2, S, K, T, r, sigma, is_call, NULL, gamma, vega, NULL, NULL);

    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(fabs(gamma[0] - gamma[1]) < 1e-10);  // Gamma should be identical
    ASSERT_TRUE(fabs(vega[0] - vega[1]) < 1e-10);    // Vega should be identical
}

void register_greeks_tests(void) {
    RUN_TEST(greeks_call_basic);
    RUN_TEST(greeks_put_basic);
    RUN_TEST(greeks_itm_call);
    RUN_TEST(greeks_otm_put);
    RUN_TEST(greeks_batch_mixed);
    RUN_TEST(greeks_null_inputs);
    RUN_TEST(greeks_zero_size);
    RUN_TEST(greeks_all_null_outputs);
    RUN_TEST(greeks_invalid_spot);
    RUN_TEST(greeks_invalid_strike);
    RUN_TEST(greeks_invalid_time);
    RUN_TEST(greeks_invalid_volatility);
    RUN_TEST(greeks_invalid_rate);
    RUN_TEST(greeks_infinite_rate);
    RUN_TEST(greeks_negative_rate);
    RUN_TEST(greeks_partial_outputs);
    RUN_TEST(greeks_large_batch);
    RUN_TEST(greeks_gamma_vega_symmetry);
    RUN_TEST(greeks_short_expiry);
    RUN_TEST(greeks_high_volatility);
    RUN_TEST(greeks_exact_atm_call);
    RUN_TEST(greeks_exact_atm_put);
    RUN_TEST(greeks_exact_deep_itm_call);
    RUN_TEST(greeks_exact_deep_otm_call);
    RUN_TEST(greeks_exact_put_call_parity_gamma_vega);
}
