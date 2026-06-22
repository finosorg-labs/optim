/**
 * @file test_optim.c
 * @brief optim module test entry point
 *
 * This file serves as the main test registration point for the optim module.
 * Individual test modules are in separate files:
 */

#include "test_framework.h"

/* External test registration functions from sub-modules */
extern void register_least_squares_tests(void);

/* Entry point for optim tests */
void register_optim_tests(void) {
    /* Register all sub-module tests */
    register_least_squares_tests();
}
