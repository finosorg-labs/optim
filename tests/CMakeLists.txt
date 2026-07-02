#
# tests/CMakeLists.txt - Test configuration for optim module
#

# Test executable
add_executable(optim_tests
    test_framework.c
    test_main.c
    test_optim.c
    test_least_squares.c
    test_greeks.c
)

# Link with optim library
target_link_libraries(optim_tests
    PRIVATE
        finkit_optim_static_base
     finkit::ds
        finkit::math
        finkit::platform
        m
)

target_compile_definitions(optim_tests PRIVATE
    FC_STATIC_BUILD
)

# Include directories
target_include_directories(optim_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/modules/platform/include
        ${CMAKE_SOURCE_DIR}/modules/math/include
        ${CMAKE_SOURCE_DIR}/modules/ds/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compiler flags for tests
target_compile_options(optim_tests PRIVATE -Wall -Wextra -Wpedantic)

# Run tests with ctest
add_test(
    NAME "optim_tests"
    COMMAND optim_tests
)

# Installation
install(TARGETS optim_tests
    RUNTIME DESTINATION bin
)
