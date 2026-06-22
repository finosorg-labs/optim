#
# benchmarks/CMakeLists.txt - Benchmark configuration for optim module
#

# Benchmark framework library
add_library(bench_framework OBJECT
    bench_framework.c
)

target_include_directories(bench_framework
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/modules/platform/include
)

target_link_libraries(bench_framework
    PUBLIC
        finkit_optim_static_base
        finkit::platform
      finkit::ds
)

# Benchmark executable
add_executable(optim_benchmarks
    bench_main.c
    bench_optim.c
    bench_least_squares.c
)

# Link with benchmark framework and optim library
target_link_libraries(optim_benchmarks
    PRIVATE
        bench_framework
        finkit_optim_static_base
        finkit::platform
        finkit::ds
        m
)

target_compile_definitions(optim_benchmarks PRIVATE
    FC_STATIC_BUILD
)

# Include directories
target_include_directories(optim_benchmarks
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/modules/platform/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compiler flags for benchmarks
target_compile_options(optim_benchmarks PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)

# Installation
install(TARGETS optim_benchmarks
    RUNTIME DESTINATION bin
)
