cmake_minimum_required(VERSION 3.15)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Project Identity ---
project(Basalt 
    VERSION 0.1.0 
    DESCRIPTION "High-Performance Seismic f-k Muter (x86_64 Optimized)"
    HOMEPAGE_URL "https://github.com/AmanSinghNp/Basalt"
    LANGUAGES CXX C
)

# --- 1. Dependencies ---
include(FetchContent)

# Google Test
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

# --- 2. Compiler Flags ---
set(BASALT_COMMON_FLAGS "")
set(BASALT_WARNING_FLAGS "")

if(MSVC)
    list(APPEND BASALT_COMMON_FLAGS /std:c++17 /O2 /arch:AVX2 /permissive-)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    list(APPEND BASALT_WARNING_FLAGS /W4)
else()
    list(APPEND BASALT_COMMON_FLAGS 
        -std=c++17 
        -O3 
        -march=native 
        -mtune=native 
        -mavx2 
        -mfma 
    )
    list(APPEND BASALT_WARNING_FLAGS
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion
    )
endif()

# --- 3. Source Groups ---
include_directories(include)

# The Core Engine Library
add_library(basalt_core
    src/io/segy_parser.cpp
    src/io/file_mapping.cpp
    src/utils/ebcdic.cpp
    src/utils/trace_utils.cpp
    src/kernel/fft.cpp
    src/kernel/transpose.cpp
    src/kernel/fft2d.cpp
    src/kernel/fft_avx.cpp
    src/kernel/transpose_avx.cpp
    src/core/arena.cpp
    src/filter/fk_filter.cpp
)
target_compile_options(basalt_core PRIVATE ${BASALT_COMMON_FLAGS} ${BASALT_WARNING_FLAGS})

# The CLI Executable
add_executable(basalt src/main.cpp)
target_link_libraries(basalt PRIVATE basalt_core)
target_compile_options(basalt PRIVATE ${BASALT_COMMON_FLAGS} ${BASALT_WARNING_FLAGS})
target_compile_definitions(basalt PRIVATE BASALT_VERSION="${PROJECT_VERSION}")

# --- 4. Installation ---
install(TARGETS basalt DESTINATION bin)

# --- 5. Testing ---
enable_testing()

add_executable(basalt_tests
    tests/integrity/arena_test.cpp
    tests/integrity/segy_header_test.cpp
    tests/integrity/file_mapping_test.cpp
    tests/integrity/ibm_float_test.cpp
    tests/integrity/trace_utils_test.cpp
    tests/integrity/fft_test.cpp
    tests/integrity/transpose_test.cpp
    tests/integrity/fft2d_test.cpp
    tests/integrity/fft_avx_test.cpp
    tests/integrity/fk_filter_test.cpp
    tests/integration/pipeline_test.cpp
)
target_link_libraries(basalt_tests PRIVATE basalt_core GTest::gtest_main)
target_compile_options(basalt_tests PRIVATE ${BASALT_COMMON_FLAGS} ${BASALT_WARNING_FLAGS})

include(GoogleTest)
gtest_discover_tests(basalt_tests)

# --- 6. Benchmarks ---
if(CMAKE_PROJECT_NAME STREQUAL "Basalt")
    add_executable(basalt_bench benchmarks/basalt_bench.cpp)
    target_link_libraries(basalt_bench PRIVATE basalt_core)
    target_compile_options(basalt_bench PRIVATE ${BASALT_COMMON_FLAGS} ${BASALT_WARNING_FLAGS})

    add_executable(fft_benchmark benchmarks/fft_benchmark.cpp)
    target_link_libraries(fft_benchmark PRIVATE basalt_core)
    target_compile_options(fft_benchmark PRIVATE ${BASALT_COMMON_FLAGS} ${BASALT_WARNING_FLAGS})

    add_executable(fk_benchmark benchmarks/fk_benchmark.cpp)
    target_link_libraries(fk_benchmark PRIVATE basalt_core)
    target_compile_options(fk_benchmark PRIVATE ${BASALT_COMMON_FLAGS} ${BASALT_WARNING_FLAGS})
endif()
