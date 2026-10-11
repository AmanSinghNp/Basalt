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
include(CMakePackageConfigHelpers)
find_package(Threads REQUIRED)

option(BASALT_ENABLE_LIBNUMA "Enable optional libnuma integration when available" ON)
if(BASALT_ENABLE_LIBNUMA)
    find_library(BASALT_NUMA_LIBRARY numa)
endif()

# Google Test
set(INSTALL_GTEST OFF CACHE BOOL "Disable GoogleTest install rules for Basalt packaging" FORCE)
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
        -mtune=generic 
        -mavx2 
        -mfma 
    )
    list(APPEND BASALT_WARNING_FLAGS
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion
    )
endif()

# --- 3. Source Groups ---
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
    src/filter/fk_filter_avx512.cpp
    src/internal/batch_executor.cpp
    src/internal/fft2d_auto.cpp
    src/internal/numa_memory.cpp
    src/internal/numa_topology.cpp
    src/internal/thread_pool.cpp
    src/fk/fk_pipeline.cpp
    src/io/segy_file.cpp
)
target_include_directories(basalt_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
# Vendored pocketfft (BSD-3), header-only. SYSTEM so project warnings stay quiet.
target_include_directories(basalt_core SYSTEM PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/pocketfft>
)
target_compile_options(basalt_core PRIVATE ${BASALT_COMMON_FLAGS} ${BASALT_WARNING_FLAGS})
target_link_libraries(basalt_core PUBLIC Threads::Threads)
if(BASALT_NUMA_LIBRARY)
    target_link_libraries(basalt_core PUBLIC ${BASALT_NUMA_LIBRARY})
    target_compile_definitions(basalt_core PRIVATE BASALT_HAS_LIBNUMA=1)
endif()

# The CLI Executable
add_executable(basalt src/main.cpp)
target_link_libraries(basalt PRIVATE basalt_core)
target_compile_options(basalt PRIVATE ${BASALT_COMMON_FLAGS} ${BASALT_WARNING_FLAGS})
target_compile_definitions(basalt PRIVATE BASALT_VERSION="${PROJECT_VERSION}")

# --- 4. Installation ---
set(BASALT_INSTALL_CMAKEDIR "lib/cmake/Basalt")

install(TARGETS basalt_core basalt
    EXPORT BasaltTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(
    DIRECTORY include/basalt
    DESTINATION include
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "internal" EXCLUDE
        PATTERN "internal/*" EXCLUDE
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/BasaltConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BasaltConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/BasaltConfig.cmake"
    INSTALL_DESTINATION "${BASALT_INSTALL_CMAKEDIR}"
)

install(
    EXPORT BasaltTargets
    FILE BasaltTargets.cmake
    NAMESPACE Basalt::
    DESTINATION "${BASALT_INSTALL_CMAKEDIR}"
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/BasaltConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/BasaltConfigVersion.cmake"
    DESTINATION "${BASALT_INSTALL_CMAKEDIR}"
)

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
    tests/integrity/runtime_test.cpp
    tests/integration/benchmark_contract_test.cpp
    tests/integration/pipeline_test.cpp
    tests/integrity/fk_pipeline_test.cpp
    tests/integrity/segy_file_test.cpp
    tests/integration/cli_test.cpp
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
    target_compile_definitions(basalt_tests PRIVATE BASALT_BENCH_PATH="$<TARGET_FILE:basalt_bench>")
    target_compile_definitions(basalt_tests PRIVATE BASALT_CLI_PATH="$<TARGET_FILE:basalt>")

    add_executable(fk_gather_bench benchmarks/fk_gather_bench.cpp)
    target_link_libraries(fk_gather_bench PRIVATE basalt_core)
    target_compile_options(fk_gather_bench PRIVATE ${BASALT_COMMON_FLAGS} ${BASALT_WARNING_FLAGS})

    add_executable(fft_benchmark benchmarks/fft_benchmark.cpp)
    target_link_libraries(fft_benchmark PRIVATE basalt_core)
    target_compile_options(fft_benchmark PRIVATE ${BASALT_COMMON_FLAGS} ${BASALT_WARNING_FLAGS})

    add_executable(fft2d_schedule_benchmark benchmarks/fft2d_schedule_benchmark.cpp)
    target_link_libraries(fft2d_schedule_benchmark PRIVATE basalt_core)
    target_compile_options(fft2d_schedule_benchmark PRIVATE ${BASALT_COMMON_FLAGS} ${BASALT_WARNING_FLAGS})

    add_executable(fk_benchmark benchmarks/fk_benchmark.cpp)
    target_link_libraries(fk_benchmark PRIVATE basalt_core)
    target_compile_options(fk_benchmark PRIVATE ${BASALT_COMMON_FLAGS} ${BASALT_WARNING_FLAGS})
endif()

# --- 7. Python bindings ---
# Builds the `basalt._basalt` pybind11 module and runs the Python test suite under ctest.
# In-tree: PYTHONPATH=<build>/python python3 -c 'import basalt'. Wheels: scikit-build-core (pyproject.toml).
option(BASALT_BUILD_PYTHON "Build the basalt Python extension (basalt._basalt)" OFF)
if(BASALT_BUILD_PYTHON)
    if(CMAKE_VERSION VERSION_LESS 3.18)
        message(FATAL_ERROR "BASALT_BUILD_PYTHON requires CMake 3.18 or newer (FindPython Development.Module)")
    endif()
    # The extension is a shared module that links the static basalt_core library.
    set_target_properties(basalt_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

    # Outside scikit-build-core, ask the selected interpreter where pybind11's CMake config lives.
    if(NOT pybind11_DIR AND NOT SKBUILD)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE BASALT_PYBIND11_CMAKEDIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE BASALT_PYBIND11_QUERY
            ERROR_QUIET)
        if(BASALT_PYBIND11_QUERY EQUAL 0)
            set(pybind11_DIR "${BASALT_PYBIND11_CMAKEDIR}")
        endif()
    endif()
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_basalt python/src/basalt_py.cpp)
    target_link_libraries(_basalt PRIVATE basalt_core)
    target_compile_options(_basalt PRIVATE ${BASALT_COMMON_FLAGS} ${BASALT_WARNING_FLAGS})

    # In-tree layout: <build>/python/basalt/{__init__.py, _basalt*.so}, importable with PYTHONPATH=<build>/python.
    set(BASALT_PY_STAGE_DIR "${CMAKE_BINARY_DIR}/python/basalt")
    set_target_properties(_basalt PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${BASALT_PY_STAGE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${BASALT_PY_STAGE_DIR}")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/basalt/__init__.py"
                   "${BASALT_PY_STAGE_DIR}/__init__.py" COPYONLY)

    # Wheel layout: scikit-build-core copies python/basalt via wheel.packages, and this puts the
    # extension next to it in the installed package.
    if(SKBUILD)
        install(TARGETS _basalt DESTINATION basalt COMPONENT python)
    endif()

    # Python tests: plain asserts, runnable directly or under pytest. Needs no pytest for ctest.
    add_test(NAME basalt_python_tests
             COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/tests/test_basalt.py")
    set_tests_properties(basalt_python_tests PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
        TIMEOUT 600)
endif()
