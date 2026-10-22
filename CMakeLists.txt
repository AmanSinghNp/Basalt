cmake_minimum_required(VERSION 3.15...3.28)

project(Basalt
    VERSION 0.1.0
    DESCRIPTION "High-Performance Seismic f-k Muter (x86_64 Optimized)"
    HOMEPAGE_URL "https://github.com/AmanSinghNp/Basalt"
    LANGUAGES CXX
)

# PROJECT_IS_TOP_LEVEL needs CMake 3.21; compute it by hand for 3.15 compatibility.
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(BASALT_IS_TOP_LEVEL ON)
else()
    set(BASALT_IS_TOP_LEVEL OFF)
endif()

# --- Options ---
option(BASALT_BUILD_TESTS      "Build the Basalt unit and integration tests"      ${BASALT_IS_TOP_LEVEL})
option(BASALT_BUILD_BENCHMARKS "Build the Basalt benchmark executables"           ${BASALT_IS_TOP_LEVEL})
option(BASALT_BUILD_CLI        "Build the basalt command line executable"         ON)
option(BASALT_INSTALL          "Generate install rules and the CMake package"     ${BASALT_IS_TOP_LEVEL})
option(BASALT_NATIVE_ARCH      "Compile Basalt itself with -march=native (non-portable binaries)" OFF)

# Single-config generators: default to an optimized build when nothing was requested.
if(BASALT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

if(BASALT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

include(GNUInstallDirs)

# --- 1. Dependencies ---
find_package(Threads REQUIRED)

# NOTE: Basalt does not link libnuma. NUMA placement uses the raw mbind syscall
# (see src/internal/numa_memory.cpp), so there is no libnuma option anymore.

# --- 2. Compiler Flags ---
# Architecture flags live in ONE variable and are applied in ONE place
# (basalt_configure_target below). The baseline is plain x86-64 (SSE2): AVX2,
# FMA and AVX-512 code carries per-function target attributes and is selected
# at runtime (see include/basalt/kernel/cpu_features.hpp), so the library and
# the CLI run on any x86-64 CPU. BASALT_NATIVE_ARCH=ON additionally compiles
# everything for the build machine's CPU (runtime dispatch still applies, but
# the binary may then fail to start on older CPUs).
if(MSVC)
    set(BASALT_ARCH_FLAGS "")
    if(BASALT_NATIVE_ARCH)
        list(APPEND BASALT_ARCH_FLAGS /arch:AVX2)  # MSVC has no "native" switch
    endif()
    set(BASALT_WARNING_FLAGS /W4 /permissive-)
else()
    set(BASALT_ARCH_FLAGS -mtune=generic)
    if(BASALT_NATIVE_ARCH)
        list(APPEND BASALT_ARCH_FLAGS -march=native)
    endif()
    set(BASALT_WARNING_FLAGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion)
endif()

# Optimization level and -std come from CMAKE_BUILD_TYPE and cxx_std_17, never from here.
function(basalt_configure_target target)
    target_compile_features(${target} PUBLIC cxx_std_17)
    set_target_properties(${target} PROPERTIES CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    target_compile_options(${target} PRIVATE ${BASALT_ARCH_FLAGS} ${BASALT_WARNING_FLAGS})
    if(MSVC)
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
endfunction()

# --- 3. Core library ---
add_library(basalt_core
    src/io/segy_parser.cpp
    src/io/file_mapping.cpp
    src/utils/ebcdic.cpp
    src/utils/trace_utils.cpp
    src/kernel/fft.cpp
    src/kernel/transpose.cpp
    src/kernel/fft2d.cpp
    src/kernel/fft_avx.cpp
    src/kernel/cpu_features.cpp
    src/kernel/transpose_avx.cpp
    src/core/arena.cpp
    src/filter/fk_filter.cpp
    src/filter/fk_filter_avx512.cpp
    src/internal/batch_executor.cpp
    src/internal/fft2d_auto.cpp
    src/internal/numa_memory.cpp
    src/internal/numa_topology.cpp
    src/internal/thread_pool.cpp
)

add_library(Basalt::basalt_core ALIAS basalt_core)
target_include_directories(basalt_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(basalt_core PUBLIC Threads::Threads)
basalt_configure_target(basalt_core)
set_target_properties(basalt_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# The CLI executable (not part of the exported package targets)
if(BASALT_BUILD_CLI)
    add_executable(basalt src/main.cpp)
    target_link_libraries(basalt PRIVATE basalt_core)
    basalt_configure_target(basalt)
    target_compile_definitions(basalt PRIVATE BASALT_VERSION="${PROJECT_VERSION}")
endif()

# --- 4. Installation ---
if(BASALT_INSTALL)
    include(CMakePackageConfigHelpers)
    set(BASALT_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/Basalt")

    install(TARGETS basalt_core
        EXPORT BasaltTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    if(BASALT_BUILD_CLI)
        install(TARGETS basalt RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()

    install(
        DIRECTORY include/basalt
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
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
endif()

# --- 5. Benchmarks ---
# basalt_bench is also needed by tests/integration/benchmark_contract_test.cpp,
# so it is built whenever tests or benchmarks are requested.
if(BASALT_BUILD_BENCHMARKS OR BASALT_BUILD_TESTS)
    add_executable(basalt_bench benchmarks/basalt_bench.cpp)
    target_link_libraries(basalt_bench PRIVATE basalt_core)
    basalt_configure_target(basalt_bench)
endif()

if(BASALT_BUILD_BENCHMARKS)
    foreach(bench_name fft_benchmark fft2d_schedule_benchmark fk_benchmark)
        add_executable(${bench_name} benchmarks/${bench_name}.cpp)
        target_link_libraries(${bench_name} PRIVATE basalt_core)
        basalt_configure_target(${bench_name})
    endforeach()
endif()

# --- 6. Testing ---
if(BASALT_BUILD_TESTS)
    enable_testing()

    # Prefer an installed GoogleTest; otherwise fetch a pinned release.
    find_package(GTest CONFIG QUIET)
    if(NOT TARGET GTest::gtest_main)
        include(FetchContent)
        # Plain (non-cache) variables so a parent project's cache is left alone.
        set(INSTALL_GTEST OFF)
        set(gtest_force_shared_crt ON)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0  # f8d7d77c06936315286eb55f8de22cd23c188571
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(googletest)
    endif()

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
    )

    target_link_libraries(basalt_tests PRIVATE basalt_core GTest::gtest_main)
    basalt_configure_target(basalt_tests)

    if(TARGET basalt_bench)
        add_dependencies(basalt_tests basalt_bench)
        target_compile_definitions(basalt_tests PRIVATE BASALT_BENCH_PATH="$<TARGET_FILE:basalt_bench>")
    endif()

    include(GoogleTest)
    gtest_discover_tests(basalt_tests)
endif()
