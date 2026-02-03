cmake_minimum_required(VERSION 3.15)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Project Identity ---
project(Basalt 
    VERSION 0.1.0 
    DESCRIPTION "High-Performance Seismic f-k Muter (x86_64 Optimized)"
    LANGUAGES CXX C
)

# --- 1. The Iron Foundation (Compiler Flags) ---
# We target Haswell+ (AVX2/FMA) architecture explicitly.
if(MSVC)
    add_compile_options(/std:c++17 /O2 /arch:AVX2 /W4 /permissive-)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(
        -std=c++17
        -O3 
        -march=native 
        -mtune=native 
        -mavx2 
        -mfma 
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion
    )
endif()

# --- 2. Safety Net (Sanitizers) ---
# Only enable in Debug builds to catch leaks/overflows during dev
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT MSVC)
    # message(STATUS "Basalt: Enabling Address Sanitizer")
    # add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    # add_link_options(-fsanitize=address)
endif()

# --- 3. Dependencies (Google Benchmark/Test) ---
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
# For now, we make GTest available but don't link it yet
FetchContent_MakeAvailable(googletest)

# --- 4. Source Groups ---
include_directories(include)

# The Core Engine Library
add_library(basalt_core
    src/io/segy_parser.cpp
    src/io/file_mapping.cpp
    src/utils/ebcdic.cpp
    # src/kernel/fft_avx.cpp  (To be created)
    src/core/arena.cpp
)

# The CLI Executable
add_executable(basalt src/main.cpp)
target_link_libraries(basalt PRIVATE basalt_core)

# --- 5. Installation ---
install(TARGETS basalt DESTINATION bin)

# --- 6. Testing ---
enable_testing()

add_executable(basalt_tests
    tests/integrity/arena_test.cpp
    tests/integrity/segy_header_test.cpp
    tests/integrity/file_mapping_test.cpp
)
target_link_libraries(basalt_tests PRIVATE basalt_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(basalt_tests)

