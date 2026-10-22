# Standalone consumer used to verify the installed Basalt package.
#   cmake -S tests/package -B build-pkg -DCMAKE_PREFIX_PATH=<install prefix>
cmake_minimum_required(VERSION 3.15...3.28)
project(BasaltPackageConsumer LANGUAGES CXX)

find_package(Basalt REQUIRED)

add_executable(basalt_package_consumer main.cpp)
target_link_libraries(basalt_package_consumer PRIVATE Basalt::basalt_core)

enable_testing()
add_test(NAME basalt_package_consumer COMMAND basalt_package_consumer)
