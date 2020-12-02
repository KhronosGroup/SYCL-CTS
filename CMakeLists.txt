# check CMake Version
cmake_minimum_required(VERSION 3.13)
project(sycl_cts)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON) # Required for hex floats in C++11 mode on gcc 6+
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(OpenCLTargets REQUIRED)
find_package(PythonInterp 3 REQUIRED)

# set host compiler flags
if(WIN32)
    add_compile_options(/bigobj)
endif()

if(${CMAKE_CXX_COMPILER_ID} MATCHES "GNU" OR
   ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    add_compile_options(-Wall -Wno-unused-variable)
endif()


include(AddSYCLExecutable)

# ------------------
# Extensive mode for improved coverage
option(SYCL_CTS_ENABLE_EXTENSIVE_MODE
       "Enable extensive mode with extended type coverage" OFF)
if(SYCL_CTS_ENABLE_EXTENSIVE_MODE)
    add_definitions(-DSYCL_CTS_EXTENSIVE_MODE)
endif()
# ------------------

# ------------------
# Double and Half variables
option(SYCL_CTS_ENABLE_DOUBLE_TESTS "Enable Double tests." ON)
option(SYCL_CTS_ENABLE_HALF_TESTS "Enable Half tests." ON)
if(SYCL_CTS_ENABLE_DOUBLE_TESTS)
    add_definitions(-DSYCL_CTS_TEST_DOUBLE)
endif()
if(SYCL_CTS_ENABLE_HALF_TESTS)
    add_definitions(-DSYCL_CTS_TEST_HALF)
endif()
# ------------------

# ------------------
# OpenCL interoperability
option(SYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS
       "Enable OpenCL interoperability tests." ON)
if(SYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS)
    add_definitions(-DSYCL_CTS_TEST_OPENCL_INTEROP)
endif()
# ------------------

enable_testing()

add_subdirectory(util)
add_subdirectory(tests)
add_subdirectory(oclmath)
