cmake_minimum_required(VERSION 3.15)
project(sycl_cts LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON) # Required for hex floats in C++11 mode on gcc 6+
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(OpenCL REQUIRED)
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
# Extensive mode for full coverage
option(SYCL_CTS_ENABLE_FULL_CONFORMANCE
       "Enable full conformance mode with extensive tests" OFF)
if(SYCL_CTS_ENABLE_FULL_CONFORMANCE)
    message(STATUS "Full conformance mode: ON")
    add_definitions(-DSYCL_CTS_FULL_CONFORMANCE)
else()
    message(STATUS "Full conformance mode: OFF")
    message(WARNING
            "Full conformance mode should be used for conformance submission")
endif()
# ------------------

# ------------------
# Debug-level verbose logging
option(SYCL_CTS_ENABLE_VERBOSE_LOG "Enable debug-level logs" OFF)
if(SYCL_CTS_ENABLE_VERBOSE_LOG)
    add_definitions(-DSYCL_CTS_VERBOSE_LOG)
endif()
# ------------------

# ------------------
# Double and Half variables
option(SYCL_CTS_ENABLE_DOUBLE_TESTS "Enable Double tests." ON)
option(SYCL_CTS_ENABLE_HALF_TESTS "Enable Half tests." ON)
if(SYCL_CTS_ENABLE_DOUBLE_TESTS)
    add_host_and_device_compiler_definitions(-DSYCL_CTS_TEST_DOUBLE)
endif()
if(SYCL_CTS_ENABLE_HALF_TESTS)
    add_host_and_device_compiler_definitions(-DSYCL_CTS_TEST_HALF)
endif()
# ------------------

# ------------------
# OpenCL interoperability
option(SYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS
       "Enable OpenCL interoperability tests." ON)
if(SYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS)
    add_host_and_device_compiler_definitions(-DSYCL_CTS_TEST_OPENCL_INTEROP)
endif()
# ------------------

# ------------------
# Device used for running with CTest (e.g. during conformance report generation)
set(SYCL_CTS_CTEST_DEVICE "" CACHE STRING "Device used when running with CTest")
# ------------------

enable_testing()

add_subdirectory(util)
add_subdirectory(tests)
add_subdirectory(oclmath)
