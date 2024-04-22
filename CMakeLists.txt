cmake_minimum_required(VERSION 3.15)
project(sycl_cts LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON) # Required for hex floats in C++11 mode on gcc 6+
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

find_package(Threads REQUIRED)
find_package(PythonInterp 3 REQUIRED)

macro(add_submodule_directory RELPATH)
    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/${RELPATH}/CMakeLists.txt")
        message(FATAL_ERROR "The git submodule ${RELPATH} is missing.\nTry running `git submodule update --init`.")
    endif()
    add_subdirectory("${RELPATH}")
endmacro()

add_submodule_directory(vendor/Catch2)

# set host compiler flags
if(WIN32 AND MSVC)
    add_compile_options(/bigobj)
endif()

if(${CMAKE_CXX_COMPILER_ID} MATCHES "GNU" OR
   ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    add_compile_options(-Wall -Wno-unused-variable)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(AddCTSOption)

add_cts_option(SYCL_CTS_ENABLE_FULL_CONFORMANCE
    "Enable full conformance with extensive tests" OFF
    WARN_IF_OFF "Full conformance mode (SYCL_CTS_ENABLE_FULL_CONFORMANCE) should be used for conformance submission")

# TODO: Should SYCL_CTS_ENABLE_FULL_CONFORMANCE=ON imply this?
add_cts_option(SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    "Enable tests for deprecated SYCL features" ON
    WARN_IF_OFF "Tests for deprecated SYCL features should be enabled for conformance submission")

add_cts_option(SYCL_CTS_ENABLE_EXT_ONEAPI_TESTS
    "Enable all extension oneAPI tests" OFF)

add_cts_option(SYCL_CTS_ENABLE_EXT_ONEAPI_PROPERTIES_TESTS
    "Enable extension oneAPI compile-time property list tests" OFF
    FORCE_ON ${SYCL_CTS_ENABLE_EXT_ONEAPI_TESTS})

add_cts_option(SYCL_CTS_ENABLE_EXT_ONEAPI_SUB_GROUP_MASK_TESTS
    "Enable extension oneAPI sub_group_mask tests" OFF
    FORCE_ON ${SYCL_CTS_ENABLE_EXT_ONEAPI_TESTS})

add_cts_option(SYCL_CTS_ENABLE_EXT_ONEAPI_DEVICE_GLOBAL_TESTS
    "Enable extension oneAPI device_global tests" OFF
    FORCE_ON ${SYCL_CTS_ENABLE_EXT_ONEAPI_TESTS})

add_cts_option(SYCL_CTS_ENABLE_EXT_ONEAPI_MEMCPY2D_TESTS
    "Enable extension oneAPI memcpy2d tests" OFF
    FORCE_ON ${SYCL_CTS_ENABLE_EXT_ONEAPI_TESTS})

add_cts_option(SYCL_CTS_ENABLE_EXT_ONEAPI_ENQUEUE_BARRIER_TESTS
    "Enable extension oneAPI enqueue_barrier tests" OFF
    FORCE_ON ${SYCL_CTS_ENABLE_EXT_ONEAPI_TESTS})

add_cts_option(SYCL_CTS_ENABLE_EXT_ONEAPI_WEAK_OBJECT_TESTS
    "Enable extension oneAPI weak_object tests" OFF
    FORCE_ON ${SYCL_CTS_ENABLE_EXT_ONEAPI_TESTS})

add_cts_option(SYCL_CTS_ENABLE_EXT_ONEAPI_QUEUE_EMPTY
    "Enable extension oneAPI queue_empty tests" OFF
    FORCE_ON ${SYCL_CTS_ENABLE_EXT_ONEAPI_TESTS})

add_cts_option(SYCL_CTS_ENABLE_EXT_ONEAPI_DISCARD_QUEUE_EVENTS_TESTS
    "Enable extension oneAPI discard_queue_events tests" OFF
    FORCE_ON ${SYCL_CTS_ENABLE_EXT_ONEAPI_TESTS})

add_cts_option(SYCL_CTS_ENABLE_EXT_ONEAPI_BFLOAT16_TESTS
    "Enable extension oneAPI bfloat16 tests" OFF
    FORCE_ON ${SYCL_CTS_ENABLE_EXT_ONEAPI_TESTS})

add_cts_option(SYCL_CTS_ENABLE_EXT_ONEAPI_AUTO_LOCAL_RANGE_TESTS
    "Enable extension oneAPI auto_local_range tests" OFF
    FORCE_ON ${SYCL_CTS_ENABLE_EXT_ONEAPI_TESTS})

add_cts_option(SYCL_CTS_ENABLE_EXT_ONEAPI_KERNEL_COMPILER_SPIRV_TESTS
    "Enable extension oneAPI kernel_compiler_spirv tests" OFF
    FORCE_ON ${SYCL_CTS_ENABLE_EXT_ONEAPI_TESTS})

add_cts_option(SYCL_CTS_ENABLE_EXT_ONEAPI_COMPOSITE_DEVICE_TESTS
    "Enable extension oneAPI composite_device tests" OFF
    FORCE_ON ${SYCL_CTS_ENABLE_EXT_ONEAPI_TESTS})

# TODO: Deprecated - remove
add_cts_option(SYCL_CTS_ENABLE_VERBOSE_LOG
    "Enable debug-level logs (deprecated)" OFF)

add_cts_option(SYCL_CTS_ENABLE_DOUBLE_TESTS
    "Enable tests that require double precision floating point capabilities" ON)

add_cts_option(SYCL_CTS_ENABLE_HALF_TESTS
    "Enable tests that require half precision floating point capabilities" ON)

add_cts_option(SYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS
    "Enable OpenCL interoperability tests" ON)

add_cts_option(SYCL_CTS_ENABLE_CUDA_INTEROP_TESTS
    "Enable CUDA interoperability tests" OFF)

add_cts_option(SYCL_CTS_ENABLE_FEATURE_SET_FULL
    "Enable full feature set, which includes all features specified in the core SYCL specification" ON)

include(AddOpenCLProxy)
include(AddSYCLExecutable)

# ------------------
# Enable CUDA language and add library, for CUDA interop tests

if(SYCL_CTS_ENABLE_CUDA_INTEROP_TESTS)
    find_package(CUDA REQUIRED)

    set(CMAKE_CUDA_COMPILER ${CMAKE_CXX_COMPILER})
    enable_language(CUDA)

    add_library(CUDA::CUDA INTERFACE IMPORTED GLOBAL)
    set_target_properties(
      CUDA::CUDA PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES ${CUDA_TOOLKIT_INCLUDE}
    )
endif()
# ------------------

# ------------------
# Device used for running with CTest (e.g. during conformance report generation)
set(SYCL_CTS_CTEST_DEVICE "" CACHE STRING "Device used when running with CTest")
# ------------------

# ------------------
# Measure build times
option(SYCL_CTS_MEASURE_BUILD_TIMES "Measure build time for each translation unit and write it to 'build_times.log'" OFF)
if(SYCL_CTS_MEASURE_BUILD_TIMES)
    if(CMAKE_GENERATOR MATCHES "Makefiles|Ninja")
        # Wrap compiler calls in utility script to measure build times.
        # Note that SYCL implementations that require custom build steps, e.g. for dedicated
        # device compiler passes may require special handling.
        # In case the user already specified a compiler launcher, make sure ours comes first.
        list(PREPEND CMAKE_CXX_COMPILER_LAUNCHER "${CMAKE_SOURCE_DIR}/tools/measure_build_time.py")
    else()
        # Only Makefiles and Ninja support CMake compiler launchers
        message(FATAL_ERROR "Build time measurements are only supported for the 'Unix Makefiles' and 'Ninja' generators.")
    endif()
endif()
# ------------------

enable_testing()

add_subdirectory(util)
add_subdirectory(tests)
add_subdirectory(oclmath)

# This should be the last line
print_cts_config_summary()
