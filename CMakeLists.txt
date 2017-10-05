# check CMake Version
cmake_minimum_required(VERSION 3.2)

project(sycl_cts)

# Find OpenCL library and headers
find_package(OpenCL REQUIRED)

# set host compiler flags
if(WIN32)
# need c++ exception handling
#   set(HOST_COMPILER_FLAGS "${HOST_COMPILER_FLAGS} /EHsc")
    set(HOST_COMPILER_FLAGS "${HOST_COMPILER_FLAGS} /bigobj")
elseif(APPLE)
# need c++11
    set(HOST_COMPILER_FLAGS "${HOST_COMPILER_FLAGS} -std=c++0x")
else(WIN32)
# need c++11
    set(HOST_COMPILER_FLAGS "${HOST_COMPILER_FLAGS} -std=c++11 -Wall -Wno-unused-variable")
endif(WIN32)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${HOST_COMPILER_FLAGS})
# ------------------

include_directories(${OpenCL_INCLUDE_DIR})

include(cmake/build_sycl.cmake)

# find the python interpreter
find_package(PythonInterp REQUIRED)
# ------------------

# Double and Half variables
#set(SYCL_CTS_ENABLE_DOUBLE_TESTS ON CACHE BOOL "Enable Double tests.")
#set(SYCL-CTS_ENABLE_HALF_TESTS ON CACHE BOOL "Enable Half tests.")
# ------------------

enable_testing()

add_subdirectory (util)
add_subdirectory (tests)
add_subdirectory (oclmath)
