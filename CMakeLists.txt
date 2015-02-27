# check CMake Version
if (WIN32)
# generating VS 2013 solution requires 2.8.11.2 minimum
  cmake_minimum_required(VERSION 2.8.11.2)
else (WIN32)
  cmake_minimum_required(VERSION 2.8)
endif (WIN32)

project(sycl_cts)

# set host compiler flags
if(WIN32)
# need c++ exception handling
#   set(HOST_COMPILER_FLAGS "${HOST_COMPILER_FLAGS} /EHsc")
    set(HOST_COMPILER_FLAGS "${HOST_COMPILER_FLAGS}")
elseif(APPLE)
# need c++11
    set(HOST_COMPILER_FLAGS "${HOST_COMPILER_FLAGS} -std=c++0x")
else(WIN32)
# need c++11
    set(HOST_COMPILER_FLAGS "${HOST_COMPILER_FLAGS} -std=c++0x -Wall -Wno-unused-variable -Wno-reorder -Wno-unneeded-internal-declaration")
endif(WIN32)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${HOST_COMPILER_FLAGS})
# ------------------

include(cmake/build_sycl.cmake)

# find the python interpreter
find_package(PythonInterp REQUIRED)

# OPENGL variables
set(OPENGL_ENABLE_INTEROP_TESTS ON CACHE BOOL "Enable OpenGL interop tests.")
if(OPENGL_ENABLE_INTEROP_TESTS)
    find_package(OpenGL REQUIRED)
    add_subdirectory (gl_util)
endif()
# ------------------

add_subdirectory (util)
add_subdirectory (tests)
add_subdirectory (oclmath)
