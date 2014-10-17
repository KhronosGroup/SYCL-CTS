# SYCLONE variables
set(OPENCL_LIBRARY "OPENCL_LIBRARY-NOTFOUND" CACHE FILEPATH "Path to the OpenCL library.")
set(DEVICE_COMPILER "DEVICE_COMPILER-NOTFOUND" CACHE FILEPATH "Path to the device compiler.")
set(SYCLONE_INCLUDE_PATH "SYCLONE_INCLUDE_PATH-NOTFOUND" CACHE PATH "Path to the SYCLONE runtime headers.")
set(SYCLONE_SYCL_LIBRARY "SYCLONE_SYCL_LIBRARY-NOTFOUND" CACHE FILEPATH "Path to the SYCLONE SYCL library.")
set(SYCLONE_IMAGE_LIBRARY_INCLUDE_PATH "SYCLONE_IMAGE_LIBRARY_INCLUDE_PATH-NOTFOUND" CACHE PATH "Path to the SYCLONE image library headers.")
set(SYCLONE_IMAGE_KERNEL_LIBRARY "SYCLONE_IMAGE_KERNEL_LIBRARY-NOTFOUND" CACHE FILEPATH "Path to the SYCLONE image kernel library.")
set(SYCLONE_IMAGE_HOST_LIBRARY "SYCLONE_IMAGE_HOST_LIBRARY-NOTFOUND" CACHE FILEPATH "Path to the SYCLONE image host library.")
# ------------------


# check if mandatory variables have been defined
message(STATUS "OPENCL_LIBRARY='${OPENCL_LIBRARY}'")
if(OPENCL_LIBRARY STREQUAL "OPENCL_LIBRARY-NOTFOUND")
    MESSAGE(FATAL_ERROR "\n\nPlease provide the path to the OpenCL library.\n\n")
endif()
message(STATUS "DEVICE_COMPILER='${DEVICE_COMPILER}'")
if(DEVICE_COMPILER STREQUAL "DEVICE_COMPILER-NOTFOUND")
    MESSAGE(FATAL_ERROR "\n\nPlease provide the path to the device compiler.\n\n")
endif()
message(STATUS "SYCLONE_INCLUDE_PATH='${SYCLONE_INCLUDE_PATH}'")
if(SYCLONE_INCLUDE_PATH STREQUAL "SYCLONE_INCLUDE_PATH-NOTFOUND")
    MESSAGE(FATAL_ERROR "\n\nPlease provide the path to the SYCLONE runtime headers.\n\n")
endif()
message(STATUS "SYCLONE_SYCL_LIBRARY='${SYCLONE_SYCL_LIBRARY}'")
if(SYCLONE_SYCL_LIBRARY STREQUAL "SYCLONE_SYCL_LIBRARY-NOTFOUND")
    MESSAGE(FATAL_ERROR "\n\nPlease provide the path to the SYCLONE SYCL library.\n\n")
endif()
message(STATUS "SYCLONE_IMAGE_LIBRARY_INCLUDE_PATH='${SYCLONE_IMAGE_LIBRARY_INCLUDE_PATH}'")
if(SYCLONE_IMAGE_LIBRARY_INCLUDE_PATH STREQUAL "SYCLONE_IMAGE_LIBRARY_INCLUDE_PATH-NOTFOUND")
MESSAGE(FATAL_ERROR "\n\nPlease provide the path to the SYCLONE image library headers.\n\n")
endif()
message(STATUS "SYCLONE_IMAGE_KERNEL_LIBRARY='${SYCLONE_IMAGE_KERNEL_LIBRARY}'")
if(SYCLONE_IMAGE_KERNEL_LIBRARY STREQUAL "SYCLONE_IMAGE_KERNEL_LIBRARY-NOTFOUND")
MESSAGE(FATAL_ERROR "\n\nPlease provide the path to the SYCLONE image kernel library.\n\n")
endif()
message(STATUS "SYCLONE_IMAGE_HOST_LIBRARY='${SYCLONE_IMAGE_HOST_LIBRARY}'")
if(SYCLONE_IMAGE_HOST_LIBRARY STREQUAL "SYCLONE_IMAGE_HOST_LIBRARY-NOTFOUND")
MESSAGE(FATAL_ERROR "\n\nPlease provide the path to the SYCLONE image host library.\n\n")
endif()
# ------------------


# ------------------
set(SYCL_LIB_NAME "SYCL")
set(IMAGE_KERNEL_LIB_NAME "image_library_kernel")
set(IMAGE_HOST_LIB_NAME "image_library_host")
set(OPENCL_LIB_NAME "OpenCL")

if(WIN32)
    set(RUNTIME_COMPILER_FLAGS " ")
    set(DEVICE_COMPILER_FLAGS "-D_SIZE_T_DEFINED")
    set(HOST_COMPILER_FLAGS " ")
elseif(APPLE)
    set(RUNTIME_COMPILER_FLAGS "-std=c++0x -stdlib=libc++ -D\"_XOPEN_SOURCE\"") 
    set(DEVICE_COMPILER_FLAGS -std=c++0x -stdlib=libc++ -mno-sse -D_ANSI_SOURCE)
    set(HOST_COMPILER_FLAGS "-std=c++0x -stdlib=libc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
else(WIN32)
    set(RUNTIME_COMPILER_FLAGS "-std=c++0x")    
    set(DEVICE_COMPILER_FLAGS "-std=c++0x")
    set(HOST_COMPILER_FLAGS "-std=c++0x -pthread")
endif(WIN32)
# ------------------


# ------------------
include_directories(${SYCLONE_INCLUDE_PATH})
add_library(${SYCL_LIB_NAME} STATIC IMPORTED )
set_target_properties(${SYCL_LIB_NAME} PROPERTIES IMPORTED_LOCATION ${SYCLONE_SYCL_LIBRARY})
add_library(${OPENCL_LIB_NAME} STATIC IMPORTED )
set_target_properties(${OPENCL_LIB_NAME} PROPERTIES IMPORTED_LOCATION ${OPENCL_LIBRARY})
include_directories(${SYCLONE_IMAGE_LIBRARY_INCLUDE_PATH})
add_library(${IMAGE_KERNEL_LIB_NAME} STATIC IMPORTED )
set_target_properties(${IMAGE_KERNEL_LIB_NAME} PROPERTIES IMPORTED_LOCATION ${SYCLONE_IMAGE_KERNEL_LIBRARY})
add_library(${IMAGE_HOST_LIB_NAME} STATIC IMPORTED )
set_target_properties(${IMAGE_HOST_LIB_NAME} PROPERTIES IMPORTED_LOCATION ${SYCLONE_IMAGE_HOST_LIBRARY})
# ------------------


# ------------------
include(cmake/syclone/build_sycl_implementation.cmake)
# ------------------

