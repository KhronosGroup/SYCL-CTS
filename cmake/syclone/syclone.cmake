# SYCLONE variables
set(OPENCL_LIBRARY "OPENCL_LIBRARY-NOTFOUND" CACHE FILEPATH "Path to the OpenCL library.")
set(DEVICE_COMPILER "DEVICE_COMPILER-NOTFOUND" CACHE FILEPATH "Path to the device compiler.")
set(SYCLONE_INCLUDE_PATH "SYCLONE_INCLUDE_PATH-NOTFOUND" CACHE PATH "Path to the SYCLONE runtime headers.")
set(SYCLONE_SYCL_LIBRARY "SYCLONE_SYCL_LIBRARY-NOTFOUND" CACHE FILEPATH "Path to the SYCLONE SYCL library.")
if(WIN32)
	set(SYCLONE_ALEX_INCLUDE_PATH "SYCLONE_ALEX_INCLUDE_PATH-NOTFOUND" CACHE PATH "Path to the SYCLONE Alex image library headers.")
	set(SYCLONE_ALEX_LIBRARY "SYCLONE_ALEX_LIBRARY-NOTFOUND" CACHE FILEPATH "Path to the SYCLONE Alex image library.")
endif()
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
if(WIN32)
	message(STATUS "SYCLONE_ALEX_INCLUDE_PATH='${SYCLONE_ALEX_INCLUDE_PATH}'")
	if(SYCLONE_ALEX_INCLUDE_PATH STREQUAL "SYCLONE_ALEX_INCLUDE_PATH-NOTFOUND")
		MESSAGE(FATAL_ERROR "\n\nPlease provide the path to the SYCLONE Alex image library headers.\n\n")
	endif()
	message(STATUS "SYCLONE_ALEX_LIBRARY='${SYCLONE_ALEX_LIBRARY}'")
	if(SYCLONE_ALEX_LIBRARY STREQUAL "SYCLONE_ALEX_LIBRARY-NOTFOUND")
		MESSAGE(FATAL_ERROR "\n\nPlease provide the path to the SYCLONE Alex image library.\n\n")
	endif()
endif()
# ------------------


# ------------------
set(SYCL_LIB_NAME "SYCL")
set(ALEX_LIB_NAME "Alex")
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
if(WIN32)
	include_directories(${SYCLONE_ALEX_INCLUDE_PATH})
	add_library(${ALEX_LIB_NAME} STATIC IMPORTED )
	set_target_properties(${ALEX_LIB_NAME} PROPERTIES IMPORTED_LOCATION ${SYCLONE_ALEX_LIBRARY})
endif()
# ------------------


# ------------------
include(cmake/syclone/build_sycl_implementation.cmake)
# ------------------

