file(GLOB_RECURSE source_files *.cpp)
file(GLOB util_headers *.h)

if(NOT SYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS)
    list(FILTER source_files EXCLUDE REGEX "test_base_opencl\.cpp")
endif()

if(NOT SYCL_CTS_ENABLE_CUDA_INTEROP_TESTS)
    list(FILTER source_files EXCLUDE REGEX "test_base_cuda\.cpp")
endif()

add_library(util ${source_files} ${util_headers})
add_library(CTS::util ALIAS util)

target_link_libraries(util PUBLIC OpenCL::OpenCL SYCL::SYCL Catch2::Catch2)
if(SYCL_CTS_ENABLE_CUDA_INTEROP_TESTS)
    target_link_libraries(util PUBLIC cuda)
endif()
