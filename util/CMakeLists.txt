file(GLOB_RECURSE source_files *.cpp)
file(GLOB util_headers *.h)

if(NOT SYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS)
    list(FILTER source_files EXCLUDE REGEX "test_base_opencl\.cpp")
endif()

add_library(util ${source_files} ${util_headers})
add_library(CTS::util ALIAS util)
target_compile_definitions(util PUBLIC ${SYCL_CTS_DETAIL_OPTION_COMPILE_DEFINITIONS})
target_link_libraries(util PUBLIC OpenCL::OpenCL SYCL::SYCL Catch2::Catch2)
