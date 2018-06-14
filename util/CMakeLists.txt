file(GLOB_RECURSE source_files *.cpp)
file(GLOB util_headers *.h)

add_library(util ${source_files} ${util_headers})
add_library(CTS::util ALIAS util)
target_link_libraries(util PUBLIC OpenCL::OpenCL SYCL::SYCL)
