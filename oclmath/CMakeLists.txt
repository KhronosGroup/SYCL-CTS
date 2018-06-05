file(GLOB_RECURSE source_files *.cpp)
file(GLOB oclmath_headers *.h)

add_library(oclmath ${source_files} ${oclmath_headers})
target_link_libraries(oclmath PUBLIC OpenCL::OpenCL)
