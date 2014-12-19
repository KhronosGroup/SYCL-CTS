file(GLOB_RECURSE source_files *.cpp)
file(GLOB common_headers ${CMAKE_CURRENT_LIST_DIR}/../tests/common/*.h)
file(GLOB util_headers *.h)

add_library(util ${source_files} ${util_headers} ${common_headers})