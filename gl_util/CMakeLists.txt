add_subdirectory(GL/glut)
add_subdirectory(GL/glew)
get_property(glut_inc GLOBAL PROPERTY glut_inc_dir)
get_property(glew_inc GLOBAL PROPERTY glew_inc_dir)
get_property(glut_flags GLOBAL PROPERTY glut_bld_flags)

file(GLOB gl_source_files *.cpp *.c)
file(GLOB common_headers ${CMAKE_CURRENT_LIST_DIR}/../tests/common/*.h)
file(GLOB gl_util_headers *.h)

include_directories(${glut_inc})
include_directories(${glew_inc})
add_definitions(${glut_flags})

add_library(gl_util ${gl_source_files} ${gl_util_headers} ${common_headers})
target_link_libraries(gl_util)
