set( glew_version "1.7.0" )
set( glew_dir     "glew-${glew_version}" )

set( glew_lib_name GLEW )
get_filename_component( glew_inc_dir "${glew_dir}/include" ABSOLUTE )
set( glew_bld_flags -DGLEW_STATIC )
add_definitions(${glew_bld_flags})

include_directories( "${glew_inc_dir}" )
if( OPENGL_INCLUDE_DIR )
    include_directories( "${OPENGL_INCLUDE_DIR}" )
endif()

add_library(
    ${glew_lib_name}
    STATIC
    ${glew_dir}/src/glew.c
    ${glew_dir}/src/glewinfo.c
    ${glew_dir}/src/visualinfo.c
)

set_property( GLOBAL PROPERTY glew_lib_name "${glew_lib_name}" )
set_property( GLOBAL PROPERTY glew_inc_dir  "${glew_inc_dir}"  )
set_property( GLOBAL PROPERTY glew_bld_flags ${glew_bld_flags} )

