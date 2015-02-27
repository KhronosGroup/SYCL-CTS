set( glut_version "2.8.0" )
set( glut_dir     "freeglut-${glut_version}" )

set( glut_lib_name freeglut_static )
get_filename_component( glut_inc_dir "${glut_dir}/include" ABSOLUTE )
set( glut_bld_flags -DFREEGLUT_STATIC )
add_definitions(${glut_bld_flags})

if( UNIX )
    #   Better way would be running `./configure && make' but it is not CMake-friendly.
    #   So let us try to build freeglut with CMake.
    set( glut_unix_bld_flags
        -D_XOPEN_SOURCE=500
        -DHAVE_GETTIMEOFDAY
        -DHAVE_FCNTL_H
        -DHAVE_UNISTD_H
        -DTIME_WITH_SYS_TIME
    )
    add_definitions(${glut_unix_bld_flags})
endif()
include_directories( "${glut_inc_dir}" )
if( OPENGL_INCLUDE_DIR )
    include_directories( "${OPENGL_INCLUDE_DIR}" )
endif()

add_library(
    ${glut_lib_name}
    STATIC
    ${glut_dir}/src/freeglut_callbacks.c
    ${glut_dir}/src/freeglut_cursor.c
    ${glut_dir}/src/freeglut_display.c
    ${glut_dir}/src/freeglut_ext.c
    ${glut_dir}/src/freeglut_font.c
    ${glut_dir}/src/freeglut_font_data.c
    ${glut_dir}/src/freeglut_gamemode.c
    ${glut_dir}/src/freeglut_geometry.c
    ${glut_dir}/src/freeglut_glutfont_definitions.c
    ${glut_dir}/src/freeglut_init.c
    ${glut_dir}/src/freeglut_input_devices.c
    ${glut_dir}/src/freeglut_joystick.c
    ${glut_dir}/src/freeglut_main.c
    ${glut_dir}/src/freeglut_menu.c
    ${glut_dir}/src/freeglut_misc.c
    ${glut_dir}/src/freeglut_overlay.c
    ${glut_dir}/src/freeglut_spaceball.c
    ${glut_dir}/src/freeglut_state.c
    ${glut_dir}/src/freeglut_stroke_mono_roman.c
    ${glut_dir}/src/freeglut_stroke_roman.c
    ${glut_dir}/src/freeglut_structure.c
    ${glut_dir}/src/freeglut_teapot.c
    ${glut_dir}/src/freeglut_videoresize.c
    ${glut_dir}/src/freeglut_window.c
)
if( UNIX )
    target_link_libraries( ${glut_lib_name} X11 )
endif()

set_property( GLOBAL PROPERTY glut_lib_name  "${glut_lib_name}" )
set_property( GLOBAL PROPERTY glut_inc_dir   "${glut_inc_dir}"  )
set_property( GLOBAL PROPERTY glut_bld_flags ${glut_bld_flags}  )

