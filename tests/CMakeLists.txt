set(TESTS_DIR ${CMAKE_CURRENT_LIST_DIR})
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")

# create bin and intermediate folders
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/intermediate")

# create separate targets for each test project
file(GLOB subdirectories RELATIVE ${TESTS_DIR} ${TESTS_DIR}/*)
set(test_projects_dir_list "")
foreach(dir ${subdirectories})
    if(IS_DIRECTORY ${TESTS_DIR}/${dir})
        # check whether to include opengl-interop based on whether the test is enabled
        if(${dir} MATCHES "opengl*")
            if(SYCL-CTS_ENABLE_OPENGL_INTEROP_TESTS)
                get_property(glut_flags GLOBAL PROPERTY glut_bld_flags)
                get_property(glew_flags GLOBAL PROPERTY glew_bld_flags)
                get_property(glut_inc GLOBAL PROPERTY glut_inc_dir)
                get_property(glew_inc GLOBAL PROPERTY glew_inc_dir)
                include_directories(${glut_inc})
                include_directories(${glew_inc})
                add_definitions(${glut_flags})
                add_definitions(${glew_flags})
                list(APPEND test_projects_dir_list ${dir})
            endif()
        # do not include common directory into the test list
        elseif(NOT ${dir} STREQUAL "common")
            if(SYCL-CTS_ENABLE_DOUBLE_TESTS)
                add_definitions(-DSYCL_CTS_TEST_DOUBLE)
            endif()
            if(SYCL-CTS_ENABLE_HALF_TESTS)
                add_definitions(-DSYCL_CTS_TEST_HALF)
            endif()
            list(APPEND test_projects_dir_list ${dir})
        endif()
    endif()
endforeach()

# generate tests if any for the test projects
message("\n\nGENERATING TESTS...")
foreach(test_project_dir ${test_projects_dir_list})
    # find test generation scripts
    file(GLOB test_generate_script_list ${TESTS_DIR}/${test_project_dir}/*.py)

    foreach(test_generate_script ${test_generate_script_list})
    message("\nTest Project : " ${test_project_dir})
        # call the python generation scripts
        execute_process(
            COMMAND ${PYTHON_EXECUTABLE} ${test_generate_script}
            WORKING_DIRECTORY "${TESTS_DIR}/${test_project_dir}")
        # print out generated test names
        file(GLOB test_cases_list ${TESTS_DIR}/${test_project_dir}/*.cpp)
        foreach(test_case ${test_cases_list})
            message(${test_case})
        endforeach()
    endforeach()
endforeach()
message("\nGENERATING TESTS DONE.\n\n")

set(all_test_cases_list "")
# create test executable targets for each test project using the build_sycl function
foreach(test_project_dir ${test_projects_dir_list}) 
    file(GLOB test_cases_list ${TESTS_DIR}/${test_project_dir}/*.cpp)
    list(APPEND all_test_cases_list ${test_cases_list})
    
    set(test_main ${TESTS_DIR}/common/main.cpp) 
    set(test_exe_name test_${test_project_dir})
    
    # invoke build_sycl()
    build_sycl("${test_exe_name}" "${test_main}" "${test_cases_list}" "${CMAKE_BINARY_DIR}/bin" "${CMAKE_BINARY_DIR}/bin/intermediate")
    
    # link the test executable with the test framework util library
    target_link_libraries(${test_exe_name} util)
    target_link_libraries(${test_exe_name} oclmath)
    
    # special flags and includes for opengl interop tests
    if(${test_project_dir} MATCHES "opengl*")
        if(SYCL-CTS_ENABLE_OPENGL_INTEROP_TESTS)
            # get global properties for gl interop tests
            get_property(glut_lib GLOBAL PROPERTY glut_lib_name)
            get_property(glew_lib GLOBAL PROPERTY glew_lib_name)
            target_link_libraries(${test_exe_name} ${glew_lib})
            # on gcc, we need to use the whole-archive option since we have both static and shared libraries
            if(UNIX)
                set(glut_lib_whole_archive_link -Wl,--whole-archive ${glut_lib} -Wl,--no-whole-archive)
                target_link_libraries(${test_exe_name} ${glut_lib_whole_archive_link})
            else()
                target_link_libraries(${test_exe_name} ${glut_lib})
            endif()
            target_link_libraries(${test_exe_name} gl_util)
            target_link_libraries(${test_exe_name} ${OPENGL_LIBRARIES})
        endif()
    endif()

    # create regenerator targets for easily regenerating the tests
    # the python script generator should have the exact name as the template
    # the template and script together are added as a target
    file(GLOB test_generate_script_list ${TESTS_DIR}/${test_project_dir}/*.py)
    foreach(test_generate_script ${test_generate_script_list})
        get_filename_component(test_generate_script_name ${test_generate_script} NAME_WE)
        # remove the "generate_" from the python script to obtain the template name
        string(SUBSTRING ${test_generate_script_name} 9 -1 test_template_name)

        set(test_template_source ${test_template_name}.template)
        set(test_regenerator re${test_generate_script_name}_tests)
        set(test_generator_folder ${test_exe_name}_generators)

        add_custom_target(${test_regenerator}
            COMMAND ${PYTHON_EXECUTABLE} ${test_generate_script}
            SOURCES ${test_generate_script} ${TESTS_DIR}/${test_project_dir}/${test_template_source}
            WORKING_DIRECTORY "${TESTS_DIR}/${test_project_dir}")
        set_property(TARGET ${test_regenerator} PROPERTY FOLDER "Tests/${test_exe_name}/${test_generator_folder}")
    endforeach()

endforeach()
# ------------------


# create a target to group all tests together into one test executable
set(test_main ${TESTS_DIR}/common/main.cpp) 
set(test_exe_name test_all) 
# invoke build_sycl()
build_sycl("${test_exe_name}" "${test_main}" "${all_test_cases_list}" "${CMAKE_BINARY_DIR}/bin" "${CMAKE_BINARY_DIR}/bin/intermediate")
target_link_libraries(${test_exe_name} util)
target_link_libraries(${test_exe_name} oclmath)
# special flags and includes for opengl interop tests
if(SYCL-CTS_ENABLE_OPENGL_INTEROP_TESTS)
    # get global properties for gl interop tests
    get_property(glut_lib GLOBAL PROPERTY glut_lib_name)
    get_property(glew_lib GLOBAL PROPERTY glew_lib_name)
    target_link_libraries(${test_exe_name} ${glew_lib})
    # on gcc, we need to use the whole-archive option since we have both static and shared libraries
    if(UNIX)
        set(glut_lib_whole_archive_link -Wl,--whole-archive ${glut_lib} -Wl,--no-whole-archive)
        target_link_libraries(${test_exe_name} ${glut_lib_whole_archive_link})
    else()
        target_link_libraries(${test_exe_name} ${glut_lib})
    endif()
    target_link_libraries(${test_exe_name} gl_util)
    target_link_libraries(${test_exe_name} ${OPENGL_LIBRARIES})
endif()

