set(TESTS_DIR ${CMAKE_CURRENT_LIST_DIR})
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# create bin and intermediate folders
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/intermediate")

# create separate targets for each test project
file(GLOB subdirectories RELATIVE ${TESTS_DIR} ${TESTS_DIR}/*)
set(test_projects_dir_list "")
foreach(dir ${subdirectories})
    if(IS_DIRECTORY ${TESTS_DIR}/${dir})
        if( NOT ${dir} STREQUAL "common")
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
