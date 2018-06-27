set(tests_dir ${CMAKE_CURRENT_SOURCE_DIR})
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

file(GLOB subdirectories RELATIVE ${tests_dir} ${tests_dir}/*)
foreach(dir ${subdirectories})
    if(EXISTS "${tests_dir}/${dir}/CMakeLists.txt")
        add_subdirectory(${dir})
    endif()
endforeach()

set(SYCL_CTS_TEST_FILTER "" CACHE FILEPATH "Path to the SYCL CTS test filter for building.")

if(SYCL_CTS_TEST_FILTER)
    message(WARNING "A filter file has been given to cts, only tests matching the ones on the file will be build.")
    message(STATUS "SYCL_CTS_TEST_FILTER='${SYCL_CTS_TEST_FILTER}'")
    file(STRINGS "${SYCL_CTS_TEST_FILTER}" test_projects_dir_list)
else()
    # create separate targets for each test project
    set(test_projects_dir_list "")
    foreach(dir ${subdirectories})
        if(EXISTS "${tests_dir}/${dir}/CMakeLists.txt")
            continue()
        endif()
        if(IS_DIRECTORY ${tests_dir}/${dir})
            list(APPEND test_projects_dir_list ${dir})
        endif()
    endforeach()
endif()

# generate tests if any for the test projects
message(STATUS "Generating tests...")
message(STATUS " ${test_projects_dir_list}")
foreach(test_project_dir ${test_projects_dir_list})
    # find test generation scripts
    file(GLOB test_generate_script_list ${tests_dir}/${test_project_dir}/*.py)

    foreach(test_generate_script ${test_generate_script_list})
        message(STATUS "Test Project: " ${test_project_dir})
        # call the python generation scripts
        execute_process(
            COMMAND ${PYTHON_EXECUTABLE} ${test_generate_script}
            WORKING_DIRECTORY "${tests_dir}/${test_project_dir}")

        # create regenerator targets for easily regenerating the tests
        # the python script generator should have the exact name as the template
        # the template and script together are added as a target
        get_filename_component(test_generate_script_name ${test_generate_script} NAME_WE)
        # remove the "generate_" from the python script to obtain the template name
        string(SUBSTRING ${test_generate_script_name} 9 -1 test_template_name)

        set(test_regenerator re${test_generate_script_name}_tests)
        set(test_generator_folder ${test_exe_name}_generators)
        set(test_exe_name test_${test_project_dir})

        add_custom_target(${test_regenerator}
            COMMAND ${PYTHON_EXECUTABLE} ${test_generate_script}
            SOURCES ${test_generate_script} ${tests_dir}/${test_project_dir}/${test_template_source}
            WORKING_DIRECTORY "${tests_dir}/${test_project_dir}")
        set_property(TARGET ${test_regenerator} PROPERTY FOLDER "Tests/${test_exe_name}/${test_generator_folder}")
    endforeach()
endforeach()
message(STATUS "Generating tests done")

# create test executable targets for each test project using the build_sycl function
foreach(test_project_dir ${test_projects_dir_list})
    file(GLOB test_cases_list ${tests_dir}/${test_project_dir}/*.cpp)
    set(test_exe_name test_${test_project_dir})

    add_sycl_executable(NAME           ${test_exe_name}
                        OBJECT_LIBRARY ${test_exe_name}_objects
                        TESTS          ${test_cases_list})

    add_test(NAME ${test_exe_name}_host COMMAND ${test_exe_name} --platform host --device host --info-dump "../Testing/${test_exe_name}_host.info")
    add_test(NAME ${test_exe_name}_opencl COMMAND ${test_exe_name} --platform intel --device opencl_cpu --info-dump "../Testing/${test_exe_name}_opencl.info")

    target_link_libraries(${test_exe_name} PRIVATE CTS::util CTS::main_function oclmath)

    set_property(TARGET ${test_exe_name}
                 PROPERTY FOLDER "Tests/${test_exe_name}")
    set_property(TARGET ${test_exe_name}_objects
                 PROPERTY FOLDER "Tests/${test_exe_name}")

    list(APPEND test_all_sources $<TARGET_OBJECTS:${test_exe_name}_objects>)
endforeach()

# create a target to group all tests together into one test executable
add_executable(test_all ${test_all_sources})
target_link_libraries(test_all PRIVATE CTS::util CTS::main_function oclmath)
