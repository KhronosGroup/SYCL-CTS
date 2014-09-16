set(TESTS_DIR ${CMAKE_CURRENT_LIST_DIR})
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

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

set(all_test_cases_list "")
# create test executable targets for each test project using the build_sycl function
foreach(test_project_dir ${test_projects_dir_list})	
	file(GLOB test_cases_list ${TESTS_DIR}/${test_project_dir}/*.cpp)
	list(APPEND all_test_cases_list ${test_cases_list})
	
	set(test_main ${TESTS_DIR}/common/main.cpp)	
	set(test_exe_name test_${test_project_dir})	
	
	# invoke build_sycl()
	build_sycl("${test_exe_name}" "${test_main}" "${test_cases_list}" "${TESTS_DIR}/${test_project_dir}")
	
	#link the test executable with the test framework util library
	target_link_libraries(${test_exe_name} util)	
endforeach()
# ------------------

# create a target to group all tests together into one test executable
set(test_main ${TESTS_DIR}/common/main.cpp)	
set(test_exe_name test_all)	
# invoke build_sycl()
build_sycl("${test_exe_name}" "${test_main}" "${all_test_cases_list}" "${TESTS_DIR}/common")
target_link_libraries(${test_exe_name} util)
