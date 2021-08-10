set(tests_dir ${CMAKE_CURRENT_SOURCE_DIR})
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(SYCL_CTS_TEST_FILTER "" CACHE FILEPATH "Path to the SYCL CTS test filter for building.")

# Get the list of test categories to build
# All test categories are built unless a test filter was given
if(SYCL_CTS_TEST_FILTER)
    message(WARNING "A filter file has been given to cts, only tests matching the ones on the file will be build.")
    message(STATUS "SYCL_CTS_TEST_FILTER='${SYCL_CTS_TEST_FILTER}'")
    file(STRINGS "${SYCL_CTS_TEST_FILTER}" test_projects_dir_list)
else()
    # create separate targets for each test project
    set(test_projects_dir_list "")
    file(GLOB subdirectories RELATIVE ${tests_dir} ${tests_dir}/*)
    foreach(dir ${subdirectories})
        if(${dir} MATCHES "common")
            # The common folder is always included
            continue()
        endif()
        if(EXISTS "${tests_dir}/${dir}/CMakeLists.txt")
            list(APPEND test_projects_dir_list ${dir})
        endif()
    endforeach()
endif()
list(REMOVE_DUPLICATES test_projects_dir_list)

# The common folder is always included
add_subdirectory("common")

function(get_std_type OUT_LIST)
  set(STD_TYPE_LIST "")
  list(APPEND STD_TYPE_LIST char
    "signed char"
    "unsigned char")

  list(APPEND STD_TYPE_LIST short
    "unsigned short")

  list(APPEND STD_TYPE_LIST int
    "unsigned int")

  list(APPEND STD_TYPE_LIST long
    "unsigned long"
    "long long"
    "unsigned long long")

  list(APPEND STD_TYPE_LIST float
    double
    sycl::half)

  set(${OUT_LIST} ${${OUT_LIST}} ${STD_TYPE_LIST} PARENT_SCOPE)
endfunction()

function(get_no_vec_alias_type OUT_LIST)
  set(NO_VEC_ALIAS_LIST "")
  list(APPEND NO_VEC_ALIAS_LIST sycl::byte)

  set(${OUT_LIST} ${${OUT_LIST}} ${NO_VEC_ALIAS_LIST} PARENT_SCOPE)
endfunction()

function(get_fixed_width_type OUT_LIST)
  set(FIXED_WIDTH_LIST "")
  list(APPEND FIXED_WIDTH_LIST std::int8_t
    std::int16_t
    std::int32_t
    std::int64_t
    std::uint8_t
    std::uint16_t
    std::uint32_t
    std::uint64_t)

  set(${OUT_LIST} ${${OUT_LIST}} ${FIXED_WIDTH_LIST} PARENT_SCOPE)
endfunction()

macro(half_double_filter list)
  if(NOT SYCL_CTS_ENABLE_DOUBLE_TESTS)
    list(REMOVE_ITEM ${list} double)
  endif()

  if(NOT SYCL_CTS_ENABLE_HALF_TESTS)
    list(REMOVE_ITEM ${list} sycl::half)
  endif()

endmacro()

# Create a target to trigger the generation of CTS test
add_custom_target(generate_test_sources)

# Test generation routine
function(generate_cts_test)
  cmake_parse_arguments(GEN_TEST "" "TESTS;GENERATOR;OUTPUT;INPUT" "EXTRA_ARGS" ${ARGN})
  message(STATUS "Setup test generation rules for: " ${GEN_TEST_OUTPUT})

  set(GEN_TEST_FILE_NAME ${GEN_TEST_OUTPUT})
  set(GEN_TEST_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${GEN_TEST_OUTPUT})
  set(GEN_TEST_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/${GEN_TEST_INPUT})
  # Add the file to the out test list
  set(${GEN_TEST_TESTS} ${${GEN_TEST_TESTS}} ${GEN_TEST_OUTPUT} PARENT_SCOPE)

  get_filename_component(test_dir ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  get_filename_component(test_name ${GEN_TEST_OUTPUT} NAME_WE)

  add_custom_command(OUTPUT ${GEN_TEST_OUTPUT}
    COMMAND
      ${PYTHON_EXECUTABLE}
      ${GEN_TEST_GENERATOR}
      ${GEN_TEST_INPUT}
      -o ${GEN_TEST_OUTPUT}
      ${GEN_TEST_EXTRA_ARGS}
    DEPENDS
      ${GEN_TEST_GENERATOR}
      ${GEN_TEST_INPUT}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Generating test ${GEN_TEST_OUTPUT}..."
    )

  add_custom_target(${GEN_TEST_FILE_NAME}_gen DEPENDS ${GEN_TEST_OUTPUT})
  add_dependencies(generate_test_sources ${GEN_TEST_FILE_NAME}_gen)
endfunction()

# create a target to group all tests together into one test executable
add_executable(test_all)

# create test executable targets for each test project using the build_sycl function
function(add_cts_test)
  get_filename_component(test_dir ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  set(test_exe_name test_${test_dir})
  set(test_cases_list ${ARGN})

  if(NOT SYCL_CTS_ENABLE_HALF_TESTS)
    list(FILTER test_cases_list EXCLUDE REGEX .*_fp16\\.cpp$)
  endif()
  if(NOT SYCL_CTS_ENABLE_DOUBLE_TESTS)
    list(FILTER test_cases_list EXCLUDE REGEX .*_fp64\\.cpp$)
  endif()

  message(STATUS "Adding test: " ${test_exe_name})

  add_sycl_executable(NAME           ${test_exe_name}
                      OBJECT_LIBRARY ${test_exe_name}_objects
                      TESTS          ${test_cases_list})

  target_include_directories(${test_exe_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  set(info_dump_dir "${CMAKE_BINARY_DIR}/Testing")
  add_test(NAME ${test_exe_name}_host
           COMMAND ${test_exe_name}
                   --platform ${host_platform_name}
                   --device ${host_device_name}
                   --info-dump "${info_dump_dir}/${test_exe_name}_host.info")
  add_test(NAME ${test_exe_name}_opencl
           COMMAND ${test_exe_name}
                   --platform ${opencl_platform_name}
                   --device ${opencl_device_name}
                   --info-dump "${info_dump_dir}/${test_exe_name}_opencl.info")

  target_link_libraries(${test_exe_name} PRIVATE CTS::util CTS::main_function oclmath)

  set_property(TARGET ${test_exe_name}
               PROPERTY FOLDER "Tests/${test_exe_name}")
  set_property(TARGET ${test_exe_name}_objects
               PROPERTY FOLDER "Tests/${test_exe_name}")

  target_sources(test_all PRIVATE $<TARGET_OBJECTS:${test_exe_name}_objects>)
endfunction()

foreach(dir ${test_projects_dir_list})
  add_subdirectory(${dir})
endforeach()

target_link_libraries(test_all PRIVATE CTS::util CTS::main_function oclmath)
