set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(SYCL_CTS_TEST_FILTER)
  message(FATAL_ERROR "SYCL_CTS_TEST_FILTER is no longer supported. Use SYCL_CTS_EXCLUDE_TEST_CATEGORIES instead.")
endif()
set(SYCL_CTS_EXCLUDE_TEST_CATEGORIES "" CACHE FILEPATH "Optional filter specifying test categories to be excluded from build.")

set(exclude_categories "")
if(SYCL_CTS_EXCLUDE_TEST_CATEGORIES)
  if(NOT EXISTS "${SYCL_CTS_EXCLUDE_TEST_CATEGORIES}" OR IS_DIRECTORY "${SYCL_CTS_EXCLUDE_TEST_CATEGORIES}")
    message(FATAL_ERROR "Invalid filter file '${SYCL_CTS_EXCLUDE_TEST_CATEGORIES}'.")
  endif()
  message(STATUS "Using test category filter '${SYCL_CTS_EXCLUDE_TEST_CATEGORIES}'.")
  file(STRINGS "${SYCL_CTS_EXCLUDE_TEST_CATEGORIES}" exclude_categories)
endif()

add_subdirectory("common")

function(get_std_type OUT_LIST)
  set(STD_TYPE_LIST "")

  list(APPEND STD_TYPE_LIST
    "bool"
    "char"
    "int"
    "float"
    "double"
    "sycl::half"
  )

  if(SYCL_CTS_ENABLE_FULL_CONFORMANCE)
    list(APPEND STD_TYPE_LIST
      "signed char"
      "unsigned char"
      "short"
      "unsigned short"
      "unsigned int"
      "long"
      "unsigned long"
      "long long"
      "unsigned long long"
    )
  endif()

  set(${OUT_LIST} ${${OUT_LIST}} ${STD_TYPE_LIST} PARENT_SCOPE)
endfunction()

function(get_marray_elem_type OUT_LIST)
  set(TYPE_LIST "")
  get_std_type(TYPE_LIST)

  list(APPEND TYPE_LIST "custom_int")

  if(SYCL_CTS_ENABLE_FULL_CONFORMANCE)
    list(APPEND TYPE_LIST
         "std::uint8_t"
         "std::int16_t"
         "std::uint16_t"
         "std::uint32_t"
         "std::int64_t"
         "std::uint64_t"
    )
  endif()

  set(${OUT_LIST} ${${OUT_LIST}} ${TYPE_LIST} PARENT_SCOPE)
endfunction()

function(get_no_vec_alias_type OUT_LIST)
  set(NO_VEC_ALIAS_LIST "")
  list(APPEND NO_VEC_ALIAS_LIST sycl::byte)
  list(APPEND NO_VEC_ALIAS_LIST std::byte)

  set(${OUT_LIST} ${${OUT_LIST}} ${NO_VEC_ALIAS_LIST} PARENT_SCOPE)
endfunction()

function(get_fixed_width_type OUT_LIST)
  set(FIXED_WIDTH_LIST "")

  list(APPEND FIXED_WIDTH_LIST
    std::int8_t
    std::int32_t
  )

  if(SYCL_CTS_ENABLE_FULL_CONFORMANCE)
    list(APPEND FIXED_WIDTH_LIST
      std::uint8_t
      std::int16_t
      std::uint16_t
      std::uint32_t
      std::int64_t
      std::uint64_t
    )
  endif()

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
  cmake_parse_arguments(
    GEN_TEST
    ""
    "TESTS;GENERATOR;OUTPUT;INPUT"
    "EXTRA_ARGS;DEPENDS"
    ${ARGN}
  )
  get_filename_component(test_dir ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  if(${test_dir} IN_LIST exclude_categories)
    return()
  endif()

  message(STATUS "Setup test generation rules for: " ${GEN_TEST_OUTPUT})

  set(GEN_TEST_FILE_NAME ${GEN_TEST_OUTPUT})
  set(GEN_TEST_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${GEN_TEST_OUTPUT})
  set(GEN_TEST_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/${GEN_TEST_INPUT})

  set(extra_deps "")
  foreach(filename ${GEN_TEST_DEPENDS})
    list(APPEND extra_deps ${CMAKE_CURRENT_SOURCE_DIR}/${filename})
  endforeach()

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
      ${extra_deps}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Generating test ${GEN_TEST_OUTPUT}..."
    )

  add_custom_target(${GEN_TEST_FILE_NAME}_gen DEPENDS ${GEN_TEST_OUTPUT})
  add_dependencies(generate_test_sources ${GEN_TEST_FILE_NAME}_gen)
endfunction()

# Test generation routine
function(generate_multifile_cts_test)
  cmake_parse_arguments(
    GEN_TEST
    ""
    "TESTS;GENERATOR;FILE_PREFIX;EXT;INPUT"
    "EXTRA_ARGS;DEPENDS"
    ${ARGN}
  )
  get_filename_component(test_dir ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  if(${test_dir} IN_LIST exclude_categories)
    return()
  endif()

  message(STATUS "Setup test generation rules for prefix: " ${GEN_TEST_FILE_PREFIX})

  set(GEN_TEST_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/${GEN_TEST_INPUT})

  # Execute generator to get the files that will be generated.
  execute_process(COMMAND
                    ${PYTHON_EXECUTABLE}
                    ${GEN_TEST_GENERATOR}
                    ${GEN_TEST_INPUT}
                    -output-prefix ${CMAKE_CURRENT_BINARY_DIR}/${GEN_TEST_FILE_PREFIX}
                    -ext ${GEN_TEST_EXT}
                    -print-output-files
                    ${GEN_TEST_EXTRA_ARGS}
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  OUTPUT_VARIABLE GEN_TEST_OUTPUT_FILES
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ECHO_ERROR_VARIABLE)

  # Add the file to the out test list
  set(${GEN_TEST_TESTS} ${${GEN_TEST_TESTS}} ${GEN_TEST_OUTPUT_FILES} PARENT_SCOPE)

  set(extra_deps "")
  foreach(filename ${GEN_TEST_DEPENDS})
    list(APPEND extra_deps ${CMAKE_CURRENT_SOURCE_DIR}/${filename})
  endforeach()

  add_custom_command(OUTPUT ${GEN_TEST_OUTPUT_FILES}
    COMMAND
      ${PYTHON_EXECUTABLE}
      ${GEN_TEST_GENERATOR}
      ${GEN_TEST_INPUT}
      -output-prefix ${CMAKE_CURRENT_BINARY_DIR}/${GEN_TEST_FILE_PREFIX}
      -ext ${GEN_TEST_EXT}
      ${GEN_TEST_EXTRA_ARGS}
    DEPENDS
      ${GEN_TEST_GENERATOR}
      ${GEN_TEST_INPUT}
      ${extra_deps}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Generating test ${GEN_TEST_FILE_PREFIX}..."
    )

  add_custom_target(${GEN_TEST_FILE_PREFIX}_gen DEPENDS ${GEN_TEST_OUTPUT_FILES})
  add_dependencies(generate_test_sources ${GEN_TEST_FILE_PREFIX}_gen)
endfunction()

# create a target to encapsulate all test categories.
add_custom_target(test_conformance)

# create test executable targets for each test project using the build_sycl function
function(add_cts_test_helper)
  get_filename_component(test_dir ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  set(test_exe_name test_${ARGV0})
  set(test_cases_list ${ARGV1})

  if(NOT ${test_dir} IN_LIST exclude_categories)
    message(STATUS "Adding test: " ${test_exe_name})
  else()
    message(STATUS "Skipping excluded test: " ${test_exe_name})
    return()
  endif()

  if(NOT SYCL_CTS_ENABLE_HALF_TESTS)
    list(FILTER test_cases_list EXCLUDE REGEX .*_fp16\\.cpp$)
  endif()
  if(NOT SYCL_CTS_ENABLE_DOUBLE_TESTS)
    list(FILTER test_cases_list EXCLUDE REGEX .*_fp64\\.cpp$)
  endif()

  add_sycl_executable(NAME           ${test_exe_name}
                      OBJECT_LIBRARY ${test_exe_name}_objects
                      TESTS          ${test_cases_list})

  target_include_directories(${test_exe_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${test_exe_name} PUBLIC ${SYCL_CTS_DETAIL_OPTION_COMPILE_DEFINITIONS})

  set(info_dump_dir "${CMAKE_BINARY_DIR}/Testing")
  add_test(NAME ${test_exe_name}
           COMMAND ${test_exe_name}
                   --device ${SYCL_CTS_CTEST_DEVICE}
                   --info-dump "${info_dump_dir}/${test_exe_name}.info")

  target_link_libraries(${test_exe_name} PRIVATE CTS::util CTS::main_function oclmath)

  target_link_libraries(${test_exe_name} PRIVATE Catch2::Catch2 Threads::Threads)

  set_property(TARGET ${test_exe_name}
               PROPERTY FOLDER "Tests/${test_exe_name}")
  set_property(TARGET ${test_exe_name}_objects
               PROPERTY FOLDER "Tests/${test_exe_name}")

  add_dependencies(test_conformance ${test_exe_name})
endfunction()

# Create one *.exe-file from all of the provided *.cpp-files
function(add_cts_test)
  # To make check that any .cpp files are passed
  # List created because, direct check on "${ARGN}" gives false-positive result
  set(tests_list "${ARGN}")
  get_filename_component(test_dir ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  if (tests_list)
    set(test_exe_name ${test_dir})
    set(test_cases_list "${ARGN}")

    add_cts_test_helper(${test_exe_name} "${test_cases_list}")
  else()
    if(${test_dir} IN_LIST exclude_categories)
      message(STATUS "Skipping excluded test: " test_${test_dir})
    endif()
  endif()
endfunction()

# Create a separate *.exe-file from each of the provided *.cpp-files
function(add_independent_cts_tests)
  set(tests_list "${ARGN}")
  foreach(ind_test IN LISTS tests_list)
    if(EXISTS "${ind_test}")
      get_filename_component(cpp_name "${ind_test}" NAME_WE)
      set(test_exe_name "${cpp_name}")
      set(test_cases_list "${ind_test}")

      add_cts_test_helper(${test_exe_name} "${test_cases_list}")
    else()
      message(FATAL_ERROR "No file named ${ind_test}")
    endif()
  endforeach()
endfunction()

file(GLOB test_category_dirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)
list(REMOVE_ITEM test_category_dirs "common")
foreach(dir ${test_category_dirs})
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt")
    add_subdirectory(${dir})
  endif()
endforeach()
