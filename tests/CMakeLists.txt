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

function(get_no_vec_alias_type OUT_LIST)
  set(NO_VEC_ALIAS_LIST "")
  list(APPEND NO_VEC_ALIAS_LIST sycl::byte)

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

# create a target to group all tests together into one test executable
add_executable(test_all)

# create test executable targets for each test project using the build_sycl function
function(add_cts_test)
  get_filename_component(test_dir ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  set(test_exe_name test_${test_dir})
  set(test_cases_list ${ARGN})

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

  set(info_dump_dir "${CMAKE_BINARY_DIR}/Testing")
  add_test(NAME ${test_exe_name}
           COMMAND ${test_exe_name}
                   --device ${SYCL_CTS_CTEST_DEVICE}
                   --info-dump "${info_dump_dir}/${test_exe_name}.info")

  target_link_libraries(${test_exe_name} PRIVATE CTS::util CTS::main_function oclmath)

  set_property(TARGET ${test_exe_name}
               PROPERTY FOLDER "Tests/${test_exe_name}")
  set_property(TARGET ${test_exe_name}_objects
               PROPERTY FOLDER "Tests/${test_exe_name}")

  target_sources(test_all PRIVATE $<TARGET_OBJECTS:${test_exe_name}_objects>)
endfunction()

file(GLOB test_category_dirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)
list(REMOVE_ITEM test_category_dirs "common")
foreach(dir ${test_category_dirs})
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt")
    add_subdirectory(${dir})
  endif()
endforeach()

target_link_libraries(test_all PRIVATE CTS::util CTS::main_function oclmath)
