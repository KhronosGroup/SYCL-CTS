find_program(SYCLCC_EXECUTABLE syclcc-clang
    PATH_SUFFIXES bin)

if(NOT SYCLCC_EXECUTABLE)
  message(SEND_ERROR "Could not find hipSYCL syclcc-clang compiler")
endif()

set(CMAKE_CXX_COMPILER  ${SYCLCC_EXECUTABLE})

add_library(SYCL::SYCL INTERFACE IMPORTED GLOBAL)
# add_sycl_executable_implementation function
# Builds a SYCL program, compiling multiple SYCL test case source files into a
# test executable, invoking a single-source/device compiler
# Parameters are:
#   - NAME             Name of the test executable
#   - OBJECT_LIBRARY   Name of the object library of all the compiled test cases
#   - TESTS            List of SYCL test case source files to be built into the
# test executable
function(add_sycl_executable_implementation)
    cmake_parse_arguments(args "" "NAME;OBJECT_LIBRARY" "TESTS" ${ARGN})
    set(exe_name            ${args_NAME})
    set(object_lib_name     ${args_OBJECT_LIBRARY})
    set(test_cases_list     ${args_TESTS})

    add_library(${object_lib_name} OBJECT ${test_cases_list})
    add_executable(${exe_name} $<TARGET_OBJECTS:${object_lib_name}>)

    set_target_properties(${object_lib_name} PROPERTIES
        INCLUDE_DIRECTORIES $<TARGET_PROPERTY:${exe_name},INCLUDE_DIRECTORIES>
        COMPILE_DEFINITIONS $<TARGET_PROPERTY:${exe_name},COMPILE_DEFINITIONS>
        COMPILE_OPTIONS     $<TARGET_PROPERTY:${exe_name},COMPILE_OPTIONS>
        COMPILE_FEATURES    $<TARGET_PROPERTY:${exe_name},COMPILE_FEATURES>
        POSITION_INDEPENDENT_CODE ON)

endfunction()

# Adds device compiler definitions
# This functions is a no-op because add_definitions should take care of it
function(add_device_compiler_definitions_implementation)
endfunction()
