find_program(INTEL_SYCL_C_EXECUTABLE clang HINTS ${INTEL_SYCL_ROOT}
    PATH_SUFFIXES bin)
find_program(INTEL_SYCL_CXX_EXECUTABLE clang++ HINTS ${INTEL_SYCL_ROOT}
    PATH_SUFFIXES bin)

set(CMAKE_C_COMPILER    ${INTEL_SYCL_C_EXECUTABLE})
set(CMAKE_CXX_COMPILER  ${INTEL_SYCL_CXX_EXECUTABLE})

if(NOT DEFINED INTEL_SYCL_TRIPLE)
   set(INTEL_SYCL_TRIPLE spir64-unknown-unknown-sycldevice)
endif()
message("Intel SYCL: compiling SYCL to ${INTEL_SYCL_TRIPLE}")

if(DEFINED INTEL_SYCL_FLAGS)
    message("Intel SYCL: compiling SYCL using `${INTEL_SYCL_FLAGS}`")
endif()

add_library(INTEL_SYCL::Runtime INTERFACE IMPORTED GLOBAL)
if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    set_target_properties(INTEL_SYCL::Runtime PROPERTIES
        INTERFACE_LINK_LIBRARIES    OpenCL::OpenCL
        INTERFACE_COMPILE_OPTIONS   "-fsycl;-fsycl-targets=${INTEL_SYCL_TRIPLE};${INTEL_SYCL_FLAGS}"
        INTERFACE_LINK_OPTIONS      "-fsycl;-fsycl-targets=${INTEL_SYCL_TRIPLE};${INTEL_SYCL_FLAGS}")
else()
    set_target_properties(INTEL_SYCL::Runtime PROPERTIES
        INTERFACE_LINK_LIBRARIES    OpenCL::OpenCL
        INTERFACE_COMPILE_OPTIONS   "-fsycl;-fsycl-targets=${INTEL_SYCL_TRIPLE};${INTEL_SYCL_FLAGS}"
        INTERFACE_LINK_OPTIONS      "-fsycl;-fsycl-device-code-split=per_source;-fsycl-targets=${INTEL_SYCL_TRIPLE};${INTEL_SYCL_FLAGS}")
endif()

add_library(SYCL::SYCL INTERFACE IMPORTED GLOBAL)
set_target_properties(SYCL::SYCL PROPERTIES
    INTERFACE_LINK_LIBRARIES INTEL_SYCL::Runtime)

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

    target_compile_options(${object_lib_name} PRIVATE
        $<TARGET_PROPERTY:INTEL_SYCL::Runtime,INTERFACE_COMPILE_OPTIONS>)

    target_link_libraries(${exe_name} PUBLIC INTEL_SYCL::Runtime)
    # CMake < 3.14 doesn't support INTERFACE_LINK_OPTIONS otherwise we just use
    # LINK_LIBRARIES
    if(${CMAKE_VERSION} VERSION_LESS 3.14)
        target_link_options(${exe_name} PRIVATE
            $<TARGET_PROPERTY:INTEL_SYCL::Runtime,INTERFACE_LINK_OPTIONS>)
    endif()
endfunction()
