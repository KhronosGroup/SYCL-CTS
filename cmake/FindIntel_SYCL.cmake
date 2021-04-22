if(${CMAKE_CXX_COMPILER_ID} MATCHES "GNU" OR
   ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    find_program(INTEL_SYCL_CXX_EXECUTABLE NAMES dpcpp clang++ HINTS ${INTEL_SYCL_ROOT}
        PATH_SUFFIXES bin)
else()
    # Remove /machine: option which is not supported by clang-cl
    string(REPLACE "/machine:x64" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
    # Remove /subsystem option which is not supported by clang-cl
    string(REPLACE "/subsystem:console" "" CMAKE_CREATE_CONSOLE_EXE ${CMAKE_CREATE_CONSOLE_EXE})
    find_program(INTEL_SYCL_CXX_EXECUTABLE NAMES dpcpp clang-cl HINTS ${INTEL_SYCL_ROOT}
        PATH_SUFFIXES bin)
endif()

if(NOT DEFINED INTEL_SYCL_TRIPLE)
   set(INTEL_SYCL_TRIPLE spir64-unknown-unknown-sycldevice)
endif()
message("Intel SYCL target triple: ${INTEL_SYCL_TRIPLE}")

# Set precise fp-model for Intel Compiler
if(WIN32)
    set(INTEL_FP_FLAG "/fp:precise")
else()
    set(INTEL_FP_FLAG "-ffp-model=precise")
endif()
set(CMAKE_CXX_FLAGS "${INTEL_FP_FLAG} ${CMAKE_CXX_FLAGS}")

set(INTEL_SYCL_FLAGS "-fsycl;-fsycl-targets=${INTEL_SYCL_TRIPLE};-sycl-std=2020;${INTEL_SYCL_FLAGS}")
message("Intel SYCL compiler flags: `${INTEL_SYCL_FLAGS}`")

add_library(INTEL_SYCL::Runtime INTERFACE IMPORTED GLOBAL)
set_target_properties(INTEL_SYCL::Runtime PROPERTIES
  INTERFACE_LINK_LIBRARIES    OpenCL::OpenCL
  INTERFACE_COMPILE_OPTIONS   "${INTEL_SYCL_FLAGS}")

if(${INTEL_SYCL_TRIPLE} MATCHES ".*-nvidia-cuda-.*")
#   The DPC++ compiler currently retains a requirement for certain OpenCL definitions when using CUDA. 
#   The INTERFACE_LINK_OPTIONS definition is required, however the '-fsycl-device-code-split=' option 
#   is not yet supported and has been removed.
    set_target_properties(INTEL_SYCL::Runtime PROPERTIES
        INTERFACE_LINK_OPTIONS      "${INTEL_SYCL_FLAGS}")
else()
    set_target_properties(INTEL_SYCL::Runtime PROPERTIES
        INTERFACE_LINK_OPTIONS      "${INTEL_SYCL_FLAGS};-fsycl-device-code-split=per_source")
endif()

set(CMAKE_CXX_COMPILER          ${INTEL_SYCL_CXX_EXECUTABLE})
# Use SYCL compiler instead of default linker for building SYCL application
set(CMAKE_CXX_LINK_EXECUTABLE   "${INTEL_SYCL_CXX_EXECUTABLE} <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")

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
