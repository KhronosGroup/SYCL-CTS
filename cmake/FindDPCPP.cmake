if(${CMAKE_CXX_COMPILER_ID} MATCHES "GNU" OR
   ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    find_program(DPCPP_CXX_EXECUTABLE NAMES dpcpp clang++
        HINTS ${DPCPP_INSTALL_DIR}
        PATH_SUFFIXES bin)
else()
    # Remove /machine: option which is not supported by clang-cl
    string(REPLACE "/machine:x64" "" CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS}")
    # Remove /subsystem option which is not supported by clang-cl
    string(REPLACE "/subsystem:console" "" CMAKE_CREATE_CONSOLE_EXE
        "${CMAKE_CREATE_CONSOLE_EXE}")
    string(REPLACE "/subsystem:console" "" CMAKE_CXX_CREATE_CONSOLE_EXE
        "${CMAKE_CXX_CREATE_CONSOLE_EXE}")
    find_program(DPCPP_CXX_EXECUTABLE NAMES dpcpp clang-cl
        HINTS ${DPCPP_INSTALL_DIR}
        PATH_SUFFIXES bin)
endif()

# Set SYCL compilation mode, SYCL 2020 standard version and user provided flags
set(DPCPP_FLAGS "-fsycl;-sycl-std=2020;${DPCPP_FLAGS}")
# Set target triple(s) if specified
if(DEFINED DPCPP_TARGET_TRIPLES)
    set(DPCPP_FLAGS "${DPCPP_FLAGS};-fsycl-targets=${DPCPP_TARGET_TRIPLES};")
    message("DPC++: compiling tests to ${DPCPP_TARGET_TRIPLES}")
    if(${DPCPP_TARGET_TRIPLES} MATCHES ".*-nvidia-cuda-.*")
        add_definitions(-DSYCL_CTS_INTEL_PI_CUDA)
    endif()
endif()
message("DPC++ compiler flags: `${DPCPP_FLAGS}`")

# Explicitly set fp-model to precise to produce reliable results for floating
# point operations.
if(WIN32)
    set(DPCPP_FP_FLAG "/fp:precise")
else()
    set(DPCPP_FP_FLAG "-ffp-model=precise")
endif()
set(CMAKE_CXX_FLAGS "${DPCPP_FP_FLAG} ${CMAKE_CXX_FLAGS}")

# Disable range rounding feature to reduce # of SYCL kernels.
set(CMAKE_CXX_FLAGS "-D__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__ ${CMAKE_CXX_FLAGS}")

option(DPCPP_DISABLE_SYCL2020_DEPRECATION_WARNINGS
  "Disable SYCL 2020 deprecation warnings" ON)
if(DPCPP_DISABLE_SYCL2020_DEPRECATION_WARNINGS)
    set(CMAKE_CXX_FLAGS "-DSYCL2020_DISABLE_DEPRECATION_WARNINGS ${CMAKE_CXX_FLAGS}")
endif()

option(DPCPP_SYCL2020_CONFORMANT_APIS
  "Comply with the SYCL 2020 specification" ON)
if(DPCPP_SYCL2020_CONFORMANT_APIS)
    set(CMAKE_CXX_FLAGS "-DSYCL2020_CONFORMANT_APIS ${CMAKE_CXX_FLAGS}")
endif()

add_library(DPCPP::Runtime INTERFACE IMPORTED GLOBAL)
set_target_properties(DPCPP::Runtime PROPERTIES
  INTERFACE_LINK_LIBRARIES    OpenCL::OpenCL
  INTERFACE_COMPILE_OPTIONS   "${DPCPP_FLAGS}"
  INTERFACE_LINK_OPTIONS      "${DPCPP_FLAGS}")

set(CMAKE_CXX_COMPILER ${DPCPP_CXX_EXECUTABLE})
# Use DPC++ compiler instead of default linker for building SYCL application
set(CMAKE_CXX_LINK_EXECUTABLE "${DPCPP_CXX_EXECUTABLE} <FLAGS> \
    <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")

add_library(SYCL::SYCL INTERFACE IMPORTED GLOBAL)
set_target_properties(SYCL::SYCL PROPERTIES
    INTERFACE_LINK_LIBRARIES DPCPP::Runtime)

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
        $<TARGET_PROPERTY:DPCPP::Runtime,INTERFACE_COMPILE_OPTIONS>)

    target_link_libraries(${exe_name} PUBLIC DPCPP::Runtime)
endfunction()
