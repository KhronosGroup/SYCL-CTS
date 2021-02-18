# ************************************************************************
#
#   SYCL Conformance Test Suite
#
#   Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
#
# ************************************************************************

find_path(ComputeCpp_INCLUDE_DIRS CL/sycl.hpp
    HINTS ${COMPUTECPP_INSTALL_DIR} ${ComputeCpp_DIR}
    PATH_SUFFIXES include)

find_library(ComputeCpp_LIBRARIES
    NAMES ComputeCpp_vs2015 ComputeCpp
    HINTS ${COMPUTECPP_INSTALL_DIR} ${ComputeCpp_DIR}
    PATH_SUFFIXES lib)

find_program(ComputeCpp_EXECUTABLE
    compute++
    HINTS ${COMPUTECPP_INSTALL_DIR} ${ComputeCpp_DIR}
    PATH_SUFFIXES bin)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ComputeCpp
    REQUIRED_VARS ComputeCpp_EXECUTABLE ComputeCpp_LIBRARIES ComputeCpp_INCLUDE_DIRS)

find_package(Threads REQUIRED)

add_executable(ComputeCpp::compute++ IMPORTED)
set_target_properties(ComputeCpp::compute++ PROPERTIES
    IMPORTED_LOCATION ${ComputeCpp_EXECUTABLE})

add_library(ComputeCpp::Runtime UNKNOWN IMPORTED GLOBAL)
set_target_properties(ComputeCpp::Runtime PROPERTIES
    IMPORTED_LOCATION                    "${ComputeCpp_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES        "${ComputeCpp_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES             "OpenCL::OpenCL;Threads::Threads"
    INTERFACE_COMPILE_DEFINITIONS        "SYCL_LANGUAGE_VERSION=2020"
    INTERFACE_DEVICE_COMPILE_DEFINITIONS "SYCL_LANGUAGE_VERSION=2020"
    INTERFACE_DEVICE_COMPILE_OPTIONS     "-sycl;-std=c++17"
  )
if (WIN32)
    set_property(TARGET ComputeCpp::Runtime APPEND PROPERTY
                 INTERFACE_DEVICE_COMPILE_DEFINITIONS
                 "_SIZE_T_DEFINED;_NO_CRT_STDIO_INLINE")
endif()

add_library(SYCL::SYCL INTERFACE IMPORTED GLOBAL)
set_target_properties(SYCL::SYCL PROPERTIES INTERFACE_LINK_LIBRARIES ComputeCpp::Runtime)

set(COMPUTECPP_USER_FLAGS "" CACHE STRING "User flags for compute++")
separate_arguments(COMPUTECPP_USER_FLAGS)
mark_as_advanced(COMPUTECPP_USER_FLAGS)

# build_spir
# Runs the device compiler on a single source file, creating the stub and the bc files.
function(build_spir exe_name spir_target_name source_file output_path)
    set(target ${spir_target_name})

    set(stub_file  ${spir_target_name}.cpp.sycl)
    set(bc_file    ${spir_target_name}.cpp.bc)

    set(output_bc ${output_path}/${bc_file})
    set(output_stub ${output_path}/${stub_file})

    if(WIN32 AND MSVC)
        set(platform_specific_args -fdiagnostics-format=msvc)
    endif()

    set(device_compile_definitions "$<TARGET_PROPERTY:ComputeCpp::Runtime,INTERFACE_DEVICE_COMPILE_DEFINITIONS>")
    set(device_compile_options "$<TARGET_PROPERTY:ComputeCpp::Runtime,INTERFACE_DEVICE_COMPILE_OPTIONS>")
    set(include_directories "$<TARGET_PROPERTY:${exe_name},INCLUDE_DIRECTORIES>")

    add_custom_command(
    OUTPUT  ${output_bc} ${output_stub}
    COMMAND ComputeCpp::compute++
            -Wno-ignored-attributes
            -O2
            -mllvm -inline-threshold=1000
            -intelspirmetadata
            ${COMPUTECPP_USER_FLAGS}
            ${platform_specific_args}
            $<$<BOOL:${include_directories}>:-I\"$<JOIN:${include_directories},\"\t-I\">\">
            $<$<BOOL:${device_compile_definitions}>:-D$<JOIN:${device_compile_definitions},\t-D>>
            $<JOIN:${device_compile_options},\t>
            -o ${output_bc}
            -c ${source_file}
    DEPENDS ${source_file}
    WORKING_DIRECTORY ${output_path}
    COMMENT "Building SPIR object ${output_bc}")

    add_custom_target(${spir_target_name}_spir DEPENDS ${output_stub})
    set_property(TARGET ${spir_target_name}_spir PROPERTY FOLDER "Tests/${exe_name}/${exe_name}_spir")
endfunction()

# add_sycl_executable_implementation function
# Builds a SYCL program, compiling multiple SYCL test case source files into a test executable, invoking a single-source/device compiler
# Parameters are:
#   - NAME             Name of the test executable
#   - OBJECT_LIBRARY   Name of the object library of all the compiled test cases
#   - TESTS            List of SYCL test case source files to be built into the test executable
function(add_sycl_executable_implementation)
    cmake_parse_arguments(args "" "NAME;OBJECT_LIBRARY" "TESTS" ${ARGN})
    set(exe_name              ${args_NAME})
    set(object_lib_name       ${args_OBJECT_LIBRARY})
    set(test_cases_list       ${args_TESTS})
    set(destination_stub_path "${PROJECT_BINARY_DIR}/bin/intermediate")

    add_library(${object_lib_name} OBJECT ${test_cases_list})
    add_executable(${exe_name} $<TARGET_OBJECTS:${object_lib_name}>)
    set_target_properties(${object_lib_name} PROPERTIES
        INCLUDE_DIRECTORIES $<TARGET_PROPERTY:${exe_name},INCLUDE_DIRECTORIES>
        COMPILE_DEFINITIONS $<TARGET_PROPERTY:${exe_name},COMPILE_DEFINITIONS>
        COMPILE_OPTIONS     $<TARGET_PROPERTY:${exe_name},COMPILE_OPTIONS>
        COMPILE_FEATURES    $<TARGET_PROPERTY:${exe_name},COMPILE_FEATURES>
        POSITION_INDEPENDENT_CODE ON)

    foreach(source_file ${test_cases_list})
        get_filename_component(spir_target_name ${source_file} NAME_WE)
        build_spir("${exe_name}" "${spir_target_name}" "${source_file}" "${destination_stub_path}")
        add_dependencies(${exe_name} ${spir_target_name}_spir)
        set(output_stub "${destination_stub_path}/${spir_target_name}.cpp.sycl")
        if(WIN32)
            set_source_files_properties(${source_file} PROPERTIES
                OBJECT_DEPENDS "${output_stub}"
                COMPILE_FLAGS  /FI\"${output_stub}\")
        else()
            set_source_files_properties(${source_file} PROPERTIES
                OBJECT_DEPENDS "${output_stub}"
                COMPILE_FLAGS  "-include ${output_stub}")
	    endif()
    endforeach()

    set_target_properties(${exe_name} PROPERTIES
        FOLDER         "Tests/${exe_name}"
        LINK_LIBRARIES "ComputeCpp::Runtime;$<$<BOOL:${WIN32}>:-SAFESEH:NO>"
        BUILD_RPATH    "$<TARGET_FILE_DIR:ComputeCpp::Runtime>")
endfunction()
