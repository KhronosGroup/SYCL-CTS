# BUILD_SPIR
# Runs the device compiler creating the stub and the bc files.
function(BUILD_SPIR exe_name spir_target_name source_file output_path)
    set(target ${spir_target_name})
    
    set(SPIR32_PREDEFINE  "__DEVICE_SPIR32__")
    set(SPIR32_STUB_FILE  ${spir_target_name}.cpp.sycl)
    set(SPIR32_BC_FILE    ${spir_target_name}.cpp.bc)
    
    set(output_bc ${output_path}/${SPIR32_BC_FILE})
    set(output_stub ${output_path}/${SPIR32_STUB_FILE})
    
    # Add any user-defined include to the device compiler
    get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES) 
    set(device_compiler_includes "")
    foreach(dir ${dirs})
        set(device_compiler_includes "-I${dir}" ${device_compiler_includes})
    endforeach()

    if (CMAKE_INCLUDE_PATH)
        foreach(dir ${CMAKE_INCLUDE_PATH})
            set(device_compiler_includes "-I${dir}" ${device_compiler_includes})
        endforeach()
    endif()

    add_custom_command(
    OUTPUT ${output_bc}
           ${output_stub}
    COMMAND ${DEVICE_COMPILER}
            ${DEVICE_COMPILER_FLAGS}
              -O2
            -Wno-ignored-attributes
            -O2
            -sycl
            -emit-llvm
            -D${SPIR32_PREDEFINE}
            -DBUILD_PLATFORM_SPIR
            ${PLATFORM_SPECIFIC_ARGS}
            ${device_compiler_includes}
            -o ${output_bc}
            -c ${source_file}
    DEPENDS ${source_file}
    WORKING_DIRECTORY ${output_path}
    COMMENT "Building SPIR object ${output_bc}")

    add_custom_target(${spir_target_name}_spir DEPENDS ${output_stub})
    set(DEPENDANT_OBJ_FILES ${DEPENDANT_OBJ_FILES} ${output_bc})
    
    set_property(TARGET ${spir_target_name}_spir PROPERTY FOLDER "Tests/${exe_name}/${exe_name}_spir")
    
    set(return_spir_target ${spir_target_name})

endfunction()


# BUILD_SYCL FUNCTION
# Builds a SYCL program, compiling multiple SYCL test case source files into a test executable, invoking a single-source/device compiler
# Parameters are:
#   - exe_name               Name of the test executable
#   - test_main              Path to the main.cpp for the test executable
#   - test_cases_list        List of SYCL test case source files to be built into the test executable
#   - destination_exe_path   Path where the test executable will be placed
#   - destination_stub_path  Path where the intermediate stub(spir, ll, etc.) files will be placed
function(BUILD_SYCL_IMPLEMENTATION exe_name test_main test_cases_list destination_exe_path destination_stub_path)
    set(source_files
    ${test_main}
    ${test_cases_list}
    )
    # executable
    add_executable(${exe_name} ${source_files})

    set_property(TARGET ${exe_name} PROPERTY PROJECT_LABEL ${exe_name})
    set_property(TARGET ${exe_name} PROPERTY FOLDER "Tests/${exe_name}")

    set_target_properties(${exe_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${destination_exe_path}/)
    set_target_properties(${exe_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${destination_exe_path}/)
    set_target_properties(${exe_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${destination_exe_path}/)

    get_property(spir_targets_list GLOBAL PROPERTY stl)
    # build spir    
    set(output_stubs_list "")
    foreach(source_file ${test_cases_list})
        get_filename_component(spir_target_name ${source_file} NAME_WE)
        list(FIND spir_targets_list ${spir_target_name} found)      
        if(${found} EQUAL -1)
            build_spir("${exe_name}" "${spir_target_name}" "${source_file}" "${destination_stub_path}")
            list(APPEND spir_targets_list ${spir_target_name})
        endif()
        add_dependencies(${exe_name} ${spir_target_name}_spir)
        list(APPEND output_stubs_list "${destination_stub_path}/${spir_target_name}.cpp.sycl")
    endforeach()    
    set_property(GLOBAL PROPERTY stl ${spir_targets_list})

    # force stub-header inclusion so that host version
    if(WIN32)
        set(compile_flags "/FI\"${output_stubs_list}\"")
    else()
        set(output_stubs_include_string "")
        foreach(output_stub ${output_stubs_list})
            set(output_stubs_include_string "${output_stubs_include_string} -include ${output_stub}")
        endforeach()
        set(compile_flags "${output_stub_include_string} ${HOST_COMPILER_FLAGS}")
    endif()
    set_target_properties(${exe_name} PROPERTIES COMPILE_FLAGS "${compile_flags}")  

    # runtime dependencies
    target_link_libraries(${exe_name} ${SYCL_LIB_NAME})
    target_link_libraries(${exe_name} ${IMAGE_KERNEL_LIB_NAME})
    target_link_libraries(${exe_name} ${IMAGE_HOST_LIB_NAME})

    if(NOT APPLE)
        target_link_libraries(${exe_name} ${OPENCL_LIB_NAME} )
    else(NOT APPLE)
        set_target_properties(${exe_name} PROPERTIES LINK_FLAGS "-framework OpenCL")
    endif(NOT APPLE)

    if(WIN32)
        set_target_properties(${exe_name} PROPERTIES LINK_FLAGS /SAFESEH:NO)
    else(WIN32)
        set_target_properties(${exe_name} PROPERTIES LINK_FLAGS "-pthread")
    endif(WIN32)

endfunction()
