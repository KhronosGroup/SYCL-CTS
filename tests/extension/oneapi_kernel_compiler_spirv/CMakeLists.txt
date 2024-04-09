function(load_spirv SPIRV_FILE RESULT)
    file(READ ${SPIRV_FILE} bytes HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1, " values ${bytes})
    string(REGEX REPLACE ", $" "" values ${values})
    set(${RESULT} ${values} PARENT_SCOPE)
endfunction()

if(SYCL_CTS_ENABLE_EXT_ONEAPI_KERNEL_COMPILER_SPIRV_TESTS)
    file(GLOB test_cases_list *.cpp)

    load_spirv("${CMAKE_CURRENT_SOURCE_DIR}/kernels.spv" KERNELS)
    load_spirv("${CMAKE_CURRENT_SOURCE_DIR}/kernels_fp16.spv" KERNELS_FP16)
    load_spirv("${CMAKE_CURRENT_SOURCE_DIR}/kernels_fp64.spv" KERNELS_FP64)
    add_compile_definitions(KERNELS=${KERNELS})
    add_compile_definitions(KERNELS_FP16=${KERNELS_FP16})
    add_compile_definitions(KERNELS_FP64=${KERNELS_FP64})

    add_cts_test(${test_cases_list})
endif()
