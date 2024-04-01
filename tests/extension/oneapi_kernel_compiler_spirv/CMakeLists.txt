if(SYCL_CTS_ENABLE_EXT_ONEAPI_KERNEL_COMPILER_SPIRV_TESTS)
    file(GLOB test_cases_list *.cpp)

    file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/kernels.spv" KERNELS_PATH)
    file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/kernels_fp16.spv" KERNELS_FP16_PATH)
    file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/kernels_fp64.spv" KERNELS_FP64_PATH)
    add_compile_definitions(KERNELS_PATH="${KERNELS_PATH}")
    add_compile_definitions(KERNELS_FP16_PATH="${KERNELS_FP16_PATH}")
    add_compile_definitions(KERNELS_FP64_PATH="${KERNELS_FP64_PATH}")

    add_cts_test(${test_cases_list})
endif()
