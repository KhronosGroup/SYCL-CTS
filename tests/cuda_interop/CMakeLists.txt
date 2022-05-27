
if(SYCL_CTS_ENABLE_CUDA_INTEROP_TESTS)
    file(GLOB test_cases_list *.cpp *.cu)
    add_cts_test(${test_cases_list})
endif()
