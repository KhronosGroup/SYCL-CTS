if(SYCL_CTS_ENABLE_KHR_STATIC_ADDRSPACE_CAST_TESTS)
    file(GLOB test_cases_list *.cpp)

    add_cts_test(${test_cases_list})
endif()
