if(SYCL_CTS_ENABLE_MAX_NUM_WORK_GROUPS)
    file(GLOB test_cases_list *.cpp)

    add_cts_test(${test_cases_list})
endif()
