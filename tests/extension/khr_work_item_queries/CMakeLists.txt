if(SYCL_CTS_ENABLE_KHR_WORK_ITEM_QUERIES_TESTS)
    file(GLOB test_cases_list *.cpp)

    add_cts_test(${test_cases_list})
endif()
