if(SYCL_CTS_ENABLE_KHR_QUEUE_EMPTY_QUERY_TESTS)
    file(GLOB test_cases_list *.cpp)

    add_cts_test(${test_cases_list})
endif()
