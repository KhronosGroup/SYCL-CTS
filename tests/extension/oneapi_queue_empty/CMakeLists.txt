if(SYCL_CTS_ENABLE_EXT_ONEAPI_QUEUE_EMPTY)
    file(GLOB test_cases_list *.cpp)

    add_cts_test(${test_cases_list})
endif()
