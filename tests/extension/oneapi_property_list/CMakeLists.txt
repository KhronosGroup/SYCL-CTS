if(SYCL_ENABLE_EXT_ONEAPI_PROPERTY_LIST_TESTS)
  file(GLOB test_cases_list *.cpp)

  add_cts_test(${test_cases_list})
endif()
