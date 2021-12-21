if(SYCL_EXT_ONEAPI_ENABLE_SUB_GROUP_MASK_TESTS)
  file(GLOB test_cases_list *.cpp)

  add_cts_test(${test_cases_list})
endif()
