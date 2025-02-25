if(SYCL_CTS_ENABLE_EXT_ONEAPI_NUM_COMPUTE_UNITS_TESTS)
  set(test_cases_list
    device_info_descriptor.cpp
  )

  add_cts_test(${test_cases_list})
endif()