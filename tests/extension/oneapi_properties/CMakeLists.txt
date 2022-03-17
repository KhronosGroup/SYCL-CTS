if(SYCL_ENABLE_EXT_ONEAPI_PROPERTIES)
  # Temporarily define for device_global properties
  # TODO: remove when device_global fully implemented
  add_definitions(-DSYCL_EXT_ONEAPI_DEVICE_GLOBAL)
  file(GLOB test_cases_list *.cpp)

  add_cts_test(${test_cases_list})
endif()
