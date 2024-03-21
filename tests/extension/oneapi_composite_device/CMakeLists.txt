if(SYCL_CTS_ENABLE_EXT_ONEAPI_COMPOSITE_DEVICE_TESTS)
  set(test_cases_list
    aspects.cpp
    descendent_device.cpp
    device_info_descriptors.cpp
    enumerating_composite_devices.cpp
    more_complex_test_cases.cpp
  )

  add_cts_test(${test_cases_list})
endif()
