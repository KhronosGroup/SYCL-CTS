file(GLOB test_cases_list *.cpp)

if(SYCL_IMPLEMENTATION STREQUAL SimSYCL)
  message(WARNING "SimSYCL does not provide true concurrency between host and device, disabling USM atomic tests")
  list(FILTER test_cases_list EXCLUDE REGEX usm_atomic_access_.*\\.cpp$)
endif()


add_cts_test(${test_cases_list})
