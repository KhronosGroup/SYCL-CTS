
if(SYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS)
  file(GLOB test_cases_list *.cpp)
else()
  file(GLOB test_cases_list scalars_sycl_types.cpp)
endif()

add_cts_test(${test_cases_list})
