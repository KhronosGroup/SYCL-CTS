
file(GLOB test_cases_list *.cpp)

if(NOT SYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS)
  list(FILTER test_cases_list EXCLUDE REGEX scalars_interoperability_types.cpp)
endif()

add_cts_test(${test_cases_list})
