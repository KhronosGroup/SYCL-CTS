file(GLOB test_cases_list *.cpp)

if(SYCL_IMPLEMENTATION STREQUAL "DPCPP")
  # For DPCPP, range rounding is disabled globally. However,
  # the range_api test needs range rounding enabled to pass,
  # so renable it here.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -U__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__")
endif()

add_cts_test(${test_cases_list})
