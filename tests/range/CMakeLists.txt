file(GLOB test_cases_list *.cpp)

if(SYCL_IMPLEMENTATION STREQUAL "DPCPP")
  # in FindDPCPP.cmake, range rounding is disabled globally to reduce
  # the number of kernels generated per range parallel_for. However,
  # range rounding enables launching of kernels over >32bit ranges,
  # and the range_api test in full conformance such kernel launches
  # over large ranges, so we renable it here.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -U__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__")
endif()

add_cts_test(${test_cases_list})
