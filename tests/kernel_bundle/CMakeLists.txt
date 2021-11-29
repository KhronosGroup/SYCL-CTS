file(GLOB test_cases_list *.cpp)

set(independent_cases_list
  has_kernel_bundle_multiple_kernels_ctx_only.cpp
  has_kernel_bundle_multiple_kernels_dev.cpp
)

list(TRANSFORM independent_cases_list PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(REMOVE_ITEM test_cases_list ${independent_cases_list})

add_cts_test(${test_cases_list})
add_independent_cts_tests(${independent_cases_list})
