file(GLOB test_cases_list *.cpp)

set(independent_cases_list
  get_kernel_ids_multiple_kernels.cpp
  get_kernel_ids_single_kernel.cpp
  get_kernel_ids_no_kernels.cpp
)

list(TRANSFORM independent_cases_list PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(REMOVE_ITEM test_cases_list ${independent_cases_list})

add_cts_test(${test_cases_list})
add_independent_cts_tests(${independent_cases_list})
