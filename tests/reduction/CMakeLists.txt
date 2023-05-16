file(GLOB test_cases_list *.cpp)

set(independent_cases_list
reduction_without_identity_param_no_one_item_core.cpp
reduction_without_identity_param_item_twice_core.cpp
reduction_without_identity_param_even_item_core.cpp
reduction_without_identity_param_core.cpp
)

list(TRANSFORM independent_cases_list PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(REMOVE_ITEM test_cases_list ${independent_cases_list})

add_cts_test(${test_cases_list})
add_independent_cts_tests(${independent_cases_list})
