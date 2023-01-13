function(configure_test_case)
    cmake_parse_arguments(CTS
            "" "HEADER;NAMESPACE;IN_FILENAME;OUT_FILENAME;TEST_LIST" "" ${ARGN})
    configure_file(${CTS_IN_FILENAME} ${CTS_OUT_FILENAME})
    list(APPEND ${CTS_TEST_LIST} "${CMAKE_CURRENT_BINARY_DIR}/${CTS_OUT_FILENAME}")
    set(${CTS_TEST_LIST} ${${CTS_TEST_LIST}} PARENT_SCOPE)
endfunction()

# SYCL 1.2.1
configure_test_case(
        HEADER "<CL/sycl.hpp>"
        NAMESPACE "::cl::sycl"
        IN_FILENAME "namespace.cpp.in"
        OUT_FILENAME "namespace_1_2_1.cpp"
        TEST_LIST test_cases_list)

# SYCL 2020
configure_test_case(
        HEADER "<sycl/sycl.hpp>"
        NAMESPACE "::sycl"
        IN_FILENAME "namespace.cpp.in"
        OUT_FILENAME "namespace_2020.cpp"
        TEST_LIST test_cases_list)

add_cts_test(${test_cases_list})
