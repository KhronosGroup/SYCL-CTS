if(SYCL_CTS_ENABLE_EXT_ONEAPI_NON_UNIFORM_GROUPS_TESTS)
    function(configure_test_case)
    cmake_parse_arguments(CTS
        "" "TYPE;IN_FILENAME;OUT_FILENAME;TEST_LIST" "" ${ARGN})
    set(CTS_TYPE_NAME ${CTS_TYPE})
    configure_file(${CTS_IN_FILENAME} ${CTS_OUT_FILENAME})
    list(APPEND ${CTS_TEST_LIST} "${CMAKE_CURRENT_BINARY_DIR}/${CTS_OUT_FILENAME}")
    set(${CTS_TEST_LIST} ${${CTS_TEST_LIST}} PARENT_SCOPE)
    endfunction()

    list(APPEND TEMPLATE_LIST
        "group_joint_scan"
        "group_scan_over_group"
        "group_joint_reduce"
        "group_reduce_over_group"
    )
    set(TYPE_LIST "")
    get_std_type(TYPE_LIST)
    half_double_filter(TYPE_LIST)

    file(GLOB test_cases_list *.cpp)

    foreach(TEMP IN LISTS TEMPLATE_LIST)
        foreach(TY IN LISTS TYPE_LIST)
            if(TY STREQUAL "bool")
            continue()
            endif()
            set(OUT_FILE "${TEMP}_${TY}.cpp")
            STRING(REGEX REPLACE ":" "_" OUT_FILE ${OUT_FILE})
            STRING(REGEX REPLACE " " "_" OUT_FILE ${OUT_FILE})
            configure_test_case(
            TYPE "${TY}"
            IN_FILENAME "${TEMP}.cpp.in"
            OUT_FILENAME ${OUT_FILE}
            TEST_LIST test_cases_list)
        endforeach()
    endforeach()

    add_cts_test(${test_cases_list})
endif()
