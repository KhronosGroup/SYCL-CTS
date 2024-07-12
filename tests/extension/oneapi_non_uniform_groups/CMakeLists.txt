if(SYCL_CTS_ENABLE_EXT_ONEAPI_NON_UNIFORM_GROUPS_TESTS)
    function(configure_typed_test_case)
    cmake_parse_arguments(CTS
        "" "TYPE;GROUP;IN_FILENAME;OUT_FILENAME;TEST_LIST" "" ${ARGN})
        set(CTS_TYPE_NAME ${CTS_TYPE})
        set(CTS_GROUP_NAME ${CTS_GROUP})
        configure_file(${CTS_IN_FILENAME} ${CTS_OUT_FILENAME})
        list(APPEND ${CTS_TEST_LIST} "${CMAKE_CURRENT_BINARY_DIR}/${CTS_OUT_FILENAME}")
        set(${CTS_TEST_LIST} ${${CTS_TEST_LIST}} PARENT_SCOPE)
    endfunction()
    function(configure_test_case)
    cmake_parse_arguments(CTS
        "" "GROUP;IN_FILENAME;OUT_FILENAME;TEST_LIST" "" ${ARGN})
        set(CTS_GROUP_NAME ${CTS_GROUP})
        configure_file(${CTS_IN_FILENAME} ${CTS_OUT_FILENAME})
        list(APPEND ${CTS_TEST_LIST} "${CMAKE_CURRENT_BINARY_DIR}/${CTS_OUT_FILENAME}")
        set(${CTS_TEST_LIST} ${${CTS_TEST_LIST}} PARENT_SCOPE)
    endfunction()

    list(APPEND TYPED_TEMPLATE_LIST
        "group_joint_scan"
        "group_scan_over_group"
        "group_joint_reduce"
        "group_reduce_over_group"
    )
    list(APPEND TEMPLATE_LIST
        "group_barrier"
        "group_broadcast"
        "group_broadcast_fp16"
        "group_broadcast_fp64"
        "group_of"
        "group_permute"
        "group_permute_fp16"
        "group_permute_fp64"
        "group_shift"
        "group_shift_fp16"
        "group_shift_fp64"
    )

    set(TYPE_LIST "")
    get_std_type(TYPE_LIST)
    half_double_filter(TYPE_LIST)

    list(APPEND GROUP_LIST
      "oneapi_ext::ballot_group<sycl::sub_group>"
      "oneapi_ext::fixed_size_group<1, sycl::sub_group>"
      "oneapi_ext::fixed_size_group<2, sycl::sub_group>"
      "oneapi_ext::fixed_size_group<4, sycl::sub_group>"
      "oneapi_ext::fixed_size_group<8, sycl::sub_group>"
      "oneapi_ext::tangle_group<sycl::sub_group>"
      "oneapi_ext::opportunistic_group"
    )

    file(GLOB test_cases_list *.cpp)

    foreach(TEMP IN LISTS TYPED_TEMPLATE_LIST)
        foreach(TY IN LISTS TYPE_LIST)
            foreach(GROUP IN LISTS GROUP_LIST)
                if(TY STREQUAL "bool")
                    continue()
                endif()
                set(OUT_FILE "${TEMP}_${TY}_${GROUP}.cpp")
                STRING(REGEX REPLACE ":" "_" OUT_FILE ${OUT_FILE})
                STRING(REGEX REPLACE " " "_" OUT_FILE ${OUT_FILE})
                STRING(REGEX REPLACE "<" "_" OUT_FILE ${OUT_FILE})
                STRING(REGEX REPLACE ">" "_" OUT_FILE ${OUT_FILE})
                configure_typed_test_case(
                    TYPE "${TY}"
                    GROUP "${GROUP}"
                    IN_FILENAME "${TEMP}.cpp.in"
                    OUT_FILENAME ${OUT_FILE}
                    TEST_LIST test_cases_list
                )
            endforeach()
        endforeach()
    endforeach()

    foreach(TEMP IN LISTS TEMPLATE_LIST)
        foreach(GROUP IN LISTS GROUP_LIST)
            set(OUT_FILE "${TEMP}_${GROUP}.cpp")
            STRING(REGEX REPLACE ":" "_" OUT_FILE ${OUT_FILE})
            STRING(REGEX REPLACE " " "_" OUT_FILE ${OUT_FILE})
            STRING(REGEX REPLACE "<" "_" OUT_FILE ${OUT_FILE})
            STRING(REGEX REPLACE ">" "_" OUT_FILE ${OUT_FILE})
            configure_test_case(
                GROUP "${GROUP}"
                IN_FILENAME "${TEMP}.cpp.in"
                OUT_FILENAME ${OUT_FILE}
                TEST_LIST test_cases_list
            )
        endforeach()
    endforeach()

    add_cts_test(${test_cases_list})
endif()
