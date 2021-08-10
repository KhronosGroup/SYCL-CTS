set(TEST_CASES_LIST "")

set(TYPE_LIST "")
get_std_type(TYPE_LIST)
get_no_vec_alias_type(TYPE_LIST)

if(SYCL_CTS_ENABLE_FULL_CONFORMANCE)
    set(ENABLE_AS_CONVERT_TYPES "16_64")
    get_fixed_width_type(TYPE_LIST)
else()
    set(ENABLE_AS_CONVERT_TYPES "_")
endif()

if(NOT SYCL_CTS_ENABLE_DOUBLE_TESTS)
    list(REMOVE_ITEM TYPE_LIST double)
    STRING(REGEX REPLACE "64" "" ENABLE_AS_CONVERT_TYPES ${ENABLE_AS_CONVERT_TYPES})
endif()

if(NOT SYCL_CTS_ENABLE_HALF_TESTS)
    list(REMOVE_ITEM TYPE_LIST sycl::half)
    STRING(REGEX REPLACE "16" "" ENABLE_AS_CONVERT_TYPES ${ENABLE_AS_CONVERT_TYPES})
endif()

foreach(TY IN LISTS TYPE_LIST)
  set(OUT_FILE "vector_api_${TY}.cpp")
  STRING(REGEX REPLACE ":" "_" OUT_FILE ${OUT_FILE})
  STRING(REGEX REPLACE " " "_" OUT_FILE ${OUT_FILE})

  # Invoke our generator
  # the path to the generated cpp file will be added to TEST_CASES_LIST
  generate_cts_test(TESTS TEST_CASES_LIST
    GENERATOR "generate_vector_api.py"
    OUTPUT ${OUT_FILE}
    INPUT "../common/vector.template"
    EXTRA_ARGS -type "${TY}" -target-enable ${ENABLE_AS_CONVERT_TYPES})
endforeach()

add_cts_test(${TEST_CASES_LIST})
