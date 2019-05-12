set(TEST_CASES_LIST "")

set(TYPE_LIST "")
get_std_type(TYPE_LIST)
get_opencl_type(TYPE_LIST)

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
    EXTRA_ARGS -type "${TY}")
endforeach()

add_cts_test(${TEST_CASES_LIST})
