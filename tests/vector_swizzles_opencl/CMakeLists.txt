set(TEST_CASES_LIST "")

set(TYPE_LIST "")
get_opencl_type(TYPE_LIST)

foreach(TY IN LISTS TYPE_LIST)
  set(OUT_FILE "vector_swizzles_opencl_${TY}.cpp")
  STRING(REGEX REPLACE ":" "_" OUT_FILE ${OUT_FILE})
  STRING(REGEX REPLACE " " "_" OUT_FILE ${OUT_FILE})
  STRING(REGEX REPLACE "cl__sycl__" "" OUT_FILE ${OUT_FILE})

  # Invoke our generator
  # the path to the generated cpp file will be added to TEST_CASES_LIST
  generate_cts_test(TESTS TEST_CASES_LIST
    GENERATOR "generate_vector_swizzles_opencl.py"
    OUTPUT ${OUT_FILE}
    INPUT "vector_swizzles_opencl.template"
    EXTRA_ARGS -type "${TY}")
endforeach()

add_cts_test(${TEST_CASES_LIST})
