set(TEST_CASES_LIST "")

set(VEC_TYPE_LIST "")
list(APPEND VEC_TYPE_LIST
  "std::int8_t"
  "std::uint8_t"
  "std::int16_t"
  "std::uint16_t"
  "std::int32_t"
  "std::uint32_t"
  "std::int64_t"
  "std::uint64_t"
  "float"
  "double"
  "sycl::half"
)

half_double_filter(VEC_TYPE_LIST)

foreach(TY IN LISTS VEC_TYPE_LIST)
  set(OUT_FILE "vector_alias_${TY}.cpp")
  STRING(REGEX REPLACE ":" "_" OUT_FILE ${OUT_FILE})
  STRING(REGEX REPLACE " " "_" OUT_FILE ${OUT_FILE})

  # Invoke our generator
  # the path to the generated cpp file will be added to TEST_CASES_LIST
  generate_cts_test(TESTS TEST_CASES_LIST
    GENERATOR "generate_vector_alias.py"
    OUTPUT ${OUT_FILE}
    INPUT "../common/vector.template"
    EXTRA_ARGS -type "${TY}")
endforeach()

add_cts_test(${TEST_CASES_LIST})
