set(TEST_CASES_LIST "")

set(MATH_CAT_WITH_VARIANT common float relational)
set(MATH_VARIANT base double half)

set(MATH_CAT integer geometric native half)

foreach(cat ${MATH_CAT_WITH_VARIANT})
  foreach(var ${MATH_VARIANT})
    # Invoke our generator
    # the path to the generated cpp file will be added to TEST_CASES_LIST
    generate_cts_test(TESTS TEST_CASES_LIST
      GENERATOR "generate_math_builtin.py"
      OUTPUT "math_builtin_${cat}_${var}.cpp"
      INPUT "math_builtin.template"
      EXTRA_ARGS -test ${cat} -variante ${var})
  endforeach()
endforeach()

foreach(cat ${MATH_CAT_WITH_VARIANT})
  # Invoke our generator
  # the path to the generated cpp file will be added to TEST_CASES_LIST
  generate_cts_test(TESTS TEST_CASES_LIST
    GENERATOR "generate_math_builtin.py"
    OUTPUT "math_builtin_${cat}.cpp"
    INPUT "math_builtin.template"
    EXTRA_ARGS -test ${cat})
endforeach()

add_cts_test(${TEST_CASES_LIST})
