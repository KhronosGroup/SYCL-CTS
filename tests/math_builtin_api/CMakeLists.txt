set(TEST_CASES_LIST "")

set(MATH_CAT_WITH_VARIANT common float relational)
set(MATH_VARIANT base double half)

set(MATH_CAT integer geometric native half)

set(math_builtin_depends
  "modules/sycl_functions.py"
  "modules/sycl_types.py"
  "modules/test_generator.py"
)

foreach(cat ${MATH_CAT_WITH_VARIANT})
  foreach(var ${MATH_VARIANT})
    # Invoke our generator
    # the path to the generated cpp file will be added to TEST_CASES_LIST
    generate_cts_test(TESTS TEST_CASES_LIST
      GENERATOR "generate_math_builtin.py"
      OUTPUT "math_builtin_${cat}_${var}.cpp"
      INPUT "math_builtin.template"
      EXTRA_ARGS -test ${cat} -variante ${var}
      DEPENDS ${math_builtin_depends}
    )
  endforeach()
endforeach()

foreach(cat ${MATH_CAT})
  # Invoke our generator
  # the path to the generated cpp file will be added to TEST_CASES_LIST
  generate_cts_test(TESTS TEST_CASES_LIST
    GENERATOR "generate_math_builtin.py"
    OUTPUT "math_builtin_${cat}.cpp"
    INPUT "math_builtin.template"
    EXTRA_ARGS -test ${cat}
    DEPENDS ${math_builtin_depends}
  )
endforeach()

add_cts_test(${TEST_CASES_LIST})
