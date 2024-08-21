set(TEST_CASES_LIST "")

set(MATH_CAT_WITH_VARIANT common float relational geometric)
set(MATH_VARIANT base)

set(MATH_CAT integer native half)

if(SYCL_CTS_ENABLE_HALF_TESTS)
  list(APPEND MATH_VARIANT half)
endif()
if(SYCL_CTS_ENABLE_DOUBLE_TESTS)
  list(APPEND MATH_VARIANT double)
endif()

set(math_builtin_depends
  "modules/sycl_functions.py"
  "modules/sycl_types.py"
  "modules/test_generator.py"
)

foreach(cat ${MATH_CAT_WITH_VARIANT})
  foreach(var ${MATH_VARIANT})
    if ("${cat}" STREQUAL geometric AND "${var}" STREQUAL half)
      continue()
    endif()
    # Invoke our generator
    # the path to the generated cpp file will be added to TEST_CASES_LIST
    generate_multifile_cts_test(TESTS TEST_CASES_LIST
      GENERATOR "generate_math_builtin.py"
      FILE_PREFIX "math_builtin_${cat}_${var}"
      EXT "cpp"
      INPUT "math_builtin.template"
      EXTRA_ARGS -test ${cat} -variante ${var} -marray true -fragment-size 50
      DEPENDS ${math_builtin_depends}
    )
  endforeach()
endforeach()

foreach(cat ${MATH_CAT})
  # Invoke our generator
  # the path to the generated cpp file will be added to TEST_CASES_LIST
  generate_multifile_cts_test(TESTS TEST_CASES_LIST
    GENERATOR "generate_math_builtin.py"
    FILE_PREFIX "math_builtin_${cat}"
    EXT "cpp"
    INPUT "math_builtin.template"
    EXTRA_ARGS -test ${cat} -marray true -fragment-size 50
    DEPENDS ${math_builtin_depends}
  )
endforeach()

add_cts_test(${TEST_CASES_LIST})
