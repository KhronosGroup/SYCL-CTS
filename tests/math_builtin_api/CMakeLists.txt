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

set(SYCL_CTS_MATH_BUILTIN_N_SPLITS 16 CACHE STRING 
    "The number of times to divide each math_builtin_api test")

foreach(cat ${MATH_CAT_WITH_VARIANT})
  foreach(var ${MATH_VARIANT})
    foreach(split_index RANGE 1 ${SYCL_CTS_MATH_BUILTIN_N_SPLITS})
      if ("${cat}" STREQUAL geometric AND "${var}" STREQUAL half)
        continue()
      endif()
      # Invoke our generator
      # the path to the generated cpp file will be added to TEST_CASES_LIST
      math(EXPR split_index_0 "${split_index} - 1")
      generate_cts_test(TESTS TEST_CASES_LIST
        GENERATOR "generate_math_builtin.py"
        OUTPUT "math_builtin_${cat}_${var}_${split_index}_${SYCL_CTS_MATH_BUILTIN_N_SPLITS}.cpp"
        INPUT "math_builtin.template"
        EXTRA_ARGS -test ${cat} -variante ${var} -marray true 
                   -i ${split_index_0} -n ${SYCL_CTS_MATH_BUILTIN_N_SPLITS}
        DEPENDS ${math_builtin_depends}
      )
    endforeach()
  endforeach()
endforeach()

foreach(cat ${MATH_CAT})
  foreach(split_index RANGE 1 ${SYCL_CTS_MATH_BUILTIN_N_SPLITS})
    # Invoke our generator
    # the path to the generated cpp file will be added to TEST_CASES_LIST
    math(EXPR split_index_0 "${split_index} - 1")
    generate_cts_test(TESTS TEST_CASES_LIST
      GENERATOR "generate_math_builtin.py"
      OUTPUT "math_builtin_${cat}_${split_index}_${SYCL_CTS_MATH_BUILTIN_N_SPLITS}.cpp"
      INPUT "math_builtin.template"
      EXTRA_ARGS -test ${cat} -marray true 
                 -i ${split_index_0} -n ${SYCL_CTS_MATH_BUILTIN_N_SPLITS}
      DEPENDS ${math_builtin_depends}
    )
  endforeach()
endforeach()

add_cts_test(${TEST_CASES_LIST})
