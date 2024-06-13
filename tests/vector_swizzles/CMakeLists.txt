set(TEST_CASES_LIST "")

set(TYPE_LIST "")
set(NUM_BATCHES "32")
get_std_type(TYPE_LIST)
get_no_vec_alias_type(TYPE_LIST)
get_fixed_width_type(TYPE_LIST)

half_double_filter(TYPE_LIST)

foreach(TY IN LISTS TYPE_LIST)
    foreach(BATCH_INDEX RANGE 1 ${NUM_BATCHES})
        set(OUT_FILE "vector_swizzles_${TY}_batch_${BATCH_INDEX}.cpp")
        STRING(REGEX REPLACE ":" "_" OUT_FILE ${OUT_FILE})
        STRING(REGEX REPLACE " " "_" OUT_FILE ${OUT_FILE})
        STRING(REGEX REPLACE "std__" "" OUT_FILE ${OUT_FILE})

        # Invoke our generator
        # the path to the generated cpp file will be added to TEST_CASES_LIST
        generate_cts_test(TESTS TEST_CASES_LIST
        GENERATOR "generate_vector_swizzles.py"
        OUTPUT ${OUT_FILE}
        INPUT "../common/vector_swizzles.template"
        EXTRA_ARGS -type "${TY}" -num_batches ${NUM_BATCHES} -batch_index ${BATCH_INDEX})
    endforeach()
endforeach()

add_cts_test(${TEST_CASES_LIST})
