set(TEST_CASES_LIST "")

set(TYPE_LIST "")

# The vector swizzle tests can become too large if we dont split them into chunks or batches
# and this can hurt compilation times or even cause it to fail on machines with 16 GB of RAM or less.
# Therefore, we output the tests in batches in order to instead generate a larger number of files
# which are individually much smaller which in turn speeds up compilation when using multiple threads.
# It also reduces the memory footprint of the tests. 
# Some experimentation has shown 32 to be a reasonable value for the number of chunks/batches 
# where the compilation suceeds under constrained space resorces and we don't produce a huge number of files.
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
