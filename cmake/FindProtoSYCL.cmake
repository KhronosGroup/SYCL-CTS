include(FetchContent)
FetchContent_Declare(
    ProtoSYCL
    SOURCE_DIR /ProtoSYCL
)
FetchContent_MakeAvailable(ProtoSYCL)

function(add_sycl_to_target)
    set(options)
    set(one_value_keywords TARGET)
    set(multi_value_keywords SOURCES)
    cmake_parse_arguments(ADD_SYCL
        "${options}"
        "${one_value_keywords}"
        "${multi_value_keywords}"
        ${ARGN}
    )

    target_link_libraries(${ADD_SYCL_TARGET} PUBLIC ProtoSYCL)

endfunction()
