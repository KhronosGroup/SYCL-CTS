include(FetchContent)
FetchContent_Declare(
    ProtoSYCL
    GIT_REPOSITORY https://github.com/0x12CC/ProtoSYCL.git
    GIT_TAG main
)
FetchContent_MakeAvailable(ProtoSYCL)
FetchContent_GetProperties(ProtoSYCL BINARY_DIR PROTOSYCL_BINARY_DIR)
set(CMAKE_CXX_COMPILER "${PROTOSYCL_BINARY_DIR}/sycl++")

# This is needed since ProtoSYCL must be the first target built. It ensures
# sycl++ is available and can be used to build the other targets.
add_dependencies(OpenCL_Proxy ProtoSYCL)

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
