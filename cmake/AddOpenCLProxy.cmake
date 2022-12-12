# Define OpenCL proxy library which is linked against by other CTS targets.
# The proxy library transitively links to either a fully fleged OpenCL
# library or the bundled OpenCL headers submodule (see below).
add_library(OpenCL_Proxy INTERFACE)
target_compile_definitions(OpenCL_Proxy INTERFACE "CL_TARGET_OPENCL_VERSION=120")
add_library(CTS::OpenCL_Proxy ALIAS OpenCL_Proxy)

# We use the OpenCL headers from the bundled submodule unless OpenCL
# interop testing is enabled, or we are compiling with ComputeCpp,
# which has a dependency on OpenCL (see FindComputeCpp.cmake module).
if(SYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS OR SYCL_IMPLEMENTATION STREQUAL "ComputeCpp")
    find_package(OpenCL 1.2 REQUIRED)
    target_link_libraries(OpenCL_Proxy INTERFACE OpenCL::OpenCL)
else()
    add_submodule_directory(vendor/OpenCL-Headers)
    target_link_libraries(OpenCL_Proxy INTERFACE OpenCL::Headers)
endif()

# Signal that the OpenCL proxy has been configured to allow for ordering checks
# in subsequent configurations.
set(SYCL_CTS_OPENCL_PROXY_CONFIGURED 1)
