# include SYCL implementation cmake
include(cmake/syclone/syclone.cmake)
# ------------------


# BUILD_SYCL FUNCTION
# Builds a SYCL program, compiling multiple SYCL test case source files into a test executable, invoking a single-source/device compiler
# Parameters are:
#   - exe_name               Name of the test executable
#   - test_main              Path to the main.cpp for the test executable
#   - test_cases_list        List of SYCL test case source files to be built into the test executable
#   - destination_exe_path   Path where the test executable will be placed
#   - destination_stub_path  Path where the intermediate stub(spir, ll, etc.) files will be placed
function(BUILD_SYCL exe_name test_main test_cases_list destination_exe_path destination_stub_path)
    
    set(BUILD_SYCL_NOT_FOUND_ERROR
    "\n\nThe build_sycl() function implementation could not be found!\nPlease include the cmake module defining the build_sycl() function for the SYCL implementation in cmake/build_sycl.cmake.\n\n"
    "BUILD_SYCL FUNCTION\n"
    "Builds a SYCL program, compiling multiple SYCL test case source files into a test executable, invoking a single-source/device compiler.\n"
    "Parameters are:\n"
    "  exe_name              Name of the test executable\n"
    "  test_main             Path to the main.cpp for the test executable\n"
    "  test_cases_list       List of SYCL test case source files to be built into the test executable\n"
    "  destination_exe_path  Path where the test executable will be placed\n"
    "  destination_stub_path Path where the intermediate stub(spir, ll, etc.) files will be placed\n\n")

    if(NOT COMMAND build_sycl_implementation)
        message(FATAL_ERROR ${BUILD_SYCL_NOT_FOUND_ERROR})  
    endif()
    
    # invoke build_sycl()
    build_sycl_implementation("${exe_name}" "${test_main}" "${test_cases_list}" "${destination_exe_path}" "${destination_stub_path}")
    
endfunction()
# ------------------
