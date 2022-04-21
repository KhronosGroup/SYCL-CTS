set (KNOWN_SYCL_IMPLEMENTATIONS "Intel_SYCL;DPCPP;ComputeCpp;hipSYCL")
if (NOT ${SYCL_IMPLEMENTATION} IN_LIST KNOWN_SYCL_IMPLEMENTATIONS)
    message(FATAL_ERROR
        "The SYCL CTS requires specifying a SYCL implementation with "
        "-DSYCL_IMPLEMENTATION=[Intel_SYCL,DPCPP;ComputeCpp,hipSYCL]")
endif()

find_package(${SYCL_IMPLEMENTATION} REQUIRED)

if(NOT TARGET SYCL::SYCL)
    message(FATAL_ERROR
        "The SYCL CTS requires a CMake Target with the name `SYCL::SYCL` to be"
        "present. It should provide all the include directories, compiler options"
        "and definitions to compile code that is dependent on SYCL, but does not"
        "contain device code."
    )
endif()

if(NOT COMMAND add_sycl_executable_implementation)
    message(FATAL_ERROR
        "The add_sycl_executable_implementation() function implementation could not be found! "
        "Please include the CMake module defining the add_sycl_executable_implementation() "
        "function for the SYCL implementation in cmake/AddSYCLExecutable.cmake "
        "or add it to your CMake Find/Config module.\n"
        "  add_sycl_executable_implementation(\n"
        "     NAME <name>\n"
        "     OBJECT_LIBRARY <object_library_name>\n"
        "     TESTS <sources>...\n"
        "  )\n"
        "  Builds a SYCL program, compiling multiple SYCL test case source files into a test executable, invoking a single-source/device compiler.\n"
        "  The options are:\n"
        "    NAME             Name of the test executable\n"
        "    OBJECT_LIBRARY   Name of the object library of all the compiled test cases\n"
        "    TESTS            List of SYCL test case source files to be built into the test executable\n"
    )
endif()

# add_sycl_executable function
# Builds a SYCL program, compiling multiple SYCL test case source files into a test executable, invoking a single-source/device compiler
# Parameters are:
#   - NAME             Name of the test executable
#   - OBJECT_LIBRARY   Name of the object library of all the compiled test cases
#   - TESTS            List of SYCL test case source files to be built into the test executable
function(add_sycl_executable)
    cmake_parse_arguments(args
        ""
        "NAME;OBJECT_LIBRARY"
        "TESTS"
        ${ARGN})

    add_sycl_executable_implementation(
        NAME           "${args_NAME}"
        OBJECT_LIBRARY "${args_OBJECT_LIBRARY}"
        TESTS          "${args_TESTS}")
endfunction()
