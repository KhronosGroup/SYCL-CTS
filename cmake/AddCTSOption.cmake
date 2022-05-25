set(SYCL_CTS_DETAIL_AVAILABLE_OPTIONS "")

#
# Adds a new CTS option with the given name, description and default value.
#
# The option is made available both as a CMake variable (either OFF or ON)
# and as a preprocessor definition with the same name (either 0 or 1).
#
# Important: The preprocessor macro is always set, regardless of its value.
#            Use `#if <OPTION>` instead of `#ifdef <OPTION>`.
#
# Optional parameters:
#  - WARN_IF_OFF <msg>     Print a warning message if this option is set to OFF.
#
function(add_cts_option option_name option_description option_default)
    cmake_parse_arguments(PARSE_ARGV 3 args "" "WARN_IF_OFF" "")

    option(${option_name} ${option_description} ${option_default})
    list(APPEND SYCL_CTS_DETAIL_AVAILABLE_OPTIONS ${option_name})
    set(SYCL_CTS_DETAIL_AVAILABLE_OPTIONS ${SYCL_CTS_DETAIL_AVAILABLE_OPTIONS} PARENT_SCOPE)

    set("${option_name}_DESCRIPTION" ${option_description} PARENT_SCOPE)

    if(args_WARN_IF_OFF)
        set("${option_name}_WARN_IF_OFF" ${args_WARN_IF_OFF} PARENT_SCOPE)
    endif()

    add_host_and_device_compiler_definitions("-D${option_name}=$<BOOL:${${option_name}}>")
endfunction()

function(print_cts_config_summary)
    set(MSG_STR "\n")
    string(APPEND MSG_STR "====================================\n")
    string(APPEND MSG_STR "  SYCL 2020 Conformance Test Suite\n")
    string(APPEND MSG_STR "       Configuration summary\n")
    string(APPEND MSG_STR "====================================\n\n")

    foreach(opt ${SYCL_CTS_DETAIL_AVAILABLE_OPTIONS})
        string(APPEND MSG_STR " * ${opt}\n   ${${opt}_DESCRIPTION}: ${${opt}}\n\n")
    endforeach()

    message(STATUS ${MSG_STR})

    # Print warnings for disabled options
    foreach(opt ${SYCL_CTS_DETAIL_AVAILABLE_OPTIONS})
        if(NOT ${${opt}} AND ${opt}_WARN_IF_OFF)
            message(WARNING ${${opt}_WARN_IF_OFF})
        endif()
    endforeach()

endfunction()
