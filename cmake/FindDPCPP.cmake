if(${CMAKE_CXX_COMPILER_ID} MATCHES "GNU" OR
   ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    find_program(DPCPP_CXX_EXECUTABLE NAMES icpx clang++ dpcpp
        HINTS ${DPCPP_INSTALL_DIR}
        PATH_SUFFIXES bin)
else()
    # Remove /machine: option which is not supported by clang-cl
    string(REPLACE "/machine:x64" "" CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS}")
    # Remove /subsystem option which is not supported by clang-cl
    string(REPLACE "/subsystem:console" "" CMAKE_CREATE_CONSOLE_EXE
        "${CMAKE_CREATE_CONSOLE_EXE}")
    string(REPLACE "/subsystem:console" "" CMAKE_CXX_CREATE_CONSOLE_EXE
        "${CMAKE_CXX_CREATE_CONSOLE_EXE}")
    find_program(DPCPP_CXX_EXECUTABLE NAMES icx clang-cl dpcpp-cl
        HINTS ${DPCPP_INSTALL_DIR}
        PATH_SUFFIXES bin)
endif()

# Set SYCL compilation mode, SYCL 2020 standard version and user provided flags
set(DPCPP_FLAGS "-fsycl;-sycl-std=2020;${DPCPP_FLAGS}")

# -fsycl-id-queries-fit-in-int is an optimization enabled by default, but
# adds non-conformant behavior that limits the number of work-items in an
# invocation of a kernel, so we disable this behavior here.
set(DPCPP_FLAGS "${DPCPP_FLAGS};-fno-sycl-id-queries-fit-in-int")

# Set target triple(s) if specified
if(DEFINED DPCPP_TARGET_TRIPLES)
    set(DPCPP_FLAGS "${DPCPP_FLAGS};-fsycl-targets=${DPCPP_TARGET_TRIPLES};")
    message(STATUS "DPC++ compiling to triples: ${DPCPP_TARGET_TRIPLES}")
    if(${DPCPP_TARGET_TRIPLES} MATCHES ".*-nvidia-cuda-.*")
        add_definitions(-DSYCL_CTS_INTEL_PI_CUDA)
    endif()
endif()
message(STATUS "DPC++ compiler flags: `${DPCPP_FLAGS}`")

# Explicitly set fp-model to precise to produce reliable results for floating
# point operations.
if(WIN32)
    set(DPCPP_FP_FLAG "/fp:precise")
else()
    set(DPCPP_FP_FLAG "-ffp-model=precise")
endif()
set(CMAKE_CXX_FLAGS "${DPCPP_FP_FLAG} ${CMAKE_CXX_FLAGS}")

# Disable range rounding feature to reduce # of SYCL kernels.
set(CMAKE_CXX_FLAGS "-D__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__ ${CMAKE_CXX_FLAGS}")

# Set flag to allow linking of large device code files. This option is currently
# not available on Windows.
if(NOT WIN32)
    set(CMAKE_CXX_LINK_FLAGS "-flink-huge-device-code ${CMAKE_CXX_LINK_FLAGS}")
endif()

option(DPCPP_DISABLE_SYCL2020_DEPRECATION_WARNINGS
  "Disable SYCL 2020 deprecation warnings" ON)
if(DPCPP_DISABLE_SYCL2020_DEPRECATION_WARNINGS)
    set(CMAKE_CXX_FLAGS "-DSYCL2020_DISABLE_DEPRECATION_WARNINGS ${CMAKE_CXX_FLAGS}")
endif()

option(DPCPP_SYCL2020_CONFORMANT_APIS
  "Comply with the SYCL 2020 specification" ON)
if(DPCPP_SYCL2020_CONFORMANT_APIS)
    set(CMAKE_CXX_FLAGS "-DSYCL2020_CONFORMANT_APIS ${CMAKE_CXX_FLAGS}")
endif()

cmake_dependent_option(DPCPP_ENABLE_PREVIEW_CHANGES
  "Enable preview changes for DPC++ Compiler" ON
  "DPCPP_SYCL2020_CONFORMANT_APIS" OFF)
if(DPCPP_ENABLE_PREVIEW_CHANGES)
    set(CMAKE_CXX_FLAGS "-fpreview-breaking-changes ${CMAKE_CXX_FLAGS}")
endif()

add_library(DPCPP::Runtime INTERFACE IMPORTED GLOBAL)
set_target_properties(DPCPP::Runtime PROPERTIES
  INTERFACE_COMPILE_OPTIONS   "${DPCPP_FLAGS}"
  INTERFACE_LINK_OPTIONS      "${DPCPP_FLAGS}")

set(CMAKE_CXX_COMPILER ${DPCPP_CXX_EXECUTABLE})
# Use DPC++ compiler instead of default linker for building SYCL application
set(CMAKE_CXX_LINK_EXECUTABLE "${DPCPP_CXX_EXECUTABLE} <FLAGS> <OBJECTS> -o <TARGET> \
    <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <LINK_LIBRARIES>")

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

    target_link_libraries(${ADD_SYCL_TARGET} PUBLIC DPCPP::Runtime)

endfunction()
