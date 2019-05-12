# ************************************************************************
#
#   SYCL Conformance Test Suite
#
#   Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
#
# ************************************************************************
#
# FindOpenCLTargets
# -------
#
# Try to find OpenCL. This module serves as a backport of the newer FindOpenCL
# with imported targets from CMake 3.7 and up
#
# This will define the following variables:
#
#   OpenCL_FOUND          - True if OpenCL was found
#   OpenCL_VERSION_STRING - Highest supported OpenCL version (eg. 1.2)
#   OpenCL_VERSION_MAJOR  - The major version of the OpenCL implementation
#   OpenCL_VERSION_MINOR  - The minor version of the OpenCL implementation
#
# and the following imported targets:
#
#   OpenCL::OpenCL   - The path to the OpenCL command.

if(NOT ${CMAKE_VERSION} VERSION_LESS 3.7)
  message(DEPRECATION "Use the official FindOpenCL module if CMake <3.7 "
                      "support can be dropped")
endif()

if(TARGET OpenCL::OpenCL)
  return()
endif()

find_package(OpenCL)
set(OpenCLTargets_FOUND ${OpenCL_FOUND})
set(OpenCLTargets_VERSION ${OpenCL_VERSION})
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenCLTargets
                                  FOUND_VAR OpenCLTargets_FOUND
                                  REQUIRED_VARS OpenCL_INCLUDE_DIRS
                                                OpenCL_LIBRARIES
                                  VERSION_VAR OpenCL_VERSION)

# If we're using a newer CMake version there will already be a target defined
if(NOT TARGET OpenCL::OpenCL)
  add_library(OpenCL::OpenCL UNKNOWN IMPORTED)
  set_property(TARGET OpenCL::OpenCL PROPERTY
               IMPORTED_LOCATION ${OpenCL_LIBRARY})
  set_property(TARGET OpenCL::OpenCL PROPERTY
               INTERFACE_INCLUDE_DIRECTORIES ${OpenCL_INCLUDE_DIRS})
  set_property(TARGET OpenCL::OpenCL PROPERTY
               INTERFACE_LINK_LIBRARIES ${OpenCL_LIBRARIES})
endif()
