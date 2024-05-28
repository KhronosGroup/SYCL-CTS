# SYCL 2020 Conformance Test Suite

This is the SYCL Conformance Test Suite for the [Khronos Group
SYCL](https://www.khronos.org/sycl) standard.

The test suite comes in the form of multiple binary executables that must be
compiled before running, a step that requires CMake. All test cases inside the
[`tests`](tests) directory are grouped into *categories*. Compilation produces
executables for each category and one *fat* executable that contains the tests
from all categories.

**Important:** Due to its large size, by default the CTS is compiled in a
lighter configuration that omits some combinations of testing logic for the sake
of reduced compilation and execution time. This can be useful during
development, however is not sufficient to establish conformance. See the
sections on [CMake Configuration Options](#cmake-configuration-options) and
[Generating a Conformance Report](#generating-a-conformance-report) for more
information.

## Configuration & Compilation

To compile the CTS, the following dependencies are required:

- Python 3.7 or higher
- CMake 3.15 or higher
- A SYCL implementation
  - The CTS currently supports DPC++ and hipSYCL
  - See the [AddSYCLExecutable.cmake](cmake/AddSYCLExecutable.cmake) module on
    how to add support for additional SYCL implementations

Configuration and compilation then follow standard CMake procedures. Begin by
cloning this repository and its submodules:

`git clone --recurse-submodules https://github.com/KhronosGroup/SYCL-CTS.git`

Then enter the `SYCL-CTS` folder and configure the build using CMake:

`cmake -S . -B build -DSYCL_IMPLEMENTATION=<DPCPP|hipSYCL>`

See [CMake Configuration Options](#cmake-configuration-options) for additional
configuration options that can be passed here.

Finally, start the compilation:

`cmake --build ./build`

After the compilation has finished, test executables for each category will be
placed in the `build/bin` directory. The `test_all` executable contains tests
for all categories.

### CMake Configuration Options

The CTS can be configured using the following CMake configuration options:

`SYCL_IMPLEMENTATION` (default: None)
 `DPCPP` or `hipSYCL`.

`SYCL_CTS_EXCLUDE_TEST_CATEGORIES` (default: None)
 Optional file specifying a list of test categories to be excluded from the build.

`SYCL_CTS_ENABLE_FULL_CONFORMANCE` (default: `OFF`)
 Enable extended type coverage and testing logic with significantly increased
 compilation and execution time. **This mode is required to establish the
 conformance of a SYCL implementation.**

`SYCL_CTS_VERBOSE_LOG` (default: `OFF`)
 Enable verbose debug-level logging.

`SYCL_CTS_ENABLE_DOUBLE_TESTS` (default: `ON`)
 Enable tests that require double precision floating point capabilities.

`SYCL_CTS_ENABLE_HALF_TESTS` (default: `ON`)
 Enable tests that require half precision floating point capabilities.

`SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS` (default: `ON`)
 Enable tests for legacy SYCL features. Should be switched on for conformance.

`SYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS` (default: `ON`)
 Enable OpenCL interoperability tests.

Additionally, the following SYCL implementation-specific options can be used:

`DPCPP_INSTALL_DIR` (default: None)

`DPCPP_FLAGS` (default: None)
 Set additional compiler flags for DPC++ compiler. This options applies only if `SYCL_IMPLEMENTATION` is set to `DPCPP`.

`DPCPP_TARGET_TRIPLES` (default: None)
 Configures compilation for specified target triple.

`DPCPP_DISABLE_SYCL2020_DEPRECATION_WARNINGS` (default: `ON`)
 Disables warnings about using features deprecated by SYCL 2020.

`DPCPP_SYCL2020_CONFORMANT_APIS` (default: `ON`)
 Enables conformant SYCL 2020 API in DPC++ implementation. Current DPC++ version exposes SYCL 1.2.1 compatible API version by default.

## Running the Test Suite

Each of the executables produced in the `build/bin` directory acts as a
standalone test runner that can be used to launch tests for a particular test
category (or all tests in the case of `build/bin/test_all`).

The ``--device`` argument is used to specify which device to run the tests on.
Selection is based on substring matching of the device name. ECMAScript regular
expression syntax is supported. To get a list of all available devices, use
`--list-devices`.

Please see `<test_executable> --help` for a complete list of available filtering
and output formatting options.

## Generating a Conformance Report

To generate a conformance report, use the `run_conformance_tests.py` script.
This script automates the configuration, compilation and execution of the CTS,
generating a report file `conformance_report.xml`. By default, the script will
enable the `SYCL_CTS_ENABLE_FULL_CONFORMANCE` option, resulting in long
compilation and execution times.

Please see `run_conformance_tests.py --help` for a complete list of available
options.

## Contributing to the CTS

See the [SYCL CTS Developer Documentation](docs).
