=================================
SYCL 2020 Conformance Test Suite
=================================

This is the SYCL Conformance Test Suite for the Khronos Group SYCL standard.

For more information: https://www.khronos.org/sycl


Introduction
------------

This conformance test suite is divided into two parts, a binary
executable containing all of the tests and a python user interface.
It is intended that the user only runs the test suite via the Python
front-end.

The test executable must be compiled before running the test suite, a
step which requires CMake.  Executables are produced for each category
and one 'fat' executable that contains the tests from all categories.


Compilation and execution prerequisites
---------------------------------------

- Python 3.7
- CMake 3.13
- A conformant implementation of OpenCL
- An implementation of SYCL
  - Note: the SYCL implementation may require additional dependencies

The SYCL conformance test suite assumes that an underlying OpenCL
implementation has passed the Khronos OpenCL CTS.


Building on Linux
-----------------

- Checkout the SYCL test suite repository.

- Run CMake
  - point the source directory to the root of the ``sycl-cts`` folder
  - configure the generator to use makefiles or ninja
  - set the build directory to a ``build`` folder inside the
    ``sycl-cts`` folder
  - set the CMake parameters
- Run make inside the ``build`` folder

After compilation, the test executables will be placed in the
``sycl-cts`` ``build/bin`` directory.  The ``test_all`` executable
contains all tests in the suite.

You can also compile each tests separately by giving target to
the make command. For instance::

  $ make test_header


Building on Windows
-------------------

- Checkout the SYCL test suite repository.

- Run CMake
  - point the source directory to the root of the ``sycl-cts`` folder
  - select "Visual Studio 14 2015 Win64" generator
  - set the build directory to a ``build`` folder inside the ``sycl-cts`` folder
  - set the CMake parameters
- Open the generated solution in Visual Studio 2015 and rebuild all

  After compilation, the test executables will be placed in the
  ``sycl-cts`` ``build/bin`` directory.  The ``test_all`` executable
  contains all tests in the suite.


CMake flags
-----------

When configuring CMake, it is possible to use these flags:

``COMPUTECPP_INSTALL_DIR``
  Required only if your SYCL implementation is ComputeCpp.

``SYCL_CTS_TEST_FILTER``
  Specify which filter to use when building tests.

``SYCL_CTS_ENABLE_FULL_CONFORMANCE``
  Enable extensive coverage with huge compilation and execution time.
  This mode is switched off by default. Should be swithed on for conformance.

``SYCL_CTS_VERBOSE_LOG``
  Enable debug-level logs with the possibly oververbose output.
  This mode is switched off by default.

``HOST_COMPILER_FLAGS``
  Flags that will be passed to the host compiler.

``DEVICE_COMPILER_FLAGS``
  Flags that will be passed to the device compiler.


Launching the test suite
------------------------

The SYCL test suite can be launched via the following command::

    $ python runtests.py --help

    usage: runtests.py [-h] [-b BINPATH] [--csvpath CSVPATH] [--list]
                       [-j JUNIT] [--device DEVICE]

    Khronos SYCL CTS

    optional arguments:

      -h, --help            show this help message and exit
      -b BINPATH, --binpath BINPATH
                            specify path to the cts executable file
      --csvpath CSVPATH     specify path to csv file for filtering tests
      --list                list all tests in a test binary
      -j JUNIT, --junit JUNIT
                            specify output path for a junit xml file
      -p PLATFORM, --platform PLATFORM
                            The platform to run on (where PLATFORM can be
                            host, intel, amd)
      -d DEVICE, --device DEVICE
                            The device to run on (where DEVICE can be host,
                            opencl_cpu, opencl_gpu, opencl_accelerator)

The ``--binpath`` argument is mandatory and must point to one of the CTS
test executables built in the previous step.

An optional CSV file can be given which can be used to narrow the range
of tests that will be executed.

The filters work using the principal of partial string matching.  Any
test in a CTS executable with a name that begins with one of the items
in CSV file will be scheduled to be run.  Those tests that don't match
will not be run.

In the future the CSV file will also be used to specify timeout values
on a per-test basis.

The ``--list`` argument can be used to examine all of the tests that are
stored in a test executable.  For instance::

    $ python runtests.py -b build\bin\test_context.exe --list

    3 tests in executable

      . context_api

      . context_constructors

      . context_getinfo

Passing the ``--junit`` option will output test results in `junit` format
when the test suite has finished executing.

The ``--device`` argument can be used to specify which device to run the
tests on. Selection is based on substring matching of the device name.
ECMAScript regular expression syntax is supported.

The following command will start a typical test run::

    $ python runtests.py --binpath tests/common/test_all

During testing any fails will be reported with details about the failure.
The following failure importantly shows the source file containing the
test and the line number that signaled the failure::

    platform_api:

     ?   note: sycl exception caught
     ?   note: what - Failed to get platform information.
     + result: fail
     !   file: ../../tests/platform/platform_api.cpp
     !  built: Aug 22 2017, 18:06:45
     !   line: 96

After the test suite is finished a summary is produced helping programmers
quickly identify failures and conformance rate::

    16 tests ran in total
     - passed : 13
     - failed : 1
       + platform_api
     - skipped: 2
     - 81% pass rate

This report should be packaged with the run tests and sent to Khronos for
conformance submission.

Conformance submission
----------------------

The conformance submission requires the use of the ``core.csv`` filter alongside
with the ``run_conformance_tests.py`` script. Please, note that this script will
switch the ``SYCL_CTS_ENABLE_FULL_CONFORMANCE`` option on.
